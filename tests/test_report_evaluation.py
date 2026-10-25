"""Report evaluator contracts and failure boundaries; no paid calls or production DB."""
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.core.judge import LLMJudge
from src.core.report_evaluation import report_contract, validate_requirements, normalize_evidence
from evaluation.experiments import validate_dataset, summarize
from evaluation.experiments import run_experiment, export_review_queue
from src.orchestrator.schemas import ResearchReport


def response(quote="正确结论", verdict="correct", scores=None):
    return {"dimensions": {key: {"score": (scores or {}).get(key, 4), "reason": "对照证据判断", "issues": []}
            for key in ("factual_accuracy", "logical_consistency", "citation_quality", "comprehensiveness", "clarity")},
            "facts": [{"id": "fact_1", "verdict": verdict, "quote": quote,
                       "reason": "对照参考事实", "evidence_ids": ["source_1"]}]}


class ReportContractTests(unittest.TestCase):
    def test_count_is_explicit_and_does_not_ignore_urls(self):
        r = report_contract("中 文\nA。 https://x.test", {"max_chars": 4})
        self.assertFalse(r["constraint_pass"])
        self.assertEqual(r["char_count"], len("中文A。https://x.test"))

    def test_citations_resolve_and_urls_are_not_truth_checks(self):
        for text in ("结论[1]。\n[1]: https://x.test", "结论[1](https://x.test)", "结论[1]。\n[1] 来源 https://x.test"):
            self.assertTrue(report_contract(text, {"citation_required": True, "allowed_urls": ["https://x.test"]})["constraint_pass"])
        self.assertFalse(report_contract("结论[9] https://x.test")["constraint_pass"])
        self.assertFalse(report_contract("结论 https://other.test", {"allowed_urls": ["https://x.test"]})["constraint_pass"])

    def test_sections_are_headings_and_empty_report_fails(self):
        self.assertFalse(report_contract("正文提到结论", {"required_sections": ["结论"]})["constraint_pass"])
        self.assertTrue(report_contract("## 结论\n内容", {"required_sections": ["结论"]})["constraint_pass"])
        self.assertFalse(report_contract("  \n")["constraint_pass"])

    def test_invalid_spec_fails_before_experiment(self):
        for requirements in ({"max_chars": True}, {"max_chars": -1}, {"min_chars": 4, "max_chars": 3},
                             {"evidence_mode": "guess"}, {"citation_required": "true"}, {"max_char": 3}):
            with self.subTest(requirements=requirements), self.assertRaises(ValueError):
                validate_requirements(requirements)
        with self.assertRaises(ValueError):
            validate_dataset({"examples": [{"id": "q", "query": "q", "requirements": {"max_chars": -1}}]})

    def test_evidence_keeps_complete_text_and_stable_hash(self):
        docs = normalize_evidence([{"url": "https://x.test", "text": "证据" * 10000}])
        self.assertEqual(len(docs[0]["text"]), 20000)
        self.assertEqual(docs, normalize_evidence(docs))
        with self.assertRaises(ValueError):
            normalize_evidence([{"id": "a", "url": "u", "text": "x"}, {"id": "a", "url": "v", "text": "y"}])


class ReportRubricTests(unittest.TestCase):
    def judge(self, output=None, **kwargs):
        judge = LLMJudge(**kwargs)
        judge._policy = Mock(return_value={"content": json.dumps(output or response(), ensure_ascii=False)})
        return judge

    def evaluate(self, judge, text="正确结论", **kwargs):
        return judge.score_report(text, "q", {"核心": "正确结论"},
            kwargs.pop("requirements", {"evidence_mode": "fixed_snapshot"}),
            [{"url": "https://evidence.example/a", "text": "正确结论"}], **kwargs)

    def test_fixed_evidence_has_no_authority_penalty_and_quote_offsets_are_real(self):
        judge = self.judge()
        r = self.evaluate(judge, "开头\n正确结论")
        self.assertTrue(r["report_pass"])
        self.assertFalse(r["source_authority_applicable"])
        self.assertEqual(r["facts"][0]["location"], {"start": 3, "end": 7})
        prompt = judge._policy.call_args.args[0][-1]["content"]
        self.assertIn("禁止因材料是人工教学样例", prompt)
        self.assertIn("正确结论", prompt)

    def test_good_quality_does_not_hide_failed_constraint(self):
        r = self.evaluate(self.judge(), requirements={"max_chars": 2})
        self.assertEqual(r["average"], 10)
        self.assertTrue(r["quality_pass"])
        self.assertFalse(r["constraint_pass"])
        self.assertFalse(r["report_pass"])

    def test_reference_reversal_cannot_get_perfect_score(self):
        r = self.evaluate(self.judge(response(verdict="incorrect")))
        self.assertEqual(r["reference_fact_contradictions"], 1)
        self.assertEqual(r["reference_fact_coverage"], 0)
        self.assertLessEqual(r["dimensions"]["factual_accuracy"]["score"], 2.5)
        self.assertFalse(r["report_pass"])

    def test_long_report_keeps_tail_and_contradiction_beats_correct_mention(self):
        judge = LLMJudge(chunk_chars=4)
        judge._policy = Mock(side_effect=[{"content": json.dumps(response("正确结论"))},
                                          {"content": json.dumps(response("错误结论", "incorrect"))}])
        r = self.evaluate(judge, "正确结论错误结论")
        self.assertEqual(r["reviewed_chars"], 8)
        self.assertEqual(r["facts"][0]["verdict"], "incorrect")
        self.assertEqual(r["facts"][0]["location"], {"start": 4, "end": 8})
        self.assertEqual(len(r["facts"][0]["occurrences"]), 2)

    def test_malformed_quotes_sources_scores_and_fact_coverage_fail_explicitly(self):
        bad = []
        x = response("不存在原文"); bad.append(x)
        x = response(); x["facts"][0]["evidence_ids"] = ["invented_source"]; bad.append(x)
        x = response(); x["dimensions"]["clarity"]["score"] = True; bad.append(x)
        x = response(); x["dimensions"]["clarity"]["score"] = 5; bad.append(x)
        x = response(); x["facts"] = []; bad.append(x)
        x = response(); x["facts"] *= 2; bad.append(x)
        x = response(); x["facts"][0]["verdict"] = "missing"; bad.append(x)
        for output in bad:
            with self.subTest(output=output):
                r = self.evaluate(self.judge(output))
                self.assertIn("error", r)
                self.assertNotIn("average", r)

    def test_review_budget_does_not_silently_cut_evidence_or_body(self):
        judge = self.judge(chunk_chars=4, max_report_chars=8)
        self.assertIn("error", self.evaluate(judge, "正确结论" * 3))
        self.assertIn("error", self.evaluate(judge, max_evidence_chars=1))
        judge._policy.assert_not_called()

    def test_no_reference_facts_is_not_zero_coverage(self):
        output = response(); output["facts"] = []
        r = self.judge(output).score_report("报告", "q")
        self.assertIsNone(r["reference_fact_coverage"])
        self.assertFalse(r["evidence_available"])

    def test_summary_separates_acceptance_quality_and_completion(self):
        rows = [{"variant": "full", "question_id": "q", "status": "success", "composite_score": .8,
                 "elapsed_seconds": 1, "judge_status": "success", "constraint_pass": False, "report_pass": False,
                 "judge": {"average": 9, "reference_fact_coverage": 1}}]
        r = summarize(rows)["variants"]["full"]
        self.assertEqual(r["completion_rate"], 1)
        self.assertEqual(r["average_judge_score"], 9)
        self.assertEqual(r["constraint_pass_rate"], 0)
        self.assertEqual(r["report_acceptance_rate"], 0)


class ReportIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_blue_inherits_role_budget_for_full_report_repair(self):
        from src.adversarial.blue_agent import BlueAgent
        from src.adversarial.verdict import Issue, Severity, Dimension
        observed = []
        class Policy:
            max_tokens = 8192
            async def acall(self, messages):
                observed.append(self.max_tokens)
                return {"content": json.dumps({"fixed_content": "这是一份修复后的完整报告内容。", "changes": []})}
        policy = Policy()
        result = await BlueAgent(policy)._fix_single_issue(ResearchReport(query="q", content="原报告"),
            Issue(Severity.MAJOR, Dimension.FACTUAL, "修复事实错误"))
        self.assertTrue(result.success)
        self.assertEqual(observed, [8192])
        self.assertEqual(policy.max_tokens, 8192)

    async def test_planner_retries_over_budget_and_fallback_cannot_bypass_limit(self):
        from src.planner.planner import Planner, PlanParseError
        task = lambda name: {"task_id": name, "task_type": "search", "description": "检索相关证据材料", "dependencies": [], "context_keys": [], "timeout_seconds": 30, "priority": 1, "expected_type": "factual", "search_hints": ["q"]}
        too_many = {"sub_tasks": [task("a"), task("b"), task("c")]}
        bounded_plan = {"sub_tasks": [task("a"), task("b")]}
        class Policy:
            tools = None
            def __init__(self):
                self.calls = 0
            async def acall(self, messages):
                self.calls += 1
                return {"content": json.dumps(too_many if self.calls == 1 else bounded_plan)}
        policy = Policy(); planner = Planner(policy, search_calls=0, max_sub_tasks=2)
        self.assertNotIn("generate 6-10", planner._build_prompt("q", ""))
        self.assertIn("Generate 1 to 2", planner._build_prompt("q", ""))
        self.assertEqual(len(await planner.generate_plan("q")), 2)
        self.assertEqual(policy.calls, 2)
        with self.assertRaises(PlanParseError):
            planner._parse_plan(json.dumps(too_many))
        with self.assertRaises(PlanParseError):
            planner._build_dag_from_data(too_many)

    async def test_red_respects_policy_budget_and_parses_plain_dictionary(self):
        from src.adversarial.red_agent import RedAgent
        class Policy:
            max_tokens = 8192
            def __init__(self):
                self.observed = []
            async def acall(self, messages):
                self.observed.append(self.max_tokens)
                return {"content": json.dumps({k: {"score": 8, "issues": []} for k in
                    ("factual_accuracy", "hallucination", "logical", "source_credibility", "coverage")})}
        policy = Policy()
        result = await RedAgent(policy).attack(ResearchReport(query="q", content="正文"))
        self.assertEqual(policy.observed, [8192] * 3)
        self.assertEqual(len(result.dimension_scores), 5)
        self.assertEqual(set(result.dimension_scores.values()), {8})
        self.assertEqual(policy.max_tokens, 8192)

    async def test_red_empty_outputs_fail_without_fabricated_midpoint_verdict(self):
        from src.adversarial.red_agent import RedAgent
        class Policy:
            max_tokens = 8192
            async def acall(self, messages):
                return {"content": "", "finish_reason": "length"}
        with self.assertRaisesRegex(ValueError, "failed schema validation"):
            await RedAgent(Policy()).attack(ResearchReport(query="q", content="正文"))

    async def test_blue_repair_budget_keeps_high_priority_and_marks_unresolved_items(self):
        from src.adversarial.blue_agent import BlueAgent
        from src.adversarial.verdict import Issue, Severity, Dimension, RedVerdict, FixOperation
        issues = [Issue(Severity.MINOR, Dimension.LOGICAL, "轻微问题"),
                  Issue(Severity.CRITICAL, Dimension.FACTUAL, "严重错误"),
                  Issue(Severity.MAJOR, Dimension.COVERAGE, "重要遗漏")]
        blue = BlueAgent(object(), max_issues_per_round=1)
        blue._fix_single_issue = AsyncMock(side_effect=lambda report, issue: FixOperation(issue=issue, action="fixed", success=True))
        blue._self_verify = AsyncMock(return_value=(True, []))
        _, operations = await blue.defend(ResearchReport(query="q", content="正文"), RedVerdict({}, 5, issues))
        self.assertEqual(blue._fix_single_issue.await_count, 1)
        self.assertEqual(blue._fix_single_issue.call_args.args[1].severity, Severity.CRITICAL)
        self.assertEqual(sum(op.action == "budget_deferred" and not op.success for op in operations), 2)
        self.assertIsNone(BlueAgent(object()).max_issues_per_round)

    async def test_complete_body_evidence_contract_and_review_queue_survive_pipeline(self):
        seen = []
        class Judge:
            def score_report(self, body, query, facts, requirements, evidence):
                seen.append((body, query, facts, requirements, evidence))
                return {"average": 9, "evaluator_version": "report-rubric-v1", "report_pass": False,
                        "reference_fact_coverage": 1}
        def initialize(config, session_id):
            class Orchestrator:
                async def run(self, query, config):
                    return ResearchReport(query=query, content="正文中有 LRU，但这份报告超出字数要求。")
            return {"orchestrator": Orchestrator(), "tools": []}
        dataset = {"name": "test", "version": "1", "examples": [{"id": "q", "query": "问题不能计入正文",
                   "ground_truth": {"LRU": "容量"}, "requirements": {"max_chars": 2}}],
                   "documents": [{"url": "https://x.test", "text": "完整证据" * 1000}]}
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.experiments.initialize_modules", side_effect=initialize):
            output = Path(tmp) / "run"
            result = await run_experiment({"adversarial": {"enabled": False}}, dataset, output, {"full": {}}, judge=Judge())
            row = result["details"][0]
            directory = output / row["report_path"]
            self.assertEqual((directory.parent / "report_body.md").read_text(), seen[0][0])
            self.assertNotIn("问题不能计入正文", seen[0][0])
            self.assertEqual(len(seen[0][4][0]["text"]), 4000)
            self.assertFalse(row["constraint_pass"])
            self.assertEqual(row["judge"]["average"], 9)
            self.assertEqual(row["status"], "success")
            queued = [json.loads(s) for s in export_review_queue(output, threshold=0).read_text().splitlines()]
            self.assertEqual(len(queued), 1) # high quality still needs contract review
            self.assertEqual(queued[0]["question"]["requirements"], {"max_chars": 2})

    async def test_judge_error_is_not_a_low_quality_score(self):
        from scripts.evaluate_reports import check_expectations
        checks = check_expectations({"contract": {"constraint_pass": True},
                 "evaluation": {"error": "invalid quotes"}, "expected": {"min_average": 7}}, {})
        self.assertEqual(checks, [{"name": "judge_valid", "actual": False, "expected": True, "passed": False}])


if __name__ == "__main__":
    unittest.main()
