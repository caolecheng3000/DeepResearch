"""New suite boundaries: no network/model calls or production memory."""
import asyncio
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch

from src.core.report_evaluation import report_contract
from src.models.vllm_policy import VLLMPolicy
from src.tools.snapshot import SnapshotSearchTool
from src.orchestrator.schemas import ResearchReport
from evaluation.experiments import run_experiment,load_dataset
from evaluation.experiments import EvaluationInfrastructureError
from evaluation.qa_metrics import extract_answer,answer_metrics

ROOT=Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_clarified_scope_keeps_oracles_separate_without_hidden_gold_url_ban(self):
        d=load_dataset(ROOT/"evaluation/data/research_capability_v4.json")
        public={doc["url"] for doc in d["documents"]}
        for q in d["examples"]:
            self.assertEqual(len(q["evidence"]),3)
            self.assertEqual(len(q["reference_doc_urls"]),2)
            self.assertNotIn("allowed_urls",q["requirements"])
            self.assertFalse({s["url"] for s in q["memory_seed"]}&public)

    def test_coverage_gold_contracts_and_private_memory_not_searchable(self):
        d=load_dataset(ROOT/"evaluation/data/research_capability_v3.json")
        self.assertEqual(len(d["examples"]),48)
        self.assertEqual(set(Counter(q["category"] for q in d["examples"]).values()),{4})
        public={doc["url"] for doc in d["documents"]}
        for q in d["examples"]:
            with self.subTest(q=q["id"]):
                self.assertTrue(report_contract(q["reference_report"],q["requirements"])["constraint_pass"])
                self.assertEqual(len(q["ground_truth"]),4)
                self.assertFalse({s["url"] for s in q["memory_seed"]}&public)
        anchors=load_dataset(ROOT/"evaluation/data/research_capability_v3_ablation.json")
        self.assertEqual(len(anchors["examples"]),12)
        self.assertEqual(len({q["category"] for q in anchors["examples"]}),12)

    def test_real_hotpot_has_provenance_and_no_synthetic_ids(self):
        d=load_dataset(ROOT/"evaluation/data/research_hotpotqa_v1.json")
        self.assertEqual(len(d["examples"]),24)
        self.assertEqual({q["category"] for q in d["examples"]},{"hotpot_bridge","hotpot_comparison"})
        self.assertEqual(len(d["provenance"]["raw_sha256"]),64)
        self.assertTrue(all(q.get("original_id") and q.get("reference_doc_urls") for q in d["examples"]))

    def test_case_local_judge_evidence_and_negative_is_actually_changed(self):
        d=json.loads((ROOT/"evaluation/data/report_judge_regression_v2.json").read_text())
        self.assertEqual(len(d["cases"]),48)
        cases={c["id"]:c for c in d["cases"]}
        for c in cases.values():
            self.assertEqual(len(c["evidence"]),2)
            if c["id"].endswith("_wrong"):
                correct=cases[c["id"].removesuffix("_wrong")+"_correct"]
                self.assertNotEqual(c["report"],correct["report"])
                self.assertEqual(c["evidence"],correct["evidence"])

    def test_qa_extraction_does_not_guess_or_use_reference(self):
        self.assertIsNone(extract_answer("报告提到了 Alice，但没有答案行"))
        self.assertEqual(extract_answer("答案：Alice\n证据"),"Alice")
        self.assertEqual(answer_metrics("The New-York","newyork"),{"exact_match":1.0,"f1":1.0})
        self.assertEqual(answer_metrics("not yes","yes")["f1"],0)
        self.assertEqual(answer_metrics(None,"Alice")["exact_match"],0)

    def test_thinking_switch_is_optional_and_cloned(self):
        p=VLLMPolicy(base_url="http://unit.test/v1",api_key="unit",thinking_mode="disabled")
        self.assertEqual(p._prepare_request([{"role":"user","content":"q"}])["extra_body"],{"thinking":{"type":"disabled"}})
        self.assertEqual(p.clone().thinking_mode,"disabled")
        p.client.close()
        with self.assertRaises(ValueError):
            VLLMPolicy(base_url="http://unit.test/v1",api_key="unit",thinking_mode="invalid")

    def test_standard_numbered_bibliography_resolves_but_unrelated_lists_do_not(self):
        for entry in ("1. [材料](https://x.test)", "1) https://x.test", "1、材料 https://x.test"):
            result=report_contract("结论[1]。\n## 参考来源\n"+entry)
            self.assertTrue(result["constraint_pass"])
            self.assertEqual(result["contract_version"],"report-contract-v2")
        for body in ("结论[1]。\n## 实施步骤\n1. 请访问 https://x.test",
                     "结论[2]。\n## 参考来源\n1. https://x.test",
                     "结论[2]。\n## 参考来源\n1. https://x.test\n## 附录\n2. https://y.test"):
            self.assertFalse(report_contract(body)["constraint_pass"])

    def test_provider_balance_failure_emits_structured_stop_signal(self):
        from src.utils.tracing import research_trace
        class BalanceError(Exception):
            status_code=402
        p=VLLMPolicy(base_url="http://unit.test/v1",api_key="unit")
        try:
            with research_trace() as trace, self.assertRaisesRegex(RuntimeError,"AUTH_OR_QUOTA"):
                p._handle_error(BalanceError("Insufficient Balance"),[])
            event=next(e for e in trace.snapshot()["events"] if e["name"]=="llm_error")
            self.assertTrue(event["data"]["resource_failure"])
            self.assertEqual(event["data"]["status_code"],402)
        finally:p.client.close()


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_uniform_review_reuses_valid_checkpoint_without_new_model_calls(self):
        from scripts import review_uniform_reports as review
        case={'id':'sample/final','stage':'controlled','sample_id':'sample','question_id':'q',
            'variant':'full','research_status':'success','category':'unit','version':'final',
            'report_path':'body.md','report_sha256':'unit','report':'saved report',
            'question':{'query':'question','requirements':{},'evidence':[]}}
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'suite_manifest.json').write_text('{"status":"completed"}')
            judge=Mock();judge.score_report.return_value={'average':8}
            with patch.object(review,'inputs',return_value=[case]),patch.object(review,'grading_identity',return_value={
                'requested_model':'unit','provider_host':'unit','grader_code_hash':'unit'}),patch.object(review,'LLMJudge',return_value=judge):
                await review.run(SimpleNamespace(base=base))
                first=json.loads((base/'uniform_report_v4/manifest.json').read_text())
                await review.run(SimpleNamespace(base=base))
            self.assertEqual(judge.score_report.call_count,1)
            self.assertEqual(first['created_at'],json.loads((base/'uniform_report_v4/manifest.json').read_text())['created_at'])

    async def test_uniform_review_retries_only_invalid_output_and_retains_first_failure(self):
        from scripts import review_uniform_reports as review
        case={'id':'sample/final','stage':'controlled','sample_id':'sample','question_id':'q',
            'variant':'full','research_status':'success','category':'unit','version':'final',
            'report_path':'body.md','report_sha256':'unit','report':'saved report',
            'question':{'query':'question','requirements':{},'evidence':[]}}
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'suite_manifest.json').write_text('{"status":"completed"}')
            judge=Mock();judge.score_report.side_effect=[{'error':'invalid JSON'},{'average':8}]
            with patch.object(review,'inputs',return_value=[case]),patch.object(review,'grading_identity',return_value={}),patch.object(review,'LLMJudge',return_value=judge):
                await review.run(SimpleNamespace(base=base))
            row=json.loads((base/'uniform_report_v4/results.json').read_text())[0]
            self.assertEqual(judge.score_report.call_count,2)
            self.assertFalse(row['initial_valid']);self.assertTrue(row['valid'])
            self.assertEqual(row['attempts'][0]['evaluation']['error'],'invalid JSON')

    async def test_uniform_review_does_not_retry_provider_quota_errors(self):
        from scripts import review_uniform_reports as review
        case={'id':'sample/final','stage':'controlled','sample_id':'sample','question_id':'q',
            'variant':'full','research_status':'success','category':'unit','version':'final',
            'report_path':'body.md','report_sha256':'unit','report':'saved report',
            'question':{'query':'question','requirements':{},'evidence':[]}}
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'suite_manifest.json').write_text('{"status":"completed"}')
            judge=Mock();judge.score_report.return_value={'error':'[LLM_AUTH_OR_QUOTA_ERROR] HTTP 402'}
            with patch.object(review,'inputs',return_value=[case]),patch.object(review,'grading_identity',return_value={}),patch.object(review,'LLMJudge',return_value=judge),self.assertRaises(EvaluationInfrastructureError):
                await review.run(SimpleNamespace(base=base))
            self.assertEqual(judge.score_report.call_count,1)
            self.assertEqual(json.loads((base/'uniform_report_v4/manifest.json').read_text())['status'],'interrupted')

    async def test_report_regression_stops_on_resource_error_after_saving_first_case(self):
        from scripts.evaluate_reports import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=root/'fixture.json'
            fixture.write_text(json.dumps({'questions':{'q':{'query':'question'}},'documents':[],
                'cases':[{'id':str(i),'question_id':'q','report':'saved report body'} for i in range(2)]}))
            args=SimpleNamespace(regression=fixture,experiment_dir=None,output_dir=root/'review',
                judge=True,backend='unit',max_tokens=100,thinking_mode=None)
            judge=Mock();judge.score_report.return_value={'error':'[LLM_AUTH_OR_QUOTA_ERROR] HTTP 402'}
            with patch('scripts.evaluate_reports.LLMJudge',return_value=judge),self.assertRaises(EvaluationInfrastructureError):
                await run(args)
            self.assertEqual(judge.score_report.call_count,1)
            self.assertEqual(len(json.loads((root/'review/results.json').read_text())),1)
            self.assertEqual(json.loads((root/'review/manifest.json').read_text())['status'],'interrupted')

    async def test_pre_adversarial_review_stops_on_resource_error_and_preserves_evaluation(self):
        from scripts.analyze_capability_suite import review_before
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'report_before_adversarial.md').write_text('saved report body')
            row={'_stage':'controlled','ablation':True,'sample_id':'first','question_id':'q',
                '_directory':str(root),'_question':{'query':'question'}}
            judge=Mock();judge.score_report.return_value={'error':'[LLM_AUTH_OR_QUOTA_ERROR] HTTP 402'}
            with self.assertRaises(EvaluationInfrastructureError):
                review_before(root,[row,{**row,'sample_id':'second'}],judge)
            self.assertEqual(judge.score_report.call_count,1)
            self.assertEqual(len(json.loads((root/'pre_adversarial_reviews.json').read_text())),1)

    async def test_schema_feedback_does_not_retry_account_errors(self):
        from src.utils.schema_validator import retry_with_schema_feedback
        call=AsyncMock(side_effect=RuntimeError("[LLM_AUTH_OR_QUOTA_ERROR] HTTP 402"))
        with self.assertRaisesRegex(RuntimeError,"AUTH_OR_QUOTA"):
            await retry_with_schema_feedback(call,[],Mock(),max_retries=2)
        self.assertEqual(call.await_count,1)

    async def test_resource_failure_stops_pending_samples_and_preserves_first_attempt(self):
        data={"examples":[{"id":str(i),"query":"q"} for i in range(5)]}
        async def fail(question,config,directory,metadata,judge):
            return {"sample_id":"failed-attempt","question_id":question["id"],"variant":"full",
                "status":"failed","judge_status":"not_requested_infrastructure_error",
                "composite_score":0,"elapsed_seconds":0,"usage":{},"infrastructure_error":"provider_http_402"}
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.experiments.run_sample",side_effect=fail) as mocked:
            output=Path(tmp)/"run"
            with self.assertRaises(EvaluationInfrastructureError):
                await run_experiment({},data,output,{"full":{}})
            self.assertEqual(mocked.await_count,1)
            self.assertEqual(len((output/"results.jsonl").read_text().splitlines()),1)
            self.assertEqual(json.loads((output/"manifest.json").read_text())["status"],"failed")

    async def test_resume_reuses_success_without_model_calls_and_rejects_changed_inputs(self):
        count=0
        async def research(query,cfg,modules,metadata):
            nonlocal count
            count+=1
            modules["last_report"]=ResearchReport(query=query,content="已经完成的正文")
            return "已经完成的正文"
        data={"examples":[{"id":str(i),"query":"q"} for i in range(2)]}
        cfg={"adversarial":{"enabled":False}}
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.experiments.initialize_modules",return_value={}),patch("evaluation.experiments.run_research",side_effect=research):
            original=Path(tmp)/"original"
            first=await run_experiment(cfg,data,original,{"full":{}})
            second=await run_experiment(cfg,data,Path(tmp)/"resumed",{"full":{}},resume_from=original)
            self.assertEqual(count,2)
            self.assertTrue(all(r.get("reused_from") for r in second["details"]))
            self.assertEqual({r["sample_id"] for r in first["details"]},{r["sample_id"] for r in second["details"]})
            changed={"examples":[{"id":"0","query":"different"}]}
            with self.assertRaisesRegex(ValueError,"identical dataset"):
                await run_experiment(cfg,changed,Path(tmp)/"bad",{"full":{}},resume_from=original)

    async def test_snapshot_math_tool_is_real_and_optional(self):
        from src.core.runner import _create_tools_factory
        config={"tools":{"snapshot_path":str(ROOT/"evaluation/data/research_capability_v3.json")}}
        self.assertEqual(len(_create_tools_factory(config)),2)
        config["tools"]["snapshot_calculator"]=True
        tools=_create_tools_factory(config)
        calculator=next(t for t in tools if t.name=="calculator")
        result=await calculator.execute(expression="(130-3-45)/(130-3)*100")
        self.assertIn("64.566",str(result))

    async def test_ranked_snapshot_uses_query_and_requires_read_for_long_tail(self):
        d={"search_mode":"bm25","snippet_chars":20,"documents":[
            {"title":"wrong","url":"https://x/no","text":"不相关"},
            {"title":"ORIONCASE","url":"https://x/yes","text":"ORIONCASE "+"背景"*100+"末尾修订"}]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data.json";p.write_text(json.dumps(d))
            t=SnapshotSearchTool(str(p))
            rows=(await t.execute("ORIONCASE"))["results"]
            self.assertEqual(rows[0]["url"],"https://x/yes")
            self.assertNotIn("末尾修订",rows[0]["snippet"])
            with self.assertRaises(ValueError):await t.execute("q",top_n=-1)

    async def test_blue_five_is_global_across_all_fix_types(self):
        from src.adversarial.blue_agent import BlueAgent
        from src.adversarial.verdict import Issue,Severity,Dimension,FixType,RedVerdict,FixOperation
        types=[FixType.IN_PLACE,FixType.SUPPLEMENTARY,FixType.REMOVAL]
        issues=[Issue(Severity.MAJOR,Dimension.LOGICAL,f"problem {i}",fix_type=types[i%3]) for i in range(12)]
        blue=BlueAgent(object(),max_issues_per_round=5)
        blue._fix_single_issue=AsyncMock(side_effect=lambda report,issue:FixOperation(issue=issue,action="fixed",success=True))
        blue._self_verify=AsyncMock(return_value=(True,[]))
        _,operations=await blue.defend(ResearchReport(query="q",content="正文"),RedVerdict({},5,issues))
        self.assertEqual(blue._fix_single_issue.await_count,5)
        self.assertEqual(sum(op.action=="budget_deferred" for op in operations),7)

    async def test_concurrent_samples_are_bounded_and_seed_same_supplied_context(self):
        active=peak=0;stores=[]
        def initialize(cfg,session_id):
            store=Mock();stores.append(store)
            return {"memory_store":store,"orchestrator":SimpleNamespace(pre_adversarial_report=None)}
        async def research(query,cfg,modules,metadata):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            try:
                await asyncio.sleep(.02)
                modules["last_report"]=ResearchReport(query=query,content="正文")
                return "正文"
            finally:active-=1
        data={"examples":[{"id":str(i),"query":"q","memory_seed":[{"url":"https://x/prior","text":"先前提供的信息"}]} for i in range(4)]}
        with tempfile.TemporaryDirectory() as tmp,patch("evaluation.experiments.initialize_modules",side_effect=initialize),patch("evaluation.experiments.run_research",side_effect=research):
            result=await run_experiment({"evaluation":{"max_sample_concurrency":2},"adversarial":{"enabled":False}},data,Path(tmp)/"run",{"full":{}})
        self.assertEqual(peak,2)
        self.assertEqual(len(result["details"]),4)
        self.assertTrue(all(store.put.call_count==1 for store in stores))
        self.assertEqual(len({r["sample_id"] for r in result["details"]}),4)


if __name__=="__main__":unittest.main()
