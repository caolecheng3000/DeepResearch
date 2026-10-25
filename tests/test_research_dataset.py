"""Research dataset provenance, runtime isolation and no-call experiment planning."""
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from evaluation.experiments import load_dataset, EvaluationInfrastructureError
from scripts.build_research_dataset import (
    build_payloads, validate, EMAIL_CONTACT, redact_contact_emails, select_spans,
    contains_contact_email,
)
from scripts.run_research_suite import prepare, execute, VARIANTS
from src.core.report_evaluation import report_contract
from src.tools.snapshot import SnapshotBrowserTool, SnapshotSearchTool

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation/data/research_deepresearch_v5.json"
CONFIG = ROOT / "configs/evaluation_research.yaml"


class ResearchDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DATA)

    def test_distinct_research_briefs_and_explicit_review_state(self):
        rows = self.dataset["examples"]
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({q["family_id"] for q in rows}), 24)
        self.assertEqual(len({q["title"] for q in rows}), 24)
        self.assertEqual(set(Counter(q["category"] for q in rows).values()), {4})
        self.assertEqual(Counter(q["split"] for q in rows), {"development": 6, "evaluation_candidate": 18})
        for q in rows:
            self.assertEqual(q["annotation_status"], "draft_for_user_review")
            self.assertGreaterEqual(len(q["evidence"]), 3)
            self.assertEqual(len(q["ground_truth"]), 6)
            self.assertNotIn("allowed_urls", q["requirements"])
            self.assertEqual(q["memory_seed"], [])

    def test_rebuild_and_exact_pdf_text_provenance(self):
        payload, dossiers, documents = build_payloads()
        expected = copy.deepcopy(self.dataset)
        expected.pop("content_hash")
        self.assertEqual(payload, expected)
        audit = validate(payload, dossiers, documents)
        self.assertEqual(audit["primary_sources"], 31)
        self.assertEqual(audit["reference_points"], 144)
        self.assertEqual(audit["verified_source_spans"], 98)
        self.assertGreater(audit["email_redacted_source_spans"], 0)
        for key, doc in documents.items():
            record = json.loads((ROOT / doc["source_record"]).read_text())
            self.assertEqual(hashlib.sha256((ROOT / doc["pdf_path"]).read_bytes()).hexdigest(), record["pdf_sha256"])
            full_text = (ROOT / doc["full_text_path"]).read_text()
            self.assertGreaterEqual(len(doc["excerpt_spans"]), 3)
            for span in doc["excerpt_spans"]:
                self.assertEqual(redact_contact_emails(full_text)[span["start"]:span["end"]], span["quote"])
                self.assertEqual(full_text.count("\f", 0, span["start"]) + 1, span["pdf_page_start"])

    def test_email_redaction_handles_pdf_layout_without_changing_other_text(self):
        at = chr(64)
        contacts = ["alice+review" + at + "example.org",
                    "{alice, bob,\ncarol}" + at + "lab.example.org",
                    "alice.smith\n" + at + "lab.example.org"]
        for contact in contacts:
            with self.subTest(contact_format=contacts.index(contact)):
                text = "Evidence before\n" + contact + "\nEvidence after\fpage two"
                result = redact_contact_emails(text)
                self.assertEqual(len(result), len(text))
                self.assertNotIn(at, result)
                self.assertEqual(result, redact_contact_emails(result))
                for old, new in zip(text, result):
                    if new != old:
                        self.assertEqual(new, "█")
                    if old.isspace():
                        self.assertEqual(new, old)
                self.assertTrue(result.startswith("Evidence before\n"))
                self.assertTrue(result.endswith("\nEvidence after\fpage two"))
        ordinary = "References https://arxiv.org/abs/2005.11401v1; " + at + "research_handle; 5 + 3 = 8."
        self.assertEqual(redact_contact_emails(ordinary), ordinary)

    def test_email_redaction_precedes_excerpt_boundary_selection(self):
        text = "x" * 2494 + " alice" + chr(64) + "example.org " + "y" * 11000
        spans = select_spans(text)
        self.assertEqual(spans[0]["end"], 2500)
        self.assertTrue(spans[0]["quote"].endswith("█████"))
        for span in spans:
            self.assertEqual(span["quote"], redact_contact_emails(text)[span["start"]:span["end"]])

    def test_published_datasets_and_materials_contain_no_contact_emails(self):
        paths = [DATA, DATA.with_name("research_deepresearch_v5_ablation.json")]
        paths += sorted((DATA.parent / "research_v5/materials").glob("*.json"))
        for path in paths:
            with self.subTest(path=path.name):
                self.assertIsNone(EMAIL_CONTACT.search(path.read_text()))
                # JSON escaping must not conceal multiline contact addresses.
                self.assertFalse(contains_contact_email(json.loads(path.read_text())))

    def test_runtime_corpus_contains_sources_not_review_answers(self):
        public = self.dataset["documents"]
        self.assertEqual(len(public), 31)
        for doc in public:
            self.assertEqual(set(doc), {"id", "title", "url", "text", "provenance"})
            self.assertTrue(doc["url"].startswith("https://arxiv.org/abs/"))
            self.assertTrue(doc["url"].endswith("v1"))
            self.assertNotIn("draft_requires_span_level_human_review", doc["text"])
            self.assertNotIn("critical_errors", doc["text"])
            # Current browser's default read must contain all selected excerpts.
            self.assertLessEqual(len(doc["text"]), 12000)
        for q in self.dataset["examples"]:
            for e in q["evidence"]:
                self.assertIn(e, public)

    def test_report_length_contract_suits_research_not_short_fact_answer(self):
        requirements = self.dataset["examples"][0]["requirements"]
        short = "结论：需要更多证据。https://arxiv.org/abs/2005.11401v1"
        self.assertFalse(report_contract(short, requirements)["constraint_pass"])
        report = "## 研究范围\n" + "本研究区分原文结论、合理推断与拟议实验。" * 110
        report += "\n## 参考来源\n1. [RAG](https://arxiv.org/abs/2005.11401v1)"
        self.assertTrue(report_contract(report, requirements)["constraint_pass"])

    def test_offline_plan_and_pilot_without_model_construction(self):
        with patch("src.core.judge.LLMJudge", side_effect=AssertionError("No model in planning")):
            _, full, anchors, plan = prepare(CONFIG)
            _, pilot, _, pilot_plan = prepare(CONFIG, pilot=True)
        self.assertEqual(plan["planned_research_attempts"], 54)
        self.assertEqual(len(anchors["examples"]), 6)
        self.assertEqual(pilot_plan["planned_research_attempts"], 6)
        self.assertFalse(plan["model_calls_started"])
        self.assertNotEqual(full["content_hash"], pilot["content_hash"])
        self.assertEqual(plan["max_adversarial_rounds"], 3)
        self.assertEqual(plan["max_total_blue_issues_per_round"], 5)

    def test_rejects_mixed_old_corpus_before_running(self):
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            prepare(ROOT / "configs/evaluation_capability.yaml")


class ResearchRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_snapshot_search_and_material_read_no_annotation(self):
        search = SnapshotSearchTool(str(DATA))
        browser = SnapshotBrowserTool(str(DATA))
        result = await search.execute("FlashAttention IO HBM SRAM", top_n=5)
        self.assertTrue(any("2205.14135" in r["url"] for r in result["results"]))
        url = next(r["url"] for r in result["results"] if "2205.14135" in r["url"])
        text = await browser.execute(url)
        self.assertIn("FlashAttention", text)
        self.assertIn("excerpt_", text)
        self.assertNotIn("reference_points", text)
        self.assertIn("not in the fixed", await browser.execute("https://example.com/not-archived"))

    async def test_experiment_reuses_full_anchors_for_paired_controls(self):
        config, dataset, anchors, plan = prepare(CONFIG)
        def row(q, variant):
            return {"question_id": q["id"], "variant": variant, "status": "success",
                    "composite_score": .8, "elapsed_seconds": 1, "usage": {"total_tokens": 10}}
        full_rows = [row(q, "full") for q in dataset["examples"]]
        controls = [row(q, variant) for variant in VARIANTS for q in anchors["examples"]]
        mock_run = AsyncMock(side_effect=[{"details": full_rows}, {"details": controls}])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            args = SimpleNamespace(output_dir=out, judge=False, pilot=False)
            with patch("evaluation.experiments.run_experiment", mock_run), patch("evaluation.experiments.export_review_queue"), patch("src.core.runner.setup_logging"):
                await execute(args, config, dataset, anchors, plan)
            saved = json.loads((out / "paired_ablation_summary.json").read_text())
            self.assertEqual(mock_run.await_count, 2)
            self.assertEqual(saved["variants"]["full"]["num_attempted"], 6)
            self.assertEqual(set(saved["comparisons"]), {f"{name}_minus_full" for name in VARIANTS})
            self.assertEqual(json.loads((out / "suite_manifest.json").read_text())["status"], "completed")

    async def test_resource_failure_does_not_start_controls(self):
        config, dataset, anchors, plan = prepare(CONFIG)
        mocked = AsyncMock(side_effect=EvaluationInfrastructureError("HTTP 402"))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            args = SimpleNamespace(output_dir=out, judge=False, pilot=False)
            with patch("evaluation.experiments.run_experiment", mocked), patch("src.core.runner.setup_logging"):
                with self.assertRaises(EvaluationInfrastructureError):
                    await execute(args, config, dataset, anchors, plan)
            self.assertEqual(mocked.await_count, 1)
            self.assertEqual(json.loads((out / "suite_manifest.json").read_text())["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
