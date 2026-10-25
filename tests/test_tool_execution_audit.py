"""Regression checks for semantic errors and double counting in offline audits."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_tool_execution import analyze, classify


class ToolExecutionAuditTests(unittest.TestCase):
    def test_returned_errors_override_successful_span(self):
        for result in ({"error": "missing tool"}, "[Calculator Error] SyntaxError: invalid syntax",
                       "[Calculator Error] Invalid value: Unsupported AST node: ListComp",
                       "Error: URL is not in the fixed evidence snapshot."):
            with self.subTest(result=result):
                self.assertEqual(classify({"status": "success", "outputs": {"result": result}})[0], "failed")

    def test_legitimate_outputs_and_censored_outcomes(self):
        for result in ("0", {"results": [], "error": False}, "An article discusses Error: handling."):
            self.assertEqual(classify({"status": "success", "outputs": {"result": result}})[0], "success")
        for status in ("timeout", "cancelled", "failed"):
            self.assertEqual(classify({"status": status})[0], status)
        self.assertEqual(classify({"status": "success"})[0], "unknown")

    def test_copies_and_nested_events_are_not_extra_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "controlled"
            directory.mkdir()
            span = {"span_id": "tool1", "name": "researcher.tool", "run_type": "tool",
                    "status": "success", "inputs": {"tool_name": "web_search"},
                    "outputs": {"result": {"results": []}}}
            trace = {"research_run_id": "run1", "spans": [span, span,
                     {"span_id": "blue1", "name": "blue_agent.defend", "run_type": "chain"}],
                     "events": [{"name": "snapshot_search", "span_id": "tool1"},
                                {"name": "snapshot_search", "span_id": "blue1"}]}
            (directory / "trace.json").write_text(json.dumps(trace))
            row = {"sample_id": "sample1", "trace_path": "trace.json", "variant": "full", "status": "success"}
            (directory / "results.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            result = analyze(root)
            self.assertEqual(result["summary"]["total"], 1)
            self.assertEqual(result["duplicate_runs_excluded"], 1)
            self.assertEqual(result["duplicate_tool_spans_excluded"], 1)
            self.assertEqual(result["supplementary_events_excluded_from_rate"]["blue_snapshot_search_success_events"], 1)

    def test_missing_trace_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "controlled").mkdir()
            (root / "controlled/results.jsonl").write_text(json.dumps({"trace_path": "missing.json"}) + "\n")
            result = analyze(root)
            self.assertEqual(result["missing_traces"], ["controlled/missing.json"])
            self.assertIsNone(result["summary"]["success_rate"])


if __name__ == "__main__":
    unittest.main()
