"""Audit saved tool spans without importing agents, calling models, or running tools."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


STAGES = ("controlled", "ablations", "hotpot")


def classify(span):
    """A normally returned error is still a tool failure, not a successful call."""
    status = span.get("status")
    if status in ("timeout", "cancelled", "failed"):
        return status, span.get("error_type") or status
    if status != "success":
        return "unknown", "unrecognized_span_status"
    outputs = span.get("outputs", {})
    if "result" not in outputs:
        return "unknown", "missing_result"
    result = outputs["result"]
    if isinstance(result, dict) and result.get("error"):
        return "failed", "returned_error"
    if isinstance(result, str):
        text = result.lstrip()
        if text.startswith("[Calculator Error]"):
            if "Unsupported AST node: ListComp" in text:
                return "failed", "calculator_unsupported_list_comprehension"
            if "SyntaxError:" in text:
                return "failed", "calculator_syntax_error"
            return "failed", "calculator_returned_error"
        if text.startswith("Error: URL is not in the fixed evidence snapshot."):
            return "failed", "browser_url_not_in_snapshot"
        if text.startswith("Error:"):
            return "failed", "returned_error_string"
    if result is None:
        return "unknown", "null_result"
    return "success", None


def summarize(calls):
    counts = Counter(c["outcome"] for c in calls)
    return {
        "total": len(calls),
        **{k: counts[k] for k in ("success", "failed", "timeout", "cancelled", "unknown")},
        "success_rate": counts["success"] / len(calls) if calls else None,
    }


def analyze(root):
    root = root.resolve()
    calls, traces, missing = [], [], []
    seen_runs, seen_spans = {}, {}
    duplicate_runs = duplicate_spans = rows = 0
    supplementary = Counter()
    other_events = Counter()
    for stage in STAGES:
        results = root / stage / "results.jsonl"
        if not results.exists():
            continue
        for line in results.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            path = (root / stage / row["trace_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Trace path must stay within the experiment directory")
            relative = path.relative_to(root).as_posix()
            if not path.is_file():
                missing.append(relative)
                continue
            raw = path.read_bytes()
            trace = json.loads(raw)
            digest = hashlib.sha256(raw).hexdigest()
            run_id = trace.get("research_run_id")
            if not run_id:
                raise ValueError(f"Missing research_run_id: {relative}")
            if run_id in seen_runs:
                if seen_runs[run_id] != digest:
                    raise ValueError(f"Conflicting copies of run {run_id}")
                duplicate_runs += 1
                continue
            seen_runs[run_id] = digest
            spans = {}
            for span in trace.get("spans", []):
                span_id = span["span_id"]
                if span_id in spans and spans[span_id] != span:
                    raise ValueError(f"Conflicting span {span_id}")
                spans[span_id] = span
            traces.append({
                "trace_path": relative, "sha256": digest, "research_run_id": run_id,
                "sample_id": row["sample_id"], "stage": stage,
                "variant": row["variant"], "research_status": row["status"],
                "tool_mode": trace.get("metadata", {}).get("tool_mode", "unknown"),
                "dropped_records": trace.get("dropped_records", 0),
            })
            for span in trace.get("spans", []):
                if span.get("run_type") != "tool":
                    continue
                identity = (run_id, span["span_id"])
                if identity in seen_spans:
                    if seen_spans[identity] != span:
                        raise ValueError(f"Conflicting tool span {identity}")
                    duplicate_spans += 1
                    continue
                seen_spans[identity] = span
                inputs = span.get("inputs", {})
                tool = inputs.get("tool_name") or inputs.get("tool_call", {}).get("function", {}).get("name") or "unknown"
                outcome, reason = classify(span)
                calls.append({
                    "research_run_id": run_id, "sample_id": row["sample_id"],
                    "stage": stage, "variant": row["variant"], "tool": tool,
                    "span_id": span["span_id"], "span_name": span["name"],
                    "span_status": span.get("status"), "outcome": outcome,
                    "failure_reason": reason, "trace_path": relative,
                })
            # Snapshot events already inside a tool span must not become a second call.
            # Blue has only successful snapshot events, so it cannot supply a full denominator.
            for event in trace.get("events", []):
                if event.get("name") not in ("snapshot_search", "snapshot_read"):
                    continue
                parent = spans.get(event.get("span_id"), {})
                if parent.get("run_type") == "tool":
                    continue
                if event["name"] == "snapshot_search" and parent.get("name") == "blue_agent.defend":
                    supplementary["blue_snapshot_search_success_events"] += 1
                else:
                    other_events[event["name"] + "/" + parent.get("name", "missing_parent")] += 1
    def grouped(key):
        groups = defaultdict(list)
        for call in calls:
            groups[call[key]].append(call)
        return {k: summarize(v) for k, v in sorted(groups.items())}

    return {
        "schema_version": "tool-execution-audit-v1", "root": str(root),
        "scope": "Selected primary results in controlled/ablations/hotpot, including unsuccessful research attempts; no Judge or ancestor directories.",
        "definition": "success / all unique recorded tool spans; explicit returned errors override span success; timeout, cancellation and unknown remain in denominator",
        "limitations": [
            "Tool execution success does not establish correct selection, arguments, retrieval relevance or numerical correctness.",
            "Only explicit known error formats are recognized; an ordinary-looking wrong result can still be counted as execution success.",
            "Blue supplementary searches lack attempt-level tool spans; successful snapshot events are listed separately, without a success rate.",
            "A missing trace, dropped records or unrecorded in-flight span can make coverage incomplete.",
        ],
        "result_rows": rows, "unique_traces": len(traces),
        "research_statuses": dict(Counter(t["research_status"] for t in traces)),
        "tool_modes": dict(Counter(t["tool_mode"] for t in traces)),
        "missing_traces": missing, "dropped_records": sum(t["dropped_records"] for t in traces),
        "duplicate_runs_excluded": duplicate_runs, "duplicate_tool_spans_excluded": duplicate_spans,
        "summary": summarize(calls), "by_tool": grouped("tool"),
        "by_role": grouped("span_name"), "by_variant": grouped("variant"),
        "failure_reasons": dict(Counter(c["failure_reason"] for c in calls if c["outcome"] != "success")),
        "supplementary_events_excluded_from_rate": dict(supplementary),
        "other_unwrapped_snapshot_events": dict(other_events),
        "sources": traces, "calls": calls,
    }


def markdown(audit):
    s = audit["summary"]
    lines = ["# 工具执行成功率离线统计", "",
        f"范围：当前扩展测评的 {audit['result_rows']} 条主实验记录、{audit['unique_traces']} 份独立研究 Trace。包含研究未成功的尝试，不混入 Judge、祖先实验副本或新模型调用。",
        "", "统计单位是实际记录的工具执行跨度；同一研究/跨度去重，搜索事件不重复计数。正常返回的错误字符串或 error 字段也计为失败。",
        "", f"**执行成功率：{s['success']}/{s['total']} = {s['success_rate']:.2%}。**", "",
        "|工具|调用数|成功|失败|超时|取消|未知|成功率|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in audit["by_tool"].items():
        lines.append(f"|{name}|{values['total']}|{values['success']}|{values['failed']}|{values['timeout']}|{values['cancelled']}|{values['unknown']}|{values['success_rate']:.2%}|")
    lines += ["", "按执行入口：", "", "|入口|调用数|成功|成功率|", "|---|---:|---:|---:|"]
    for name, values in audit["by_role"].items():
        lines.append(f"|{name}|{values['total']}|{values['success']}|{values['success_rate']:.2%}|")
    lines += ["", "按实验配置：", "", "|配置|调用数|成功|失败|成功率|", "|---|---:|---:|---:|---:|"]
    for name, values in audit["by_variant"].items():
        lines.append(f"|{name}|{values['total']}|{values['success']}|{values['failed']}|{values['success_rate']:.2%}|")
    lines += ["", "错误分类：", ""]
    lines.extend(f"- `{reason}`：{count} 次。" for reason, count in audit["failure_reasons"].items())
    blue = audit["supplementary_events_excluded_from_rate"].get("blue_snapshot_search_success_events", 0)
    lines += ["", f"另有 {blue} 条 Blue 补搜成功事件，未混入上述分母。该路径没有独立的工具尝试跨度，失败/取消尝试可能没有对应事件，因此不能给出 Blue 补搜成功率或全系统完整成功率。",
        "", f"数据检查：缺失 Trace {len(audit['missing_traces'])}，丢弃记录 {audit['dropped_records']}，排除重复研究 {audit['duplicate_runs_excluded']}，排除重复工具跨度 {audit['duplicate_tool_spans_excluded']}。",
        "", "这批实验的工具模式是固定材料快照；本结果不代表真实联网搜索服务的可靠性，也不等同于工具选择准确率、材料相关率或计算结果正确率。Trace 的 success 只表示函数正常返回，不能单独当执行成功依据。",
        "", "原始记录保持不变；JSON 中保存每次调用的 Trace/跨度 ID、判定原因和来源文件 SHA256，便于复核。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(args.root)
    if not audit["summary"]["total"]:
        raise SystemExit("No tool spans found; no success rate can be reported")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tool_execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "tool_execution_report.md").write_text(markdown(audit), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("unique_traces", "summary", "by_tool", "failure_reasons", "supplementary_events_excluded_from_rate")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
