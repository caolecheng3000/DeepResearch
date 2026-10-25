#!/usr/bin/env python3
"""Render saved local traces without network requests or model calls (stdlib only)."""

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path


MODEL_SPANS = {"policy.call", "policy.acall"}


def pretty(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def block(value):
    return "<pre>" + escape(pretty(value)) + "</pre>"


def details(title, value):
    return "<details><summary>" + escape(title) + "</summary>" + block(value) + "</details>"


def load_optional(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_view(trace):
    """Keep causal ancestry; sort by start time, since spans are saved on completion."""
    spans = sorted(trace.get("spans", []), key=lambda s: s.get("offset_seconds", 0))
    by_id = {s["span_id"]: s for s in spans}
    tasks = {
        e["span_id"]: e.get("data", {})
        for e in trace.get("events", []) if e["name"] == "task_started"
    }
    usages = {
        e["span_id"]: e.get("data", {})
        for e in trace.get("events", []) if e["name"] == "llm_usage"
    }
    rows, rounds = [], Counter()
    for span in spans:
        name = span["name"]
        if name not in MODEL_SPANS and name != "researcher.tool":
            continue
        ancestors, seen, ancestor = [], set(), span
        while ancestor and ancestor["span_id"] not in seen:
            seen.add(ancestor["span_id"])
            ancestors.append(ancestor)
            ancestor = by_id.get(ancestor.get("parent_span_id"))
        worker = next((a for a in ancestors if a["name"] == "worker.task"), None)
        task = dict(tasks.get(worker["span_id"], {})) if worker else {}
        researcher = next((a for a in ancestors if a["name"] == "researcher.run"), {})
        task_input = researcher.get("inputs", {}).get("task", {})
        if isinstance(task_input, dict):
            task = {**task_input, **task}
        if worker:
            task_id = task.get("task_id", worker.get("inputs", {}).get("task_id", "worker"))
            role = f"{task_id} · {task.get('task_type', 'Worker')}"
            # Separate retries/replans with the same task ID by worker span identity.
            lane = worker["span_id"]
        else:
            role = next((label for prefix, label in (
                ("planner.", "Planner"), ("summarizer.", "Summarizer"), ("judge.", "Judge")
            ) if any(a["name"].startswith(prefix) for a in ancestors)), "模型")
            lane = role
        result = span.get("outputs", {}).get("result", {})
        is_model = name in MODEL_SPANS
        if is_model:
            rounds[lane] += 1
        rows.append({
            "span": span, "kind": "model" if is_model else "tool", "role": role,
            "round": rounds[lane], "task": task, "result": result,
            "usage": usages.get(span["span_id"], result.get("usage", {}) if isinstance(result, dict) else {}),
        })
    question = next((s.get("inputs", {}).get("context", {}).get("query")
                     for s in spans if s["name"] == "researcher.run"), None)
    if not question:
        question = next((s.get("inputs", {}).get("query") for s in spans
                         if s["name"] == "planner.generate_plan"), "原问题未保存在预览中")
    return {"trace": trace, "rows": rows, "question": question}


def render_rows(view):
    rendered, model_number = [], 0
    elapsed = max(float(view["trace"].get("elapsed_seconds", 0)), 0.001)
    for row in view["rows"]:
        s, result = row["span"], row["result"]
        start, duration = s.get("offset_seconds", 0), s.get("duration_seconds", 0)
        is_model = row["kind"] == "model"
        content = result.get("content", "") if isinstance(result, dict) else ""
        if is_model:
            model_number += 1
            title = f"模型 {model_number} · {row['role']} · 第 {row['round']} 轮"
            calls = result.get("tool_calls", []) if isinstance(result, dict) else []
            call_names = [str(c.get("function", {}).get("name", "未知工具")) for c in calls if isinstance(c, dict)]
            summary = content[:220] if content else "本轮未输出文字，直接请求工具。" if calls else "本地未记录文字输出。"
            body = "<p class='preview'>" + escape(summary) + ("…" if len(content) > 220 else "") + "</p>"
            if calls:
                body += "<p class='request'>请求工具：" + escape(" + ".join(call_names)) + "</p>"
                body += details("工具请求与参数", calls)
            body += details("模型文字输出（已保存的预览）", content or "（无文字输出）")
            body += details("模型输入消息（已保存的预览）", s.get("inputs", {}))
            tokens = row["usage"].get("total_tokens")
            body += "<p class='meta'>Token：" + (f"{tokens:,}" if isinstance(tokens, int) else "Usage 未记录") + "</p>"
        else:
            tool_input = s.get("inputs", {})
            title = f"工具 · {tool_input.get('tool_name', '未知')} · {row['role']} 第 {row['round']} 轮"
            if isinstance(result, dict) and isinstance(result.get("results"), list):
                overview = f"返回 {len(result['results'])} 条材料 · 来源：{result.get('source', '未记录')}"
            else:
                overview = pretty(result)[:180]
            body = "<p class='preview'>" + escape(overview) + "</p>"
            body += details("调用参数", tool_input.get("args", {}))
            body += details("工具返回（已保存的预览）", result)
        body += details("跨度位置与状态", {
            "span_id": s["span_id"], "parent_span_id": s.get("parent_span_id"),
            "span_status": s.get("status"), "error_type": s.get("error_type"),
            "task_id": row["task"].get("task_id"),
            "dependencies": row["task"].get("dependencies", []),
        })
        rendered.append(
            f"<article class='card {row['kind']}' data-kind='{row['kind']}'>"
            f"<div class='meta'>+{start:.3f}s → +{start + duration:.3f}s · 耗时 {duration:.3f}s"
            f" · {escape(str(s.get('status', 'unknown')))}</div><h3>{escape(title)}</h3>"
            f"<div class='bar'><i style='margin-left:{min(start / elapsed * 100, 100):.3f}%;"
            f"width:{max(min(duration / elapsed * 100, 100), .2):.3f}%'></i></div>{body}</article>"
        )
    return "".join(rendered)


CSS = """
:root{color-scheme:light;--ink:#1e293b;--muted:#64748b;--line:#dbe4ef}
*{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:var(--ink);font:15px/1.65 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:36px 22px}h1{font-size:28px;margin:0}h2{font-size:21px;margin-top:32px}h3{font-size:17px;margin:6px 0 10px}
.meta,.note{color:var(--muted);font-size:13px}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}.stat{background:white;border:1px solid var(--line);border-radius:12px;padding:12px 20px}.stat b{font-size:24px;display:block}
.question,.card,.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:14px 0}.question{border-left:4px solid #334155}
.timeline{border-left:2px solid var(--line);padding-left:18px}.model{border-left:4px solid #6366f1}.tool{border-left:4px solid #0d9488;background:#f5fffc}
.preview{white-space:pre-wrap}.request{color:#4338ca}.bar{height:5px;background:#edf1f7;margin:12px 0;overflow:hidden;border-radius:5px}.bar i{display:block;height:5px;background:#818cf8}.tool .bar i{background:#14b8a6}
summary{cursor:pointer;font-weight:600;padding:7px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;border-radius:8px;padding:14px;font:13px/1.7 ui-monospace,monospace;max-height:540px;overflow:auto}
button{border:1px solid var(--line);background:white;padding:8px 14px;border-radius:8px;cursor:pointer;margin:4px}button:hover,button[aria-pressed=true]{background:#e0e7ff}
.flow{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.chip{background:#e8ecff;padding:7px 12px;border-radius:8px}.arrow{color:#94a3b8}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid var(--line)}
@media(max-width:600px){main{padding:22px 12px}.timeline{padding-left:8px}.card{padding:14px}.stat{padding:10px 14px}}
"""


def render_html(view, judge_view, result, report):
    trace, rows = view["trace"], view["rows"]
    models = [r for r in rows if r["kind"] == "model"]
    tools = [r for r in rows if r["kind"] == "tool"]
    usage = trace.get("usage", {})
    stages = [s for s in sorted(trace.get("spans", []), key=lambda s:s.get("offset_seconds", 0))
              if s["name"].startswith("stage.")]
    flow = "<span class='arrow'>→</span>".join(
        "<span class='chip'>" + escape(s["name"].removeprefix("stage.")) + "</span>" for s in stages)
    stats = "".join(f"<div class='stat'><b>{escape(str(value))}</b>{label}</div>" for value, label in (
        (len(models), "研究模型调用"), (len(tools), "实际工具执行"),
        (f"{trace.get('elapsed_seconds', 0):.2f}s", "研究总耗时"),
        (f"{usage.get('total_tokens', 0):,}", "研究 Token（含推理用量）"),
    ))
    compression = [e for e in trace.get("events", []) if e["name"] == "compression"]
    compression_html = "".join(details(
        f"+{e.get('offset_seconds', 0):.3f}s · 压缩事件 L{e.get('data', {}).get('level', '?')}", e.get("data", {})
    ) for e in compression)
    judge_html = ""
    if judge_view:
        j = judge_view["trace"]
        judge_html = "<h2>独立 Judge 评审</h2><p class='note'>下方时间从评审开始重新计时；不计入上面的研究调用与耗时。</p>"
        judge_html += f"<p>评审耗时 {j.get('elapsed_seconds', 0):.2f}s · Token {j.get('usage', {}).get('total_tokens', 0):,}</p>"
        judge_html += "<details><summary>展开评审模型输出</summary>" + render_rows(judge_view) + "</details>"
        judge_html += details("最终评审结果", result.get("judge", {}))
    metadata = trace.get("metadata", {})
    mode = "固定证据库快照" if metadata.get("tool_mode") == "snapshot" else str(metadata.get("tool_mode", "未记录"))
    adversary = {True: "开启", False: "关闭"}.get(metadata.get("adversarial_enabled"), "未记录")
    warning = "记录为有界预览，长输出可能含 …&lt;preview&gt;，深层对象可能含 &lt;depth limit&gt;；不展示模型内部推理。完整报告另附。"
    dropped = trace.get("dropped_records")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepResearch 问答轨迹</title><style>{CSS}</style></head><body><main>
<h1>DeepResearch 问答轨迹</h1><p class="meta">样本 {escape(str(result.get('sample_id', trace.get('research_run_id', ''))))} · {escape(str(metadata.get('variant', '')))} · {escape(str(result.get('models', {}).get('default_policy', '')))}</p>
<div class="question"><b>用户问题</b><p>{escape(view['question'])}</p></div>
<div class="stats">{stats}</div><p>工具来源：{escape(mode)} · 对抗阶段：{adversary} · 研究状态：{escape(str(trace.get('status', 'unknown')))}</p>
<div class="flow">{flow}</div><p class="note">{warning} 按跨度开始时间排序；相近时间不代表工具一定并行。跨度状态表示函数是否正常返回。</p>
<h2>逐轮模型输出与工具调用</h2><div class="controls"><button data-filter="all" aria-pressed="true">全部</button><button data-filter="model" aria-pressed="false">仅模型</button><button data-filter="tool" aria-pressed="false">仅工具</button><button id="expand">展开全部详情</button><button id="collapse">收起详情</button></div>
<div class="timeline" id="research-timeline">{render_rows(view)}</div>
<h2>汇总过程与最终报告</h2><div class="panel">{compression_html or '<p>未记录压缩事件。</p>'}{details('完整最终报告（report.md）', report)}{details('追踪完整性信息', {'dropped_records': dropped, 'export_errors': trace.get('export_errors', [])})}</div>
{judge_html}<p class="note">离线展示已保存产物；本页面不发送网络请求，也不重新调用模型。</p>
</main><script>
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{{
 document.querySelectorAll('#research-timeline .card').forEach(card=>{{card.hidden=button.dataset.filter!=='all'&&card.dataset.kind!==button.dataset.filter;}});
 document.querySelectorAll('[data-filter]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
}}));
document.getElementById('expand').addEventListener('click',()=>document.querySelectorAll('details').forEach(d=>d.open=true));
document.getElementById('collapse').addEventListener('click',()=>document.querySelectorAll('details').forEach(d=>d.open=false));
</script></body></html>"""


def render_markdown(view, judge_view):
    """Small IDE-friendly companion; detailed arguments/results remain in HTML."""
    lines = ["# 问答轨迹速览", "", view["question"], "", "```mermaid", "flowchart TD"]
    previous = None
    for index, row in enumerate(view["rows"]):
        name = row['role'] if row['kind'] == 'model' else row['span'].get('inputs', {}).get('tool_name', '工具')
        label = f"{'模型' if row['kind'] == 'model' else '工具'} · {name} · 第{row['round']}轮"
        # Mermaid labels must not interpret arbitrary trace text as diagram syntax.
        label = label.replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', ' ')
        lines.append(f'  n{index}["{label}"]')
        if previous is not None:
            lines.append(f"  n{previous} --> n{index}")
        previous = index
    lines += ["```", "", "箭头表示按开始时间排列的阅读顺序；实际任务依赖见下表，并非并发任务的因果关系。", "",
              "| 开始 / 耗时 | 执行者 | 输出或调用摘要 |", "| --- | --- | --- |"]
    for row in view['rows']:
        s = row['span']
        if row['kind'] == 'model':
            r = row['result'] if isinstance(row['result'], dict) else {}
            summary = str(r.get('content', ''))[:160]
            calls = r.get('tool_calls', [])
            if calls:
                summary += '；请求工具：' + ', '.join(str(c.get('function', {}).get('name')) for c in calls if isinstance(c, dict))
        else:
            summary = pretty(s.get('inputs', {}))
        summary = summary.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '\\|').replace('\n', ' ')
        role = row['role'].replace('|', '\\|').replace('\n', ' ')
        if row['kind'] == 'tool':
            role = '工具 ' + str(s.get('inputs', {}).get('tool_name', '未知')).replace('|', '\\|') + ' · ' + role
        lines.append(f"| +{s.get('offset_seconds', 0):.3f}s / {s.get('duration_seconds', 0):.3f}s | {role} 第{row['round']}轮 | {summary} |")
    tasks = {r['task'].get('task_id'): r['task'] for r in view['rows'] if r['task'].get('task_id')}
    lines += ['', '任务依赖：', '', '```json', pretty({k:v.get('dependencies', []) for k,v in tasks.items()}), '```', '',
              '模型输出和工具结果是本地保存的有界预览，可能截断；完整报告见同目录 report.md。', '',
              '可交互时间线：[trajectory.html](trajectory.html)。所有内容来自现有产物，无新增模型调用。']
    if judge_view:
        lines += ['', f"Judge 独立评审：{len([r for r in judge_view['rows'] if r['kind'] == 'model'])} 次模型调用，"
                  f"耗时 {judge_view['trace'].get('elapsed_seconds', 0):.2f}s，不计入研究阶段。"]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', type=Path, required=True, help='Directory containing trace.json and result.json')
    args = parser.parse_args()
    directory = args.sample_dir
    trace = json.loads((directory / 'trace.json').read_text(encoding='utf-8'))
    view = build_view(trace)
    judge_trace = load_optional(directory / 'judge_trace.json')
    judge_view = build_view(judge_trace) if judge_trace else None
    result = load_optional(directory / 'result.json')
    report_path = directory / 'report.md'
    report = report_path.read_text(encoding='utf-8') if report_path.exists() else '未找到 report.md'
    html_path, md_path = directory / 'trajectory.html', directory / 'trajectory.md'
    html_path.write_text(render_html(view, judge_view, result, report), encoding='utf-8')
    md_path.write_text(render_markdown(view, judge_view), encoding='utf-8')
    print(html_path.resolve())
    print(md_path.resolve())


if __name__ == '__main__':
    main()
