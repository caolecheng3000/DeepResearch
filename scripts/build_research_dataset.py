"""Build the v5 research dataset and review dossiers from archived primary sources.

Offline, deterministic and free of model calls. Run fetch_research_sources.py
--fetch explicitly to acquire missing primary material, never synthesize it here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.research_tasks import DOMAIN_LABELS, TASKS

DATA = ROOT / "evaluation/data"
ARCHIVE = DATA / "research_v5"
REVIEW = ROOT / "docs/research_v5_review"
DATASET = DATA / "research_deepresearch_v5.json"
ANCHORS = DATA / "research_deepresearch_v5_ablation.json"

# PDF extraction can split a contact across lines or group several local parts
# under one domain. Process the complete source before selecting excerpts so an
# excerpt boundary cannot leave part of a contact visible.
EMAIL_CONTACT = re.compile(
    r"(?<![\w.%+-])(?:\{[A-Za-z0-9._%+,\-\s]{1,500}\}|[A-Za-z0-9._%+-]+)"
    r"[ \t\r\n]*@[ \t\r\n]*[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9-])"
)
CONTACT_POLICY = "email-redaction-v1; length-preserving block mask; original offsets and page breaks retained"


def redact_contact_emails(text: str) -> str:
    """Remove standard, grouped and line-wrapped emails without moving offsets.

    All non-whitespace characters of a contact become blocks. Original private
    source archives and their hashes remain unchanged; published quotes must be
    checked against this explicitly documented transformation of the source.
    """
    return EMAIL_CONTACT.sub(
        lambda match: "".join(char if char.isspace() else "█" for char in match[0]), text
    )


def contains_contact_email(value) -> bool:
    """Inspect decoded JSON values as well as plain text (including line breaks)."""
    if isinstance(value, str):
        return EMAIL_CONTACT.search(value) is not None
    if isinstance(value, dict):
        return any(contains_contact_email(key) or contains_contact_email(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(contains_contact_email(item) for item in value)
    return False


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def select_spans(text: str) -> list[dict]:
    """Select email-redacted substrings with original offsets and PDF page numbers.

    Introduction + method + evaluation + discussion/limitations where available.
    A missing heading is recorded; no prose is invented as a replacement.
    """
    public_text = redact_contact_emails(text)
    heading = re.compile(r"(?m)^[ \t]*(?:\d[\d.]*[ \t]+)?([A-Za-z][A-Za-z \t:&-]{2,70})[ \t]*$")
    found = []
    for match in heading.finditer(text):
        compact = re.sub(r"[^a-z]", "", match[1].lower())
        if re.fullmatch(r"(?:methods?|methodology|approach|framework|experiments?|experimental(?:setup|results|evaluation)|evaluation|results(?:anddiscussion)?|limitations?|discussion(?:andconclusions?)?|conclusions?)", compact):
            found.append((match.start(), compact, match[1].strip()))
    candidates = [(0, "正文开头", 2500)]
    for name, terms, size in (
        ("方法", ("method", "approach", "framework"), 2400),
        ("实验或评价", ("experiment", "evaluation", "result"), 2600),
        ("讨论或结论", ("limitation", "discussion", "conclusion"), 1800),
    ):
        match = next((item for item in found if item[0] > 1800 and any(term in item[1] for term in terms)), None)
        if match:
            candidates.append((match[0], f"{name}：{match[2]}", size))
    # If PDF formatting prevents heading recognition, retain additional real text
    # from the early body, clearly labelled rather than claiming it is a section.
    spans = []
    for start, label, size in candidates:
        end = min(len(text), start + size)
        if any(start < old["end"] and end > old["start"] for old in spans):
            continue
        spans.append({"id": f"excerpt_{len(spans) + 1}", "label": label,
                      "start": start, "end": end,
                      "pdf_page_start": text.count("\f", 0, start) + 1,
                      "pdf_page_end": text.count("\f", 0, end) + 1,
                      "quote": public_text[start:end]})
    if len(spans) < 3:
        for start in (3000, 6500, 10000):
            end = min(len(text), start + 2200)
            if start >= len(text) or any(start < old["end"] and end > old["start"] for old in spans):
                continue
            spans.append({"id": f"excerpt_{len(spans) + 1}", "label": "正文连续摘录（未自动识别章节）",
                          "start": start, "end": end,
                          "pdf_page_start": text.count("\f", 0, start) + 1,
                          "pdf_page_end": text.count("\f", 0, end) + 1,
                          "quote": public_text[start:end]})
            if len(spans) >= 3:
                break
    return spans


def material(record: dict) -> dict:
    if digest((ROOT / record["pdf_path"]).read_bytes()) != record["pdf_sha256"]:
        raise ValueError(f"Changed original PDF: {record['id']}")
    text = (ROOT / record["text_path"]).read_text(encoding="utf-8")
    if digest(text.encode()) != record["text_sha256"]:
        raise ValueError(f"Changed full text: {record['id']}")
    spans = select_spans(text)
    body = (f"{record['title']}\n检索主题：{record['title_zh']}\n"
            f"来源：{record['url']}，固定v1论文快照。以下为选段；联系邮箱已用等长█遮盖，不是整篇论文。\n\n"
            f"[摘要（联系邮箱已脱敏）]\n{redact_contact_emails(record['abstract'])}\n")
    for span in spans:
        # Collapse layout whitespace for readability; redacted quotes keep original
        # offsets in excerpt_spans for independent audit of the transformation.
        quote = re.sub(r"[ \t]+", " ", span["quote"]).replace("\f", "\n[分页]\n")
        body += f"\n[{span['id']} | {span['label']} | PDF页{span['pdf_page_start']}-{span['pdf_page_end']}]\n{quote}\n"
    return {"id": record["id"], "title": f"{record['title']} / {record['title_zh']}",
            "url": record["url"], "text": body,
            "provenance": "Pinned primary paper; abstract and selected excerpts with contact emails redacted; original PDF and complete extracted text archived separately.",
            "contact_redaction": CONTACT_POLICY,
            "text_sha256": digest(body.encode()), "excerpt_spans": spans,
            "source_record": str((ARCHIVE / "sources" / f"{record['id']}.json").relative_to(ROOT)),
            "full_text_path": record["text_path"], "pdf_path": record["pdf_path"]}


INSTRUCTIONS = (
    "\n\n交付要求：基于固定论文材料撰写中文研究报告，建议1800—4500字，含可追溯的来源URL；"
    "合同按非空白字符计1800—9000字符，包含Markdown、标点与URL。报告至少包括研究问题与范围、方法或证据比较、"
    "结论边界和待验证实验。可以给出不同的条件性建议，但必须解释证据链、反例或不确定性。"
    "原文中的已测结果、你的推断和拟议设计要明确区分；引用数字必须附任务、模型与测量条件。"
    "工具搜索整个固定论文池；browser返回选定原文摘录而非论文全文，不联网补全；"
    "材料无法支持的结论请列为待验证，不猜测缺失数据。可使用英文方法名与关键词检索。"
    "材料固定为各论文v1历史版本，不要求回答当前最新进展。"
)


def build_payloads() -> tuple[dict, dict, dict]:
    manifest = json.loads((ARCHIVE / "source_manifest.json").read_text())
    if manifest.get("errors"):
        raise ValueError("Source manifest has failed downloads; resolve them before building")
    records = {s["id"]: s for s in manifest["sources"]}
    needed = {s for task in TASKS for s in task["sources"]}
    if needed - records.keys():
        raise ValueError(f"Missing primary sources: {sorted(needed - records.keys())}")
    docs = {key: material(records[key]) for key in sorted(needed)}
    counts = Counter()
    examples, dossiers = [], {}
    for task in TASKS:
        counts[task["category"]] += 1
        case_docs = [docs[key] for key in task["sources"]]
        ground_truth = {topic: text for topic, text, _ in task["points"]}
        # A separate annotation archive maps each point to candidate supporting
        # papers. Those mappings are review drafts, not machine-verified entailment.
        annotations = [dict(id=f"fact_{i + 1}", topic=topic, expected=text,
                            source_ids=sources, verification="draft_requires_span_level_human_review")
                       for i, (topic, text, sources) in enumerate(task["points"])]
        scope = "\n\n本题材料范围（固定v1，可按英文题名或方法名检索）：\n\n" + "\n".join(
            f"- {records[key]['title']}" for key in task["sources"])
        scope += "\n\n本次论证以所列材料为依据；材料未提供的最新进展、数字和实验结果应列为待验证，不能自行补造。"
        example = {"id": task["id"], "category": task["category"],
                   "category_label": DOMAIN_LABELS[task["category"]],
                   "family_id": task["id"], "domain": DOMAIN_LABELS[task["category"]],
                   "split": "development" if counts[task["category"]] == 1 else "evaluation_candidate",
                   "difficulty": "research", "ablation": counts[task["category"]] == 2,
                   "query": task["brief"] + scope + INSTRUCTIONS,
                   "ground_truth": ground_truth, "expected_topics": list(ground_truth),
                   "requirements": {"min_chars": 1800, "max_chars": 9000,
                                    "citation_required": True, "evidence_mode": "fixed_snapshot"},
                   "evidence": [{k: d[k] for k in ("id", "title", "url", "text", "provenance")} for d in case_docs],
                   "reference_doc_urls": [d["url"] for d in case_docs], "memory_seed": [],
                   "annotation_status": "draft_for_user_review", "title": task["title"],
                   "deliverable": task["deliverable"]}
        examples.append(example)
        dossiers[task["id"]] = {
            "id": task["id"], "title": task["title"], "annotation_status": "draft_for_user_review",
            "review_status": "not_reviewed", "question": example["query"],
            "source_ids": task["sources"], "reference_points": annotations,
            "acceptable_alternatives": "不要求唯一技术选型；不同建议必须满足问题约束，并有材料支持、适用条件及可检验理由。",
            "critical_errors": ["编造论文实验或数字", "把跨论文不可比结果当成统一排名", "把拟议系统设计写成论文已经实现的事实", "忽略证据适用范围并给无条件结论"],
            "review_checklist": ["问题需要多来源综合而非单段查找", "原文版本与材料完整性已核对", "每条参考要点有具体原文片段支持或明确属于设计要求", "允许合理替代建议且不存在隐藏答案或引用白名单", "结论边界、反例与实验设计均可评分"],
            "rubric_note": "沿用现有report-rubric-v1：事实与论证、引用、覆盖、清晰度；六条参考要点进入现有逐事实评审。独立的方法学评分器尚未实现，不声称存在新的深度研究总分。",
        }
    payload = {"name": "deepresearch-primary-literature-review", "version": "5.0-draft-email-redacted-v1",
               "status": "draft_for_user_review", "search_mode": "bm25", "snippet_chars": 550,
               "description": "24 distinct research briefs across six domains, using archived primary papers; not a human-certified held-out benchmark.",
               "protocol": {"retrieval": "shared primary-paper excerpt corpus; no live web",
                            "contact_redaction": CONTACT_POLICY,
                            "split": "six development and eighteen evaluation candidates; public papers shared across splits, not leakage-free or blinded holdout",
                            "review_state": "not yet human reviewed; no paid model experiments run",
                            "memory": "no private seeded answers; no_memory here cannot alone certify cross-session memory benefits"},
               "documents": [{k: d[k] for k in ("id", "title", "url", "text", "provenance")} for d in docs.values()],
               "examples": examples}
    return payload, dossiers, docs


def validate(payload: dict, dossiers: dict, documents: dict) -> dict:
    from evaluation.experiments import validate_dataset
    validate_dataset(payload)
    if len(payload["examples"]) != 24 or set(Counter(q["category"] for q in payload["examples"]).values()) != {4}:
        raise ValueError("Expected 24 distinct briefs, four per domain")
    source_ids = set(documents)
    for q in payload["examples"]:
        if len(q["evidence"]) < 3 or len(q["ground_truth"]) < 6 or "allowed_urls" in q["requirements"]:
            raise ValueError(f"Insufficient evidence or hidden whitelist: {q['id']}")
        archive = dossiers[q["id"]]
        for point in archive["reference_points"]:
            if not point["source_ids"] or not set(point["source_ids"]) <= set(archive["source_ids"]):
                raise ValueError(f"Unmapped annotation: {q['id']}/{point['id']}")
        if not {d["id"] for d in q["evidence"]} <= source_ids:
            raise ValueError("Missing runtime material")
    span_count = 0
    redacted_span_count = 0
    for doc in documents.values():
        text = (ROOT / doc["full_text_path"]).read_text()
        public_text = redact_contact_emails(text)
        if doc["contact_redaction"] != CONTACT_POLICY:
            raise ValueError(f"Missing contact redaction provenance: {doc['id']}")
        for span in doc["excerpt_spans"]:
            start, end = span["start"], span["end"]
            if public_text[start:end] != span["quote"]:
                raise ValueError(f"Excerpt mismatch: {doc['id']}")
            redacted_span_count += text[start:end] != span["quote"]
            span_count += 1
    if contains_contact_email(payload) or contains_contact_email(documents):
        raise ValueError("Contact email remains in published research material")
    return {"status": "offline_integrity_passed", "cases": len(payload["examples"]),
            "domains": dict(Counter(q["category"] for q in payload["examples"])),
            "primary_sources": len(documents), "verified_source_spans": span_count,
            "unchanged_source_spans": span_count - redacted_span_count,
            "email_redacted_source_spans": redacted_span_count,
            "contact_redaction": CONTACT_POLICY,
            "reference_points": sum(len(q["ground_truth"]) for q in payload["examples"]),
            "annotation_review": "pending_user_review; redaction-aware source equality does not establish semantic entailment",
            "model_experiments": "not_run"}


def dossier_markdown(task, example, dossier, documents):
    lines = [f"# {task['title']}", "", f"档案ID：`{task['id']}`。状态：待用户审查，尚无模型实验结果。", "",
             "## 题目", "", example["query"], "", "## 材料", "",
             "下列是研究者可搜索、阅读的原文摘录。PDF和完整提取文本供审查；运行时browser读取的是材料JSON中的选段，不是所有页。", ""]
    for key in task["sources"]:
        doc = documents[key]
        record = json.loads((ROOT / doc["source_record"]).read_text())
        lines += [f"### {key}：{record['title']}", "",
                  f"[原始论文（固定v1）]({doc['url']}) · [本地PDF](../../{doc['pdf_path']}) · [完整提取文本](../../{doc['full_text_path']}) · [来源档案](../../{doc['source_record']}) · [运行材料](../../evaluation/data/research_v5/materials/{key}.json)", "",
                  f"PDF SHA256：`{record['pdf_sha256']}`。下载时间：{record['retrieved_at']}。", "",
                  "<details>", "<summary>展开研究者可读取的材料正文</summary>", "", "```text", doc["text"], "```", "", "</details>", ""]
    lines += ["## 标注档案（仅供裁判与审查）", "", "以下是参考要点与设计要求，允许有证据支持的替代结论；不是一份必须逐字复述的唯一答案。论文映射仍需逐条人工核验，离线脚本只验证来源和摘录一致性。", "",
              "|编号|研究要点|候选依据|", "|---|---|---|"]
    for p in dossier["reference_points"]:
        lines.append(f"|{p['id']} · {p['topic']}|{p['expected']}|{', '.join(p['source_ids'])}|")
    lines += ["", "允许替代：" + dossier["acceptable_alternatives"], "", "关键错误：" + "；".join(dossier["critical_errors"]) + "。", "",
              "评分接入：" + dossier["rubric_note"], "", "审查清单：", ""]
    lines += ["- [ ] " + item for item in dossier["review_checklist"]]
    lines += ["", "审查记录：审查人／日期／通过或修改／理由：待填写。", "",
              f"[机器可读完整档案](../../evaluation/data/research_v5/dossiers/{task['id']}.json)", ""]
    return "\n".join(lines)


def write_all():
    payload, dossiers, docs = build_payloads()
    audit = validate(payload, dossiers, docs)
    for folder in (ARCHIVE / "materials", ARCHIVE / "dossiers", REVIEW):
        folder.mkdir(parents=True, exist_ok=True)
    DATASET.write_text(encoded(payload))
    ANCHORS.write_text(encoded({**payload, "name": payload["name"] + "-ablation",
                               "examples": [q for q in payload["examples"] if q["ablation"]]}))
    for key, doc in docs.items():
        (ARCHIVE / "materials" / f"{key}.json").write_text(encoded(doc))
    for key, dossier in dossiers.items():
        (ARCHIVE / "dossiers" / f"{key}.json").write_text(encoded(dossier))
    examples = {q["id"]: q for q in payload["examples"]}
    for task in TASKS:
        key = task["id"]
        (REVIEW / f"{key}.md").write_text(dossier_markdown(task, examples[key], dossiers[key], docs))
    lines = ["# 深度研究题集 v5：题目—材料—档案审查入口", "",
             "24个不同研究问题、6个领域，基于31篇真实论文的固定v1版本；当前是待用户审查草案，没有运行付费模型或宣称新的研究分数。", "",
             "每份档案包含完整题目、研究者可见材料、原始PDF及全文、参考要点、来源映射和审查清单。建议先审查下面六道，再查看全部题目。", "",
             "## 建议先审查的六道样例", ""]
    for task in TASKS[::4]:
        lines.append(f"- [{task['title']}](research_v5_review/{task['id']}.md)：{DOMAIN_LABELS[task['category']]}。")
    lines += ["", "## 全部题目", "", "|档案|领域|交付物|材料数|划分|", "|---|---|---|---:|---|"]
    for task in TASKS:
        q = examples[task["id"]]
        lines.append(f"|[{task['title']}](research_v5_review/{task['id']}.md)|{DOMAIN_LABELS[task['category']]}|{task['deliverable']}|{len(task['sources'])}|{q['split']}|")
    lines += ["", "## 版本边界", "",
              "v3/v4保留为短能力探针及历史实验口径；v5面向文献研究，不覆盖旧文件。6道development与18道evaluation_candidate只是用途标记，原始材料跨划分共享，尚未形成盲测或无泄漏保留集。", "",
              "材料来自真实论文原文，未添加参考答案或编造实验数据。研究者只接收query和固定材料工具；标注档案供裁判与人工审查。144条参考要点含论文结论和设计验收要求，仍须核对，不能把来源存在、摘录一致视作事实已认证。", "",
              "当前沿用report-rubric-v1与报告契约；研究方法比较、边界和实验设计通过参考要点评审进入现有评分。尚未实现独立的深度研究量表；规则key覆盖分仍仅是诊断，不代表研究水平。", "",
              "[数据卡与使用说明](EVALUATION_DATASET_V5.md) · [离线完整性检查](../evaluation/data/research_v5/integrity_report.json)", ""]
    (ROOT / "docs/RESEARCH_DATASET_V5_REVIEW.md").write_text("\n".join(lines))
    audit["dataset_sha256"] = digest(DATASET.read_bytes())
    (ARCHIVE / "integrity_report.json").write_text(encoded(audit))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify current files match their offline deterministic build")
    args = parser.parse_args()
    if args.check:
        payload, dossiers, docs = build_payloads()
        audit = validate(payload, dossiers, docs)
        if not DATASET.exists() or DATASET.read_text() != encoded(payload):
            raise ValueError("Dataset differs from authored questions and archived sources; rebuild first")
        expected_anchors = {**payload, "name": payload["name"] + "-ablation",
                            "examples": [q for q in payload["examples"] if q["ablation"]]}
        if ANCHORS.read_text() != encoded(expected_anchors):
            raise ValueError("Ablation dataset is stale")
        for key, doc in docs.items():
            if (ARCHIVE / "materials" / f"{key}.json").read_text() != encoded(doc):
                raise ValueError(f"Runtime material archive is stale: {key}")
        examples = {q["id"]: q for q in payload["examples"]}
        for task in TASKS:
            key = task["id"]
            if (ARCHIVE / "dossiers" / f"{key}.json").read_text() != encoded(dossiers[key]):
                raise ValueError(f"Annotation archive differs from authored definition: {key}")
            expected_page = dossier_markdown(task, examples[key], dossiers[key], docs)
            if (REVIEW / f"{key}.md").read_text() != expected_page:
                raise ValueError(f"Review page is stale: {key}")
    else:
        audit = write_all()
    print(encoded(audit))


if __name__ == "__main__":
    main()
