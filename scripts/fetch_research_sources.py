"""Fetch pinned primary research papers for the v5 review corpus; never calls an LLM.

Network access is explicit (--fetch). Existing verified artifacts are reused.
PDF extraction uses the system pdftotext executable. No fabricated fallback.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evaluation/data/research_v5"
PAPERS = {
    "rag": ("2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "检索增强生成 RAG"),
    "dpr": ("2004.04906", "Dense Passage Retrieval for Open-Domain Question Answering", "稠密检索 DPR"),
    "selfrag": ("2310.11511", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "自适应检索与反思 Self-RAG"),
    "lost": ("2307.03172", "Lost in the Middle: How Language Models Use Long Contexts", "长上下文位置偏差"),
    "raptor": ("2401.18059", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", "层次摘要检索 RAPTOR"),
    "memgpt": ("2310.08560", "MemGPT: Towards LLMs as Operating Systems", "分层持久记忆 MemGPT"),
    "react": ("2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models", "推理与工具行动 ReAct"),
    "toolformer": ("2302.04761", "Toolformer: Language Models Can Teach Themselves to Use Tools", "工具调用自监督 Toolformer"),
    "reflexion": ("2303.11366", "Reflexion: an autonomous agent with dynamic memory and self-reflection", "语言反馈与情景记忆 Reflexion（v1标题）"),
    "swebench": ("2310.06770", "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", "真实软件问题基准 SWE-bench"),
    "agentbench": ("2308.03688", "AgentBench: Evaluating LLMs as Agents", "多环境智能体测评 AgentBench"),
    "webarena": ("2307.13854", "WebArena: A Realistic Web Environment for Building Autonomous Agents", "网站智能体环境 WebArena"),
    "truthfulqa": ("2109.07958", "TruthfulQA: Measuring How Models Mimic Human Falsehoods", "真实性评测 TruthfulQA"),
    "factscore": ("2305.14251", "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation", "长文本原子事实评测 FActScore"),
    "mtbench": ("2306.05685", "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", "LLM裁判与偏好评测"),
    "ragas": ("2309.15217", "RAGAS: Automated Evaluation of Retrieval Augmented Generation", "RAG无参考评测 RAGAS"),
    "checklist": ("2005.04118", "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList", "行为测试与最小对比 CheckList"),
    "lora": ("2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", "低秩参数高效微调 LoRA"),
    "qlora": ("2305.14314", "QLoRA: Efficient Finetuning of Quantized LLMs", "量化微调 QLoRA"),
    "flash": ("2205.14135", "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "注意力IO优化 FlashAttention"),
    "paged": ("2309.06180", "Efficient Memory Management for Large Language Model Serving with PagedAttention", "推理服务KV缓存 PagedAttention"),
    "speculative": ("2211.17192", "Fast Inference from Transformers via Speculative Decoding", "推测解码 Speculative Decoding"),
    "chinchilla": ("2203.15556", "Training Compute-Optimal Large Language Models", "训练计算最优缩放 Chinchilla"),
    "dp": ("1607.00133", "Deep Learning with Differential Privacy", "差分隐私深度学习"),
    "membership": ("1610.05820", "Membership Inference Attacks against Machine Learning Models", "成员推断攻击"),
    "extraction": ("2012.07805", "Extracting Training Data from Large Language Models", "语言模型训练数据提取"),
    "injection": ("2302.12173", "More than you've asked for: A Comprehensive Analysis of Novel Prompt Injection Threats to Application-Integrated Large Language Models", "外部材料间接提示注入（v1标题）"),
    "energy": ("1906.02243", "Energy and Policy Considerations for Deep Learning in NLP", "NLP训练能耗与研究成本"),
    "carbon": ("2104.10350", "Carbon Emissions and Large Neural Network Training", "大型模型碳排放核算"),
    "bloom": ("2211.02001", "Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model", "BLOOM生命周期碳足迹"),
    "greenai": ("1907.10597", "Green AI", "绿色AI与效率报告"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_one(item):
    import requests
    from bs4 import BeautifulSoup

    key, (identifier, expected_title, zh) = item
    folder = FOLDER / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / f"{key}.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        for field, hash_field in (("text_path", "text_sha256"), ("pdf_path", "pdf_sha256")):
            if sha((ROOT / record[field]).read_bytes()) != record[hash_field]:
                raise ValueError(f"Cached artifact changed: {key}/{field}")
        return record
    version = identifier + "v1"
    landing = f"https://arxiv.org/abs/{version}"
    pdf_url = f"https://arxiv.org/pdf/{version}"
    session = requests.Session()
    def get(url):
        failure = None
        for attempt in range(2):
            try:
                r = session.get(url, timeout=(10, 45))
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                failure = exc
                if attempt == 0:
                    time.sleep(1)
        raise failure
    try:
        page = get(landing)
        soup = BeautifulSoup(page.content, "html.parser")
        title_meta = soup.find("meta", attrs={"name": "citation_title"})
        if title_meta is None:
            raise ValueError(f"No paper title: {key}")
        title = title_meta.get("content", "")
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        if norm(title) != norm(expected_title):
            raise ValueError(f"Title mismatch for {key}: {title!r}")
        abstract = soup.select_one("blockquote.abstract")
        if abstract is None:
            raise ValueError(f"No abstract: {key}")
        license_link = soup.select_one(".license a")
        if license_link is None:
            license_link = soup.find("a", href=re.compile(r"creativecommons.org/licenses|arxiv.org/licenses"))
        response = get(pdf_url)
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"Not PDF: {key}")
        pdf_path = folder / f"{key}.pdf"
        text_path = folder / f"{key}.txt"
        with tempfile.TemporaryDirectory(prefix="deepr-source-") as tmp:
            draft = Path(tmp) / "source.pdf"
            draft.write_bytes(response.content)
            converted = subprocess.run(["pdftotext", str(draft), "-"],
                                       check=True, capture_output=True).stdout.decode("utf-8")
        if len(converted.strip()) < 8000:
            raise ValueError(f"Insufficient full text: {key}")
        pdf_path.write_bytes(response.content)
        text_path.write_text(converted, encoding="utf-8")
        record = {
            "id": key, "title": title, "title_zh": zh, "arxiv_id": identifier,
            "version": "v1", "url": landing, "pdf_url": pdf_url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "abstract": abstract.get_text(" ", strip=True),
            "license_url": license_link.get("href") if license_link else None,
            "license_note": "Original publisher/author terms apply; no blanket corpus redistribution licence is asserted.",
            "pdf_path": str(pdf_path.relative_to(ROOT)), "pdf_sha256": sha(response.content),
            "text_path": str(text_path.relative_to(ROOT)), "text_sha256": sha(converted.encode()),
            "extraction": "pdftotext reading order; full paper, page boundaries retained as form-feed; inspect PDF for tables/equations",
            "text_chars": len(converted), "source_kind": "primary_research_paper",
        }
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        return record
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Allow HTTP downloads of pinned public papers")
    args = parser.parse_args()
    if not args.fetch:
        parser.error("Pass --fetch to download papers; offline rebuilding uses build_research_dataset.py")
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(key, pool.submit(fetch_one, (key, data))) for key, data in PAPERS.items()]
        for key, future in futures:
            try:
                record = future.result()
                records.append(record)
                print(f"OK {key}: {record['text_chars']} chars", flush=True)
            except Exception as exc:
                errors.append({"id": key, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {key}: {exc}", flush=True)
    manifest = {"source_policy": "Pinned arXiv v1 primary papers; never fabricated", "sources": records, "errors": errors}
    (FOLDER / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise SystemExit(f"{len(errors)} source downloads failed; corpus is incomplete")


if __name__ == "__main__":
    main()
