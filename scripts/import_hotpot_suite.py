"""Import a deterministic, stratified PUBLIC dev subset; never use mock fallback."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]


def build(path, count=24):
    import pyarrow.parquet as pq
    rows=pq.read_table(path).to_pylist()
    buckets=defaultdict(list)
    for row in rows:
        buckets[(row["type"],row["level"])].append(row)
    for group in buckets.values():
        group.sort(key=lambda r:hashlib.sha256(("deepr-capability-v1:"+r["id"]).encode()).hexdigest())
    selected=[]
    while len(selected)<count:
        progress=False
        for key in sorted(buckets):
            if buckets[key] and len(selected)<count:
                selected.append(buckets[key].pop(0));progress=True
        if not progress:raise ValueError("Not enough genuine HotpotQA records")
    examples=[]; documents={}
    for row in selected:
        docs=[]
        for title,sentences in zip(row["context"]["title"],row["context"]["sentences"]):
            url="https://en.wikipedia.org/wiki/"+quote(title.replace(" ","_"),safe="()_")
            doc={"id":hashlib.sha256(url.encode()).hexdigest()[:16],"title":title,"url":url,
                 "text":"".join(sentences),"provenance":"HotpotQA public dev Wikipedia context snapshot (2018 benchmark); not live web"}
            if url in documents and documents[url]["text"]!=doc["text"]:
                raise ValueError("Same article has incompatible snapshots; choose explicit snapshot IDs")
            documents[url]=doc;docs.append(doc)
        supporting=row["supporting_facts"]
        support_titles=set(supporting["title"]);support=[]
        for title,index in zip(supporting["title"],supporting["sent_id"]):
            titles=row["context"]["title"]
            sentences=row["context"]["sentences"][titles.index(title)]
            if 0<=index<len(sentences):support.append(sentences[index])
        examples.append({"id":"hotpot_"+row["id"],"original_id":row["id"],
            "category":"hotpot_"+row["type"],"family_id":row["id"],"split":"public_dev_evaluation",
            "difficulty":row["level"],"benchmark":"HotpotQA public dev adapted research protocol",
            "query":row["question"]+"\n请根据固定证据库检索并回答，必要时使用 browser 阅读正文。第一行单独写“答案：<简短答案>”，保留原始英文实体拼写或 yes/no；随后用中文解释多跳或比较依据，并引用来源 URL。正文不超过 2200 个非空白字符（含 Markdown 与 URL），最多四个子任务。材料为历史快照，不查询当下事实。",
            "ground_truth":{"答案":row["answer"],"支持依据":"依据以下材料连接或比较得出答案："+" ".join(support)},
            "short_answer":row["answer"],"expected_topics":list(support_titles),
            "reference_doc_urls":[d["url"] for d in docs if d["title"] in support_titles],
            "requirements":{"max_chars":2200,"citation_required":True,"evidence_mode":"fixed_snapshot"},
            "evidence":docs})
    data={"name":"HotpotQA-public-dev-adapted","version":"1","search_mode":"bm25","snippet_chars":550,
        "provenance":{"homepage":"https://hotpotqa.github.io/","mirror":"https://huggingface.co/datasets/hotpotqa/hotpot_qa",
            "license":"CC BY-SA 4.0","raw_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "sampling":"round-robin by (type, level), hash order seed deepr-capability-v1",
            "protocol":"shared retrieved context pool; Chinese report with explicit English answer line; not official leaderboard protocol"},
        "documents":list(documents.values()),"examples":examples}
    output=ROOT/"evaluation/data/research_hotpotqa_v1.json"
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(f"Imported {len(examples)} real questions, {len(documents)} documents; types/levels: "+
          str({str(k):sum((r['category'].removeprefix('hotpot_'),r['difficulty'])==k for r in examples) for k in sorted(buckets)}))


if __name__=="__main__":
    build(ROOT/"evaluation/data/external/hotpotqa_validation.parquet")
