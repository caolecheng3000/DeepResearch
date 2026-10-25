"""Uniform v4 report-only re-evaluation. Original generation and scores stay immutable.

Run after all research stages finish; resume validated case checkpoints, never
select cases by their score. One retry is permitted ONLY for invalid Judge output.
"""
import sys
from pathlib import Path
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evaluation.experiments import source_revision, infrastructure_error, EvaluationInfrastructureError
from src.core.judge import LLMJudge
from src.core.report_evaluation import report_contract, normalize_evidence
from src.core.runner import load_config
from src.utils.execution import deadline_scope, bounded, run_sync
from src.utils.tracing import research_trace, flush_traces

def grading_identity():
    from src.models.model_router import ModelRouter
    from src.utils.env_config import ensure_env_loaded
    ensure_env_loaded()
    config=ModelRouter._load_backend_config('deepseek')
    files=('src/core/judge.py','src/core/report_evaluation.py','src/models/vllm_policy.py')
    return {'requested_model':config['model_name'],'provider_host':urlsplit(config['base_url']).hostname,
        'grader_code_hash':hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in files)).hexdigest()}


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")

def digest(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def inputs(base):
    base=base.resolve()
    v4=json.loads((ROOT/"evaluation/data/research_capability_v4.json").read_text())
    revised={q["id"]:q for q in v4["examples"]}
    cases=[]
    for stage in ("controlled","ablations","hotpot","positive_controls","positive_memory_control"):
        directory=base/stage
        if not (directory/"results.jsonl").exists():continue
        data=json.loads((directory/"dataset.json").read_text())
        questions={q["id"]:q for q in data["examples"]}
        for line in (directory/"results.jsonl").read_text().splitlines():
            old=json.loads(line);q=questions[old["question_id"]]
            if stage in ("controlled","ablations"):
                q=revised[q["id"]]
            elif stage.startswith("positive_"):
                q=json.loads(json.dumps(q));q["requirements"].pop("allowed_urls",None)
                q["reference_doc_urls"]=[d["url"] for d in q["evidence"]]
                draft=next(d for d in data["documents"] if d["url"].endswith('/'+q['id']+'/draft'))
                q["evidence"]=[*q["evidence"],draft]
            sample=directory/Path(old["report_path"]).parent
            common={"stage":stage,"sample_id":old["sample_id"],"question_id":q["id"],
                "variant":old["variant"],"research_status":old["status"],"category":q.get("category"),
                "question":q,"original_result":old}
            for version,filename in (("final","report_body.md"),("before_m5","report_before_adversarial.md")):
                if version=="before_m5" and not (stage=="controlled" and old.get("ablation")):continue
                path=sample/filename
                if not path.exists():continue
                body=path.read_text()
                cases.append({**common,"id":f"{stage}/{old['sample_id']}/{version}","version":version,
                    "report":body,"report_sha256":hashlib.sha256(body.encode()).hexdigest(),
                    "report_path":str(path.relative_to(ROOT))})
    return cases

async def run(args):
    if json.loads((args.base/"suite_manifest.json").read_text())["status"]!="completed":
        raise ValueError("Primary research must finish before final uniform evaluation")
    cases=inputs(args.base)
    output=args.base/"uniform_report_v4"
    settings=load_config("configs/evaluation_capability.yaml")["model"]["backend_sampling"]["modules"]["judge"]
    identity={"input_hash":digest(cases),"num_cases":len(cases),"judge_settings":settings,**grading_identity(),
        "evidence_protocol":"capability-v4-local-three-documents-no-hidden-gold-url-ban",
        "rubric_version":"report-rubric-v1","contract_version":"report-contract-v2",
        "research_rerun":False,"retry_policy":"at most once, only when initial output invalid; keep both",
        "created_at":datetime.now(timezone.utc).isoformat(),**source_revision()}
    if output.exists():
        old=json.loads((output/"manifest.json").read_text())
        if any(old[k]!=identity[k] for k in ("input_hash","judge_settings","evidence_protocol","requested_model","provider_host","grader_code_hash")):
            raise ValueError("Re-evaluation checkpoint inputs changed")
        identity=old
    else:
        (output/"samples").mkdir(parents=True)
    identity["status"]="running";dump(output/"manifest.json",identity)
    sem=asyncio.Semaphore(4);stopped=asyncio.Event()
    async def evaluate(case):
        async with sem:
            if stopped.is_set():return
            directory=output/"samples"/hashlib.sha256(case["id"].encode()).hexdigest()[:20]
            directory.mkdir(exist_ok=True)
            result_path=directory/"result.json"
            if result_path.exists():
                saved=json.loads(result_path.read_text())
                if saved["input_hash"]!=digest(case):raise ValueError("Changed case input")
                if saved.get("completed"):return
            dump(directory/"input.json",case)
            q=case["question"];body=case["report"]
            record={k:case[k] for k in ("id","stage","sample_id","question_id","variant","research_status","category","version","report_path","report_sha256")}
            record.update({"input_hash":digest(case),"contract":report_contract(body,q.get("requirements")),"attempts":[]})
            if not body.strip():
                record.update({"evaluation":{"error":"empty report"},"completed":True,"initial_valid":False,"valid":False})
                dump(result_path,record);return
            for index in range(2):
                judge=LLMJudge(backend="deepseek",**settings)
                with research_trace({"phase":"uniform_report_v4","example_id":case["id"],"attempt":index}) as trace:
                    try:
                        with deadline_scope(seconds=180):
                            evaluation=await bounded(run_sync(judge.score_report,body,q["query"],q.get("ground_truth"),q.get("requirements"),normalize_evidence(q.get("evidence",[]))))
                    except Exception as exc:
                        evaluation={"error":f"{type(exc).__name__}: {exc}"}
                snapshot=trace.snapshot();dump(directory/f"judge_trace_{index}.json",snapshot)
                record["attempts"].append({"evaluation":evaluation,"usage":snapshot["usage"]})
                record["evaluation"]=evaluation
                resource=infrastructure_error({"judge":evaluation},snapshot)
                if resource:
                    record["infrastructure_error"]=resource;record["completed"]=False
                    dump(result_path,record);stopped.set();raise EvaluationInfrastructureError(resource)
                if "error" not in evaluation:break
            record.update({"completed":True,"initial_valid":"error" not in record["attempts"][0]["evaluation"],
                "valid":"error" not in record["evaluation"]})
            dump(result_path,record)
            print(case["id"],"quality",record["evaluation"].get("average"),"attempts",len(record["attempts"]),flush=True)
    tasks=[asyncio.create_task(evaluate(c)) for c in cases]
    try:
        await asyncio.gather(*tasks);identity["status"]="completed"
    except BaseException:
        identity["status"]="interrupted";raise
    finally:
        for t in tasks:
            if not t.done():t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        rows=[json.loads(p.read_text()) for p in sorted((output/"samples").glob("*/result.json"))]
        dump(output/"results.json",rows)
        summary={"num_expected":len(cases),"num_completed":sum(r.get("completed",False) for r in rows),
            "initial_valid":sum(r.get("initial_valid",False) for r in rows),"valid_after_bounded_retry":sum(r.get("valid",False) for r in rows),
            "retry_cases":sum(len(r["attempts"])>1 for r in rows),
            "recorded_tokens":sum(a["usage"].get("total_tokens",0) for r in rows for a in r["attempts"]),
            "quality_valid_counts_by_stage":dict(Counter(r["stage"] for r in rows if r.get("valid")))}
        dump(output/"summary.json",summary);dump(output/"manifest.json",identity);flush_traces()
        print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,required=True)
    asyncio.run(run(p.parse_args()))
