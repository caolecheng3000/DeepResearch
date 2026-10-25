"""Audit saved capability experiments and generate an analysis; optional pre-M5 Judge."""
import argparse
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.experiments import summarize, infrastructure_error, EvaluationInfrastructureError
from evaluation.qa_metrics import extract_answer,answer_metrics
from evaluation.metrics.stats import bootstrap_ci_paired
from src.core.judge import LLMJudge
from src.utils.tracing import research_trace
from src.utils.execution import deadline_scope


def collect(root):
    rows=[]
    for stage in ("controlled","ablations","hotpot"):
        directory=root/stage
        if not (directory/"results.jsonl").exists():continue
        dataset=json.loads((directory/"dataset.json").read_text())
        questions={q["id"]:q for q in dataset["examples"]}
        for line in (directory/"results.jsonl").read_text().splitlines():
            r=json.loads(line);r["_stage"]=stage;r["_directory"]=str((directory/r["report_path"]).parent)
            r["_question"]=questions[r["question_id"]];rows.append(r)
    return rows


def review_before(root,rows,judge):
    path=root/"pre_adversarial_reviews.json"
    saved=json.loads(path.read_text()) if path.exists() else []
    seen={r["sample_id"] for r in saved}
    for row in rows:
        if row["_stage"]!="controlled" or not row.get("ablation") or row["sample_id"] in seen:continue
        directory=Path(row["_directory"]);body=directory/"report_before_adversarial.md"
        if not body.exists():continue
        q=row["_question"]
        with research_trace({"phase":"pre_adversarial_evaluation","sample_id":row["sample_id"]}) as trace:
            with deadline_scope(seconds=120):
                result=judge.score_report(body.read_text(),q["query"],q.get("ground_truth"),q.get("requirements"),q.get("evidence"))
        snap=trace.snapshot()
        (directory/"pre_judge_trace.json").write_text(json.dumps(snap,ensure_ascii=False,indent=2))
        saved.append({"sample_id":row["sample_id"],"question_id":row["question_id"],"evaluation":result,"usage":snap["usage"]})
        path.write_text(json.dumps(saved,ensure_ascii=False,indent=2)+"\n")
        print("pre-M5 reviewed",len(saved),row["question_id"],result.get("average"),flush=True)
        resource=infrastructure_error({"judge":result},snap)
        if resource:raise EvaluationInfrastructureError(resource)


def analyze(root):
    rows=collect(root);checks=[];retrieval=[];module_usage=defaultdict(lambda:Counter())
    def check(name,ok):checks.append({"name":name,"passed":bool(ok)})
    family=defaultdict(list); failures=[];short=[]
    for row in rows:
        label=row["variant"]+"/"+row["question_id"]
        directory=Path(row["_directory"]);trace=json.loads((directory/"trace.json").read_text())
        spans=trace.get("spans",[]);events=trace.get("events",[]);byid={s["span_id"]:s for s in spans}
        check(label+":span_links",len(byid)==len(spans) and all(not s.get("parent_span_id") or s["parent_span_id"] in byid for s in spans))
        check(label+":usage",row.get("usage",{}).get("total_tokens",0)==sum(e["data"].get("total_tokens",0) for e in events if e["name"]=="llm_usage"))
        check(label+":round_cap",row.get("adversarial_rounds",0)<=3 and sum(s["name"]=="red_agent.attack" for s in spans)<=3)
        budgets=[e["data"] for e in events if e["name"]=="blue_repair_budget"]
        check(label+":blue_global_cap",all(b["selected"]<=5 and b["selected"]+b["deferred"]==b["total_issues"] for b in budgets))
        check(label+":thinking_profile",all(e["data"].get("thinking_mode")=="disabled" for e in events if e["name"]=="llm_request"))
        names=Counter(s["name"] for s in spans)
        if row["variant"]=="no_adversarial":check(label+":m5_off",not names["stage.adversarial"])
        if row["variant"]=="no_compressor":check(label+":m3_off",not names["compressor.collect_l2"])
        if row["variant"]=="no_memory":check(label+":m4_off",not names["memory.put"])
        if row["variant"]=="no_recon":check(label+":recon_off",not names["planner.search_tool"])
        timeline=[]
        for s in spans:
            if s["name"]=="researcher.run":timeline.extend([(s["offset_seconds"],1),(s["offset_seconds"]+s["duration_seconds"],-1)])
        active=peak=0
        for _,delta in sorted(timeline):active+=delta;peak=max(peak,active)
        check(label+":worker_cap",peak<=(1 if row["variant"]=="serial_workers" else 3))
        seen_urls={u for e in events if e["name"]=="snapshot_search" for u in e["data"].get("urls",[])}
        read_urls={e["data"]["url"] for e in events if e["name"]=="snapshot_read"}
        q=row["_question"];seeds={d["url"] for d in q.get("memory_seed",[])}
        gold=set(q.get("reference_doc_urls",[d["url"] for d in q.get("evidence",[])]))-seeds
        retrieval.append({"sample_id":row["sample_id"],"variant":row["variant"],"category":row.get("category"),
            "gold_recall_search":len(gold&seen_urls)/len(gold) if gold else None,
            "gold_recall_read":len(gold&read_urls)/len(gold) if gold else None,"worker_peak":peak,
            "adversarial_status":row.get("adversarial_status"),"blue_selected":sum(b["selected"] for b in budgets)})
        for e in events:
            if e["name"]!="llm_usage":continue
            s=byid.get(e.get("span_id"));role="other";visited=set()
            while s and s["span_id"] not in visited:
                visited.add(s["span_id"]);name=s["name"]
                if name.startswith(("red_agent.","blue_agent.","planner.","summarizer.","compressor.")):
                    role=name.split(".")[0];break
                if name=="researcher.run":role="worker";break
                s=byid.get(s.get("parent_span_id"))
            key=row["variant"]+"/"+role
            module_usage[key]["calls"]+=1;module_usage[key]["tokens"]+=e["data"].get("total_tokens",0)
        if row.get("judge_status")!="success" or row["status"]!="success":
            failures.append({"sample_id":row["sample_id"],"variant":row["variant"],"question_id":row["question_id"],
                "status":row["status"],"judge_status":row.get("judge_status"),"error":row.get("judge",{}).get("error",row.get("error"))})
        if "short_answer" in q:
            body=(directory/"report_body.md").read_text()
            prediction=extract_answer(body)
            short.append({"sample_id":row["sample_id"],"prediction":prediction,"gold":q["short_answer"],
                **answer_metrics(prediction,q["short_answer"])})
    full=[r for r in rows if r["variant"]=="full"]
    paired=[r for r in rows if r["_stage"]=="ablations" or r["_stage"]=="controlled" and r.get("ablation")]
    by_category={c:summarize([r for r in full if r.get("category")==c])["variants"] for c in sorted({r.get("category","unknown") for r in full})}
    before_path=root/"pre_adversarial_reviews.json"
    before=json.loads(before_path.read_text()) if before_path.exists() else []
    full_byid={r["sample_id"]:r for r in full};diffs=[]
    for r in before:
        after=full_byid[r["sample_id"]].get("judge",{})
        if "average" in r["evaluation"] and "average" in after:diffs.append(after["average"]-r["evaluation"]["average"])
    pre_stats=bootstrap_ci_paired(diffs,n_bootstrap=2000)
    pre_stats["bootstrap_nonpositive_fraction"]=pre_stats.pop("p_value")
    pre_stats["ci_positive"]=pre_stats.pop("significant")
    raw_summary=summarize(paired) if paired else {}
    result={"num_research_attempts":len(rows),"num_full_reports":len(full),
        "by_source":{s:summarize([r for r in full if r["_stage"]==s]) for s in ("controlled","hotpot") if any(r["_stage"]==s for r in full)},
        "ablation_matched":raw_summary,"by_category":by_category,"artifact_checks":checks,
        "artifact_check_failures":[c for c in checks if not c["passed"]],"operational_failures":failures,
        "retrieval":retrieval,"module_usage":dict(module_usage),"pre_post_m5":pre_stats,
        "qa":short,"qa_summary":{"num":len(short),"answer_line_present":sum(r["prediction"] is not None for r in short),
           "exact_match":sum(r["exact_match"] for r in short)/len(short) if short else None,
           "f1":sum(r["f1"] for r in short)/len(short) if short else None},
        "research_tokens":sum(r.get("usage",{}).get("total_tokens",0) for r in rows),
        "judge_tokens":sum(r.get("judge_usage",{}).get("total_tokens",0) for r in rows),
        "pre_judge_tokens":sum(r.get("usage",{}).get("total_tokens",0) for r in before),
        "scope":"Descriptive one-run results. Synthetic siblings share templates; matched ablation uses one question per capability cluster. Public HotpotQA adapted, not official leaderboard."}
    (root/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    lines=["# 扩展能力测评结果","","所有验证在 conda deepr；配置为非思考模式，对抗最多 3 轮、Blue 每轮三类合计最多 5 个问题。",
        f"\n研究尝试 {len(rows)} 次，完整组 {len(full)} 份。产物核验 {len(checks)} 项，失败 {len(result['artifact_check_failures'])} 项。研究/评审操作异常 {len(failures)} 份。",
        "\n## 同题模块对照","","完整组仅取与消融相同的 12 道分层题；不能用全部完整组均值与 12 题对照均值直接相减。",
        "\n|配置|次数|研究成功率|合法 Judge|质量均分|验收率|平均秒数|研究 Token|","\n|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,s in raw_summary.get("variants",{}).items():
        score=f"{s['average_judge_score']:.3f}" if s["average_judge_score"] is not None else "不可用"
        lines.append(f"|{name}|{s['num_attempted']}|{s['completion_rate']:.1%}|{s['judge_successes']}|{score}|{s['report_acceptance_rate'] or 0:.1%}|{s['average_elapsed_seconds']:.2f}|{s['total_provider_tokens']}|")
    lines+=["\n## 按能力分层（完整组）","","|方向|样本数|质量均分|验收率|","","|---|---:|---:|---:|"]
    for category,v in by_category.items():
        s=v["full"];score=s["average_judge_score"]
        lines.append(f"|{category}|{s['num_attempted']}|{score if score is not None else '不可用'}|{s['report_acceptance_rate'] or 0:.1%}|")
    lines+=["\n## 配对与检索诊断","",f"对抗前后相同正文配对数 {len(diffs)}，质量变化统计：{json.dumps(pre_stats,ensure_ascii=False)}。",
        "\n检索 Recall 是轨迹中所有搜索结果 URL 的累计覆盖，不是单次 Recall@5；阅读覆盖另列。QA 短答案从正文独立提取，不参考 gold，缺少答案行按未答计。题集和输出协议改变，不能与官方榜单 EM/F1 比较。",
        "\nQA："+json.dumps(result["qa_summary"],ensure_ascii=False),
        f"\n已记录研究 Token {result['research_tokens']}，独立 Judge {result['judge_tokens']}，对抗前 Judge {result['pre_judge_tokens']}；裁判回归与 pilot 另计。",
        "\n## 适用范围","","受控集验证机制与边界，不是跨领域真实网页研究基准。公开 HotpotQA 为历史快照小子集，模型可能训练时见过；构造样例未经独立人工盲审。没有 M6 权重训练，也不由新会话平均分推断长期记忆收益。端到端耗时受四样本并发、网络和生成长度影响，不能仅凭均时判断 Worker 并发的因果收益。",
        "\n原始结果、证据、报告、前后版本及 Trace 保留；失败样本保持在分母，详见 analysis.json。"]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:result[k] for k in ("num_research_attempts","artifact_check_failures","research_tokens","judge_tokens","qa_summary")},ensure_ascii=False),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--review-before",action="store_true")
    args=p.parse_args()
    if args.review_before:review_before(args.root,collect(args.root),LLMJudge(thinking_mode="disabled"))
    analyze(args.root)
