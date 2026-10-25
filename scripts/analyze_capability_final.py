"""Build final descriptive report from immutable research and uniform v4 reviews."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import hashlib
import json
import statistics
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

from evaluation.qa_metrics import extract_answer,answer_metrics
from scripts.doc_paths import write_report
from src.core.report_evaluation import report_contract


def read(path):return json.loads(path.read_text())
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
def mean(values):return statistics.mean(values) if values else None
def f(value,digits=3):return f"{value:.{digits}f}" if value is not None else "不可用"
def pct(value):return f"{value:.1%}" if value is not None else "不可用"

def bootstrap(differences):
    if not differences:return {"n":0,"mean":None,"ci95":None}
    if len(differences)==1:return {"n":1,"mean":differences[0],"ci95":None,"scope":"one pair; no interval estimate"}
    rng=np.random.default_rng(20261018)
    values=np.asarray(differences)
    samples=rng.choice(values,size=(10000,len(values)),replace=True).mean(axis=1)
    return {"n":len(values),"mean":float(values.mean()),"ci95":[float(x) for x in np.quantile(samples,[.025,.975])],
        "scope":"question-level paired descriptive bootstrap; single runs, no multiple-comparison correction"}

def collect(base):
    rows=[]
    for stage in ("controlled","ablations","hotpot","positive_controls","positive_memory_control"):
        directory=base/stage
        if not (directory/"results.jsonl").exists():continue
        dataset=read(directory/"dataset.json");qs={q["id"]:q for q in dataset["examples"]}
        for line in (directory/"results.jsonl").read_text().splitlines():
            r=json.loads(line);r.update({"stage":stage,"directory":directory/Path(r["report_path"]).parent,"question":qs[r["question_id"]]})
            rows.append(r)
    return rows

def run(base,initial_only=False):
    base=base.resolve()
    rows=collect(base)
    if initial_only:
        reviews=[]
        for r in rows:
            body=(r["directory"]/"report_body.md").read_text();evaluation=r.get("judge",{"error":"Judge not requested"})
            reviews.append({"sample_id":r["sample_id"],"version":"final","evaluation":evaluation,
                "valid":r.get("judge_status")=="success","initial_valid":r.get("judge_status")=="success",
                "contract":report_contract(body,r["question"].get("requirements")),
                "report_sha256":hashlib.sha256(body.encode()).hexdigest()})
        review_summary={"scope":"v3 initial scores only; deterministic contract v2 recheck, no new semantic evaluation",
            "num_expected":len(rows),"num_completed":len(rows),"initial_valid":sum(r["valid"] for r in reviews),
            "valid_after_bounded_retry":sum(r["valid"] for r in reviews),"retry_cases":0,"recorded_tokens":0}
        derived=base/'initial_report_analysis';derived.mkdir(exist_ok=True)
        write(derived/'results.json',reviews);write(derived/'summary.json',review_summary)
    else:
        reviews=read(base/"uniform_report_v4/results.json")
        if read(base/"uniform_report_v4/manifest.json")["status"]!="completed":raise ValueError("Uniform review incomplete")
        review_summary=read(base/"uniform_report_v4/summary.json")
    index={(r["sample_id"],r["version"]):r for r in reviews}
    for r in rows:r["review"]=index[(r["sample_id"],"final")]
    def stats(group):
        valid=[r for r in group if r["status"]=="success" and r["review"]["valid"]]
        facts=Counter(fact["verdict"] for r in valid for fact in r["review"]["evaluation"].get("facts",[]))
        return {"n":len(group),"complete":sum(r["status"]=="success" for r in group),"quality_n":len(valid),
            "quality_mean":mean([r["review"]["evaluation"]["average"] for r in valid]),
            "quality_pass":sum(r["review"]["evaluation"].get("quality_pass",False) for r in valid),
            "initial_review_valid":sum(r["review"]["initial_valid"] for r in group),
            "effective_review_valid":sum(r["review"]["valid"] for r in group),
            "contract_pass":sum(r["review"]["contract"]["constraint_pass"] for r in group),
            "accepted":sum(r["status"]=="success" and r["review"]["valid"] and r["review"]["evaluation"].get("report_pass",False) for r in group),
            "acceptance_evaluable":sum(r["review"]["valid"] or not r["review"]["contract"]["constraint_pass"] for r in group),
            "mean_seconds":mean([r["elapsed_seconds"] for r in group]),
            "research_tokens":sum(r.get("usage",{}).get("total_tokens",0) for r in group),"facts":dict(facts),
            "contract_failures":dict(Counter(c["name"] for r in group for c in r["review"]["contract"]["checks"] if not c["passed"]))}
    primary=[r for r in rows if r["stage"] in ("controlled","ablations","hotpot")]
    full=[r for r in primary if r["variant"]=="full"]
    anchors={r["question_id"]:r for r in rows if r["stage"]=="controlled" and r.get("ablation")}
    groups={"controlled":stats([r for r in rows if r["stage"]=="controlled"]),
        "hotpot":stats([r for r in rows if r["stage"]=="hotpot"]),
        "positive_controls":stats([r for r in rows if r["stage"]=="positive_controls"]),
        "positive_memory_control":stats([r for r in rows if r["stage"]=="positive_memory_control"])}
    category={c:stats([r for r in full if r.get("category")==c]) for c in sorted({r.get("category") for r in full})}
    ablations={"full":stats(list(anchors.values()))};paired={}
    for variant in sorted({r["variant"] for r in rows if r["stage"]=="ablations"}):
        control=[r for r in rows if r["stage"]=="ablations" and r["variant"]==variant]
        ablations[variant]=stats(control);differences=[];pairs=[]
        for b in control:
            a=anchors[b["question_id"]]
            if a["status"]==b["status"]=="success" and a["review"]["valid"] and b["review"]["valid"]:
                delta=a["review"]["evaluation"]["average"]-b["review"]["evaluation"]["average"]
                differences.append(delta);pairs.append({"question_id":a["question_id"],"full_minus_control":delta})
        paired[variant]={**bootstrap(differences),"pairs":pairs,
            "research_token_ratio_full_to_control":ablations["full"]["research_tokens"]/ablations[variant]["research_tokens"] if ablations[variant]["research_tokens"] and len(control)==len(anchors) and all(r['status']=='success' for r in control) else None}
    pre=[]
    for a in anchors.values():
        before=index.get((a["sample_id"],"before_m5"));after=a["review"]
        if before and before["valid"] and after["valid"] and a["status"]=="success":
            pre.append({"question_id":a["question_id"],"before":before["evaluation"]["average"],
                "after":after["evaluation"]["average"],"difference":after["evaluation"]["average"]-before["evaluation"]["average"]})
    pre_summary={**bootstrap([r["difference"] for r in pre]),"pairs":pre,
        "improved":sum(r["difference"]>0 for r in pre),"unchanged":sum(r["difference"]==0 for r in pre),"degraded":sum(r["difference"]<0 for r in pre)}
    qa=[]
    for r in rows:
        if r["stage"]!="hotpot":continue
        prediction=extract_answer((r["directory"]/"report_body.md").read_text());gold=r["question"]["short_answer"]
        qa.append({"question_id":r["question_id"],"prediction":prediction,"gold":gold,**answer_metrics(prediction,gold)})
    qa_summary={"n":len(qa),"answer_line_present":sum(r["prediction"] is not None for r in qa),
        "exact_match":mean([r["exact_match"] for r in qa]),"f1":mean([r["f1"] for r in qa])}
    modules=defaultdict(Counter);retrieval=[];budgets=[];compressions=[];unknown_sources=[];rounds=Counter()
    for r in rows:
        t=read(r["directory"]/"trace.json");byid={s["span_id"]:s for s in t.get("spans",[])}
        rounds[str(r.get("adversarial_rounds",0))]+=1
        for e in t.get("events",[]):
            if e["name"]=="llm_usage":
                s=byid.get(e.get("span_id"));role="other";seen=set()
                while s and s["span_id"] not in seen:
                    seen.add(s["span_id"]);name=s["name"]
                    if name.startswith(("red_agent.","blue_agent.","planner.","summarizer.","compressor.")):
                        role=name.split('.')[0];break
                    if name=="researcher.run":role="worker";break
                    s=byid.get(s.get("parent_span_id"))
                modules[r["variant"]+"/"+role]["tokens"]+=e["data"].get("total_tokens",0)
                modules[r["variant"]+"/"+role]["calls"]+=1
            elif e["name"]=="blue_repair_budget":budgets.append({"sample_id":r["sample_id"],**e["data"]})
            elif e["name"]=="compression":compressions.append({"sample_id":r["sample_id"],"variant":r["variant"],**e["data"]})
        q=r["question"];private={d["url"] for d in q.get("memory_seed",[])}
        gold=set(q.get("reference_doc_urls",[d["url"] for d in q.get("evidence",[])]))-private
        searched={url for e in t.get("events",[]) if e["name"]=="snapshot_search" for url in e["data"].get("urls",[])}
        opened={e["data"]["url"] for e in t.get("events",[]) if e["name"]=="snapshot_read"}
        retrieval.append({"sample_id":r["sample_id"],"stage":r["stage"],"variant":r["variant"],"category":r.get("category"),
            "gold_cumulative_search_recall":len(gold&searched)/len(gold) if gold else None,
            "gold_full_read_recall":len(gold&opened)/len(gold) if gold else None})
        snapshot=read(r["directory"].parents[1]/"dataset.json")
        known={d["url"] for d in snapshot.get("documents",[])}|private
        outside=set(r["review"]["contract"].get("citation_urls",[]))-known
        if outside:unknown_sources.append({"sample_id":r["sample_id"],"urls":sorted(outside)})
    regression=read(base/"judge_regression/summary.json") if (base/"judge_regression/summary.json").exists() else {
        "num_completed":0,"num_expected":48,"judge_valid":0,"expectation_checks":0,"expectation_checks_passed":0,"total_tokens":0}
    initial_judges=Counter(r.get("judge_status") for r in primary)
    resource={"primary_research_tokens":sum(r.get("usage",{}).get("total_tokens",0) for r in primary),
        "primary_initial_judge_tokens":sum(r.get("judge_usage",{}).get("total_tokens",0) for r in primary),
        "additional_research_tokens":sum(r.get("usage",{}).get("total_tokens",0) for r in rows if r not in primary),
        "additional_initial_judge_tokens":sum(r.get("judge_usage",{}).get("total_tokens",0) for r in rows if r not in primary),
        "uniform_v4_judge_tokens":review_summary["recorded_tokens"],
        "regression_judge_tokens":regression["total_tokens"],
        "legacy_pre_review_tokens":sum(r.get("usage",{}).get("total_tokens",0) for r in read(base/"pre_adversarial_reviews.json")) if (base/"pre_adversarial_reviews.json").exists() else 0}
    resource["selected_artifact_recorded_total"]=sum(resource.values())
    selected_ids={r["sample_id"] for r in rows};abandoned_by_id={};ancestor=base;visited=set()
    while ancestor not in visited:
        visited.add(ancestor);manifest=read(ancestor/'suite_manifest.json')
        prior=manifest.get('resume_from')
        if not prior:break
        ancestor=Path(prior).resolve()
        for p in ancestor.glob('*/results.jsonl'):
            for line in p.read_text().splitlines():
                row=json.loads(line)
                if row['sample_id'] not in selected_ids:abandoned_by_id.setdefault(row['sample_id'],row)
    abandoned=list(abandoned_by_id.values())
    resource["abandoned_ancestor_attempt_tokens"]=sum(r.get("usage",{}).get("total_tokens",0)+r.get("judge_usage",{}).get("total_tokens",0) for r in abandoned)
    pilot=ROOT/"outputs/evaluation/capability_v3_pilot_20261018/controlled/results.jsonl"
    resource["pilot_recorded_tokens"]=sum(r.get("usage",{}).get("total_tokens",0)+r.get("judge_usage",{}).get("total_tokens",0) for r in map(json.loads,pilot.read_text().splitlines())) if pilot.exists() else 0
    resource["this_expansion_recorded_total"]=resource["selected_artifact_recorded_total"]+resource["abandoned_ancestor_attempt_tokens"]+resource["pilot_recorded_tokens"]
    research_total=sum(r.get("usage",{}).get("total_tokens",0) for r in rows)
    if sum(x["tokens"] for x in modules.values())!=research_total:raise ValueError("Exclusive module usage does not reconcile")
    m5=sum(x["tokens"] for name,x in modules.items() if name.split('/')[-1] in ('red_agent','blue_agent'))
    completed=[r for r in full if r["status"]=="success"]
    summary={"primary_attempts":len(primary),"additional_attempts":len(rows)-len(primary),
        "reused_research_samples":sum(bool(r.get('reused_from')) for r in rows),
        "mechanism_examples":[{'question_id':r['question_id'],
            'before':str((r['directory']/'report_before_adversarial.md').relative_to(ROOT)),
            'after':str((r['directory']/'report_body.md').relative_to(ROOT)),
            'trace':str((r['directory']/'trace.json').relative_to(ROOT))}
            for r in rows if r['stage']=='controlled' and r['question_id']=='memoryreusecasea'],
        "groups":groups,"by_category":category,"ablations":ablations,"paired_full_minus_control":paired,
        "pre_post_m5":pre_summary,"qa_summary":qa_summary,"qa":qa,"modules":dict(modules),"round_counts":dict(rounds),
        "blue_budget_events":len(budgets),"blue_selected":sum(b["selected"] for b in budgets),"blue_deferred":sum(b["deferred"] for b in budgets),
        "blue_max_selected":max([b["selected"] for b in budgets],default=0),"resource":resource,
        "initial_generation_judge_statuses":dict(initial_judges),"uniform_review":review_summary,
        "evaluation_scope":"v3_initial_only" if initial_only else "uniform_v4",
        "regression":regression,"retrieval":retrieval,"compressions":compressions,"out_of_snapshot_citations":unknown_sources,
        "primary_source_hashes":dict(Counter(r["source_hash"] for r in primary)),
        "m5_share_of_research_tokens":m5/research_total if research_total else None,
        "module_usage_reconciled":True,"abandoned_ancestor_attempts":len(abandoned),
        "max_rounds_observed":max([r.get("adversarial_rounds",0) for r in rows],default=0),
        "report_char_median":statistics.median(r["review"]["contract"]["char_count"] for r in completed),
        "generation_body_sha256_verified":all(hashlib.sha256((r["directory"]/"report_body.md").read_bytes()).hexdigest()==r["review"]["report_sha256"] for r in rows)}
    write(base/"final_analysis.json",summary)
    if initial_only:render_partial(base,summary)
    else:render(base,summary)
    print(json.dumps({k:summary[k] for k in ('primary_attempts','additional_attempts','groups','qa_summary','paired_full_minus_control','pre_post_m5','resource')},ensure_ascii=False,indent=2))
    return summary

def render(base,s):
    rel=str(base.relative_to(ROOT));g=s["groups"]
    lines=["# 扩展测评报告与分析（2026-10-18）","",
        "状态：本次扩展研究与诊断已执行，下面分别列研究完成、评审合法、报告质量和任务验收；操作失败保留在分母，未得到合法评分时不填质量零分。",
        "","## 测评范围与结果口径","",
        "受控集 48 题（12 方向×4）＋真实 HotpotQA hard 改编 24 题，共 72 道主题。12 道预选锚点分别对照关闭 M5、关闭 M3、关闭 M4、关闭规划前侦察、串行 Worker，主实验 132 次研究。另有 12 道正向反事实完整组和 3 次额度记忆对照，共 15 次补充研究，分开汇总；48 份正确/反转报告用于裁判开发诊断。",
        "",
        "所有验证在 conda deepr。研究采用同一 DeepSeek 后端别名、非思考模式、温度 .1、每次输出上限 8192；样本并发 4、Worker 并发 3（串行对照 1）、每研究 deadline 900 秒。对抗上限 3 轮，Blue 每轮原位修改/补搜/删除合计最多 5 项；补搜另有最多 3 次限制。M6 不进行权重训练。",
        "",
        "生成与原 v3 首评全部保留；结论采用独立 uniform_report_v4 统一复评。v4 修复隐藏金标准 URL 限制，向裁判提供本题两份参考材料和一份错误草稿；公开题使用原十段证据。没有重做正文、没有按结果抽题。Judge 无效输出最多单独重试一次，初次与重试分别留档。五维 report-rubric-v1 保持不变，引用契约使用 report-contract-v2。",
        "",
        "|组别|研究完成/尝试|完整研究合法评分|质量均分/10|契约通过|报告验收|",
        "|---|---:|---:|---:|---:|---:|"]
    for label,key in (("受控完整组","controlled"),("公开改编完整组","hotpot"),("正向反事实补充","positive_controls"),("补充关闭记忆","positive_memory_control")):
        x=g[key];lines.append(f"|{label}|{x['complete']}/{x['n']}|{x['quality_n']}|{f(x['quality_mean'])}|{x['contract_pass']}/{x['n']}|{x['accepted']}/{x['n']}|")
    lines += ["","质量均分仅取完整研究且取得合法评分的报告；契约不通过即可判定不验收，评审无效且契约通过时验收状态未知，不按内容零分。以下消融只用相同 12 道锚点，不能用全部 48 题均分减对照均分。","",
        "## 同题模块对照","","|配置|研究完成/12|合法完整评分|质量均分/10|验收/12|平均研究秒数|研究 Token|","|---|---:|---:|---:|---:|---:|---:|"]
    for name,x in s["ablations"].items():lines.append(f"|{name}|{x['complete']}/{x['n']}|{x['quality_n']}|{f(x['quality_mean'])}|{x['accepted']}/{x['n']}|{f(x['mean_seconds'],1)}|{x['research_tokens']}|")
    lines += ["","|对照|有效配对数|完整组减对照质量分|95%配对Bootstrap区间|完整组/对照Token比|","|---|---:|---:|---|---:|"]
    for name,x in s["paired_full_minus_control"].items():
        ci=x['ci95'];lines.append(f"|{name}|{x['n']}|{f(x['mean'])}|{f(ci[0])+' 至 '+f(ci[1]) if ci else '不可用'}|{f(x['research_token_ratio_full_to_control'],2)}|")
    lines += ["","差值为完整组减关闭模块组，正值表示本批题的完整组质量较高。仅对两边研究完整且评分合法的相同问题计算；单题单次运行，样本和模型随机性未分离。区间是题级重采样的描述统计，未做多重比较校正，不把其包装为模块普遍有效的证明。端到端耗时还受四样本共享请求负载影响。","",
        "## 分方向结果","","|能力方向|完整/题数|有效评分数|质量均分/10|契约通过|","|---|---:|---:|---:|---:|"]
    for name,x in s['by_category'].items():lines.append(f"|{name}|{x['complete']}/{x['n']}|{x['quality_n']}|{f(x['quality_mean'])}|{x['contract_pass']}/{x['n']}|")
    pre=s['pre_post_m5'];qa=s['qa_summary'];reg=s['regression'];rev=s['uniform_review']
    lines += ["","## 对抗前后与公开题诊断","",
        f"12 道预选锚点保存了同一次研究的对抗前后正文。有效配对 {pre['n']}，平均后减前 {f(pre['mean'])}/10，改善 {pre['improved']}、持平 {pre['unchanged']}、退化 {pre['degraded']}；95%题级Bootstrap区间 {pre['ci95']}。它描述相同报告被修复后的变化，另列 no_adversarial 研究对照，不混作一项指标。",
        "","|题目|对抗前|对抗后|变化|","|---|---:|---:|---:|"]
    for x in pre['pairs']:lines.append(f"|{x['question_id']}|{f(x['before'])}|{f(x['after'])}|{f(x['difference'])}|")
    lines += ["",f"公开改编题 {qa['n']} 道，独立从正文提取到答案行 {qa['answer_line_present']} 道；EM={pct(qa['exact_match'])}，Token F1={pct(qa['f1'])}。提取不读取标准答案，缺答案行计未答。共享 239 文档检索、中文报告与答案行协议均不同于官方 HotpotQA，且仅为 public dev 历史小子集，不能与官方排行榜直接比较。",
        "","## 裁判可靠性与来源诊断","",
        f"48 份构造裁判诊断实际完成 {reg['num_completed']}/{reg['num_expected']}，合法 {reg['judge_valid']}，预期检查通过 {reg['expectation_checks_passed']}/{reg['expectation_checks']}。它检验当前锚点与输出协议，不是独立人工标注保留集准确率。失败预期逐项保留在 judge_regression/summary.json。",
        "",f"v4 复评含最终与对抗前正文共 {rev['num_completed']}/{rev['num_expected']} 份；初次合法 {rev['initial_valid']}，有界重试后合法 {rev['valid_after_bounded_retry']}，重试案例 {rev['retry_cases']}。初次无效不能算作首评成功。全文分段审查保留原文位置，跨段事实合并采用错误优先规则，但仍需人工盲审校准。",
        "",f"固定快照外 URL 出现在 {len(s['out_of_snapshot_citations'])} 份报告中（含补充组）；这是来源可追溯性告警，不能单凭 URL 直接判事实错误。引用并排除本题错误草稿属于合理分析。完整来源列表与匹配记录见 final_analysis.json。",
        "","## 模块作用、已观察到的缺点与改进方向","",
        "M1 负责 DAG 并发、依赖失败传播、deadline 与取消；M2 负责拆分和规划前侦察。当前检索按 BM25 真实匹配查询；累计搜索支持文档覆盖与全文阅读覆盖分别统计，前者不是单次 Recall@5。并发上限与开关通过 Trace 检查，串行对照仍受请求调度和报告长度影响。下一步可重复同题、固定子任务，并增加图调度单独基准。",
        "",
        "M3 在规划记忆、收集中间结果、旧记忆整理使用不同压缩粒度。本次关闭开关比较的是已接入编排器的压缩路径，不能拆成 L1/L2/L3 各自收益。长文末尾有效规则必须先读到、再保留并传给合成及审查；后续应建立带 document_id/span 的证据对象和规则保留探针，按信息损失与总成本一同评价。",
        "",
        "M4 将先前私有额度写入当前会话的临时 Store，搜索库不含该附件；关闭记忆切断历史输入通道。正向反事实把额度改为 6000 元，检验系统能否在条件满足时批准，避免一律拒绝的捷径。这是受控通道诊断，尚不代表真实跨多次研究的长期记忆效果。",
        "",
        "M5 已实跑 Red 审查与 Blue 三类修复，预算按轨迹逐轮核验。重要缺点是 Worker 读取全文或 Planner 检索到私有记忆后，report.sources 仍主要传搜索摘要；Red/Blue 来源展示只取前 300 字符，审查报告与自检也有截取。memoryreusecasea 对抗前正确使用 1300 元历史额度、算出 2100 元超额，对抗后却以缺来源为由删除额度和确定结论；memory_query 事件证实曾返回该记忆。pilot 长文也出现已读到末尾规则却被审查误删的案例。这说明证据传递断层会抵消上游模块作用；不能把全部长文或记忆失败归因于检索。P0 应把实际读到的全文片段和私有来源传入审查，区分“当前未展示”与“原文不支持”，分数绑定完整修订版本。",
        "",
        "报告交付的主要问题是固定长文模板与用户字符上限冲突。质量分和契约验收已分开，但测评只能暴露问题。P0 应把长度、答案行、引用与结构要求作为类型化生成上下文，输出后先确定性检查，再做一次有界修订；直接截断会损坏结论和引用。",
        "",
        "裁判仍有 Schema、JSON 或引文定位失败。P1 可使用后端支持的 JSON 输出模式和一次带校验反馈的有限修复，保留初评合法率与重试成本；再请独立人工对事实错误、证据不足及关键结论做盲审。M5 的轮数与问题数限制并非总 Token 硬上限，P1 可增加共享证据缓存、去重、差分补丁及按预期收益早停，随后对质量和成本一起复测。",
        "",
        "样本已扩充但仍有边界：同方向受控题共享模板、只测一个模型别名且单次运行、公开题可能存在训练污染、未测试实时网页时效、未验证真实长会话或 M6 训练。P2 应加入不同领域真实研究任务、人工证据标注、跨模型重复运行、检索排序与时间鲁棒性测试。不能把本次均分解释成全领域研究能力。",
        "","## 预算、成本与复现","",
        f"实际观察最多 {s['max_rounds_observed']} 轮，{s['blue_budget_events']} 个 Blue 预算事件，最大单轮选择 {s['blue_max_selected']} 项，累计选择 {s['blue_selected']}、延后 {s['blue_deferred']}。延后项不算已修复，操作成功标记也不等于事实一定修正。",
        "","|已选结果的用量类别|服务端记录Token|","|---|---:|"]
    for name,x in s['resource'].items():lines.append(f"|{name}|{x}|")
    lines += ["","selected_artifact_recorded_total 包括复用的 40 次研究一次，不重复累加；this_expansion_recorded_total 再加首次中断中未入选的失败/部分尝试和 pilot，不重复相加各级小计。旧三题历史实验不计入本次扩展。尚未返回 Usage 的取消在途请求无法从本地确认，不承诺等于最终账单。按角色独占归属的 Token 与研究调用总数在 final_analysis.json，不把字符估算压缩率当作账单节约。",
        "",f"本次所选研究轨迹中 Red/Blue 占研究 Token 的 {pct(s['m5_share_of_research_tokens'])}；所有角色独占归属之和与研究 Usage 一致。这解释了只限制轮数和问题数仍可能成本较高，后续应优先减少重复发送全文。",
        "",f"生成源码有 {len(s['primary_source_hashes'])} 个指纹：复用部分保留首次运行身份，续跑新增账户错误熔断和同配置检查点复用；研究模型、输入与生成预算保持一致。所有评审正文 SHA256 校验：{s['generation_body_sha256_verified']}。原始 v3、统一 v4、重试和历史结果分开留档。",
        "",f"实际产物：[完整分析]({rel}/final_analysis.json)、[统一复评]({rel}/uniform_report_v4/summary.json)、[裁判诊断]({rel}/judge_regression/summary.json)、[原始轨迹检查]({rel}/analysis.json)。方法和许可见 [数据卡](EVALUATION_DATASET_V3.md)。",
        "","工程回归：127 个离线用例通过；均在 deepr 中使用临时库或模拟传输，真实模型实验另计。所有正式报告、对抗前正文、Trace、Judge 输入、证据哈希与元信息均归档，正式 data/memory.db 未作为实验库。"]
    write_report(base,'\n'.join(lines)+'\n')

def render_partial(base,s):
    rel=str(base.relative_to(ROOT));groups=s['groups'];a=s['ablations']
    lines=['# 扩展测评报告与分析（2026-10-18）','',
        '**状态：全量测评尚未完成。充值后已用同一 DeepSeek 续跑，但再次遇到 HTTP 402（余额不足），实验已自动停止。**','',
        '按当前用户选择，先整理已有结果，暂不继续真实模型调用。','',
        '以下采用原 v3 首评和 report-contract-v2 的确定性重算，不冒充尚未执行的 v4 统一复评；原始正文与评分保留。账户错误、取消和评审无效不当作模型质量零分。','',
        '## 已完成范围','',
        '已构建 48 道受控题（12 方向×4）、24 道真实 HotpotQA hard 改编题、12 道分层消融锚点、48 份正确/错误成对裁判诊断；另有 12 道正向反事实补充题。主实验计划 132 次研究，补充计划 15 次，两者分开。','',
        f"本次目录有 {s['primary_attempts']} 条主实验记录，完整研究 {sum(x['complete'] for x in a.values())+groups['controlled']['complete']-a['full']['complete']} 次。复用首次 40 次成功研究，不重复调用模型。全部 48 道受控题已完成；关闭对抗、关闭压缩、关闭记忆各 12 道完整；关闭侦察仅完成 1 道，其余发生资源失败或取消；串行 Worker、24 道公开题、48 份裁判诊断、12 份对抗前评审及 15 次补充研究未执行。仍需补齐 47 个主实验研究槽位和上述诊断，不能宣称完成全量测评。",'',
        '## 当前可用结果','',
        '|方向|研究完成/题数|合法完整首评份数|质量均分/10|契约通过|','|---|---:|---:|---:|---:|']
    for name,x in s['by_category'].items():lines.append(f"|{name}|{x['complete']}/{x['n']}|{x['quality_n']}|{f(x['quality_mean'])}|{x['contract_pass']}/{x['n']}|")
    c=groups['controlled'];lines += ['',f"受控完整组 48 次研究均完成，其中 {c['quality_n']} 份有合法首评，均分 {f(c['quality_mean'])}/10；质量均分只取这 {c['quality_n']} 份。参考事实判定合计 {c['facts']}。契约通过 {c['contract_pass']}/48，报告验收 {c['accepted']}/48；违约类型 {c['contract_failures']}。高质量分不能替代交付验收。",'',
        '## 已完成的同题模块对照','',
        '下面只用相同的 12 道预选锚点；质量均分仅取完整且首评合法结果。各组合法数量可能不同，因此模块差值另外按共同有效题计算。','',
        '|配置|研究完成/尝试|合法完整首评|质量均分/10|契约通过|平均研究秒数|研究Token|','|---|---:|---:|---:|---:|---:|---:|']
    for name,x in a.items():lines.append(f"|{name}|{x['complete']}/{x['n']}|{x['quality_n']}|{f(x['quality_mean'])}|{x['contract_pass']}/{x['n']}|{f(x['mean_seconds'],1)}|{x['research_tokens']}|")
    lines += ['', '|已完成对照|共同有效配对|完整组减关闭组质量分|描述性95%Bootstrap区间|完整组/关闭组研究Token比|','|---|---:|---:|---|---:|']
    for name,x in s['paired_full_minus_control'].items():
        if a[name]['complete']!=12:continue
        ci=x['ci95'];lines.append(f"|{name}|{x['n']}|{f(x['mean'])}|{f(ci[0])} 至 {f(ci[1])}|{f(x['research_token_ratio_full_to_control'],2)}|")
    lines += ['',
        '差值为完整组减关闭组，正值表示本批题完整组质量较高。各题只运行一次，合成兄弟题共享模板；题级 Bootstrap 仅描述这些题的波动，未做多重比较校正，也未分离模型随机性，不能宣布普遍因果收益。关闭侦察和串行对照未齐，不给总体结论。当前同题质量均采用 v3、相同参考证据和同一评审参数；v4 尚未实跑，不混入新口径分数。','',
        '## 模块作用与已观察到的问题','',
        'M1 管理 DAG 并发、依赖失败传播、deadline 和取消。当前运行的跨度关系、Usage、对抗轮数和 Blue 全局修复预算核验通过；串行对照尚未执行，因此不能量化 Worker 并发收益。M2 执行计划拆分与规划前侦察，当前 BM25 按查询真实排序，搜索摘要与全文阅读分别记录；no_recon 不完整，尚不能据此评价侦察作用。','',
        'M3 控制规划记忆和收集结果的上下文规模，当前 no_compressor 切断已接入编排器的压缩路径。上表展示同题质量与总研究成本，不等同于 L1/L2/L3 分别产生的收益；压缩事件中的 token 是字符估算，不能冒充服务端账单节省。下一版可增加关键数字、时间边界、否定与原文 span 的保留率测试。','',
        'M4 把先前私有额度写入会话 Store，搜索库不包含私有附件；no_memory 不创建 Store，但仍有任务依赖上下文。此对照反映当前完整系统的端到端表现，不能把上游成功检索等同于最终正确使用。额度正向反事实与更长真实会话尚未实测，不据此宣布长期记忆收益。','',
        'M5 的真实代价与效果可以从 no_adversarial 的同题对照观察，但对抗前后 12 题复评尚未执行，不能虚构其量化改善。轨迹已定位到证据传递断层：memoryreusecasea 中 memory_query 返回了私有记忆，初稿正确使用 1300 元额度与 2100 元含运费报价，判断超额不能直接批准；对抗后以“缺乏来源”为由删除额度和确定结论。pilot 的长文题也出现读到全文末尾规则后，正确结论被误删的情况。','',
        '代码原因是 Summarizer 的 report.sources 主要收集搜索摘要；Red/Blue 只展示每来源前 300 字符，审查报告和 Blue 自检另有 4000/2000 字符截取。全文被 Worker 读到或记忆被 Planner 取回，并不保证审查拿到这些证据。P0 应建立带 document_id/span 的统一证据对象，把私有来源与实际阅读片段传到审查，区分未显示和原文不支持，分数绑定完整正文版本。','',
        '交付层面的主要问题是固定长报告模板与字符上限冲突。P0 应把长度、引用、答案行和章节要求传入生成上下文，输出后确定性检查，再做一次有界修订；不能用简单截断破坏结论和来源。P1 是裁判 JSON 输出约束、有校验反馈的有限重试和独立人工盲审，继续区分首评合法率与补评可用率。P1 成本优化应优先采用证据缓存、问题去重、差分补丁与收益早停。','',
        'M6 是权重训练与演化，本次 API 研究没有更新权重，不构造空转消融。P2 应补充真实时效网页、跨模型和重复运行、独立标注研究题与长会话测评；当前一个模型别名、构造模板与 public dev 小子集不足以代表全领域能力。','',
        '## 评测口径的修正与边界','',
        'report-contract-v2 支持参考来源章节中的普通编号文献，并防止把正文列表误当引用定义。对原始结果确定性重算，不改语义评分。v3 隐藏了仅两份金标准 URL 的白名单，导致合理引用并排除草稿也被算违规；因此这些标记不能直接叫事实引用错误。已建立 v4：查询、真值与语料不变，删除该隐藏限制，让裁判同时看到本题正确材料和草稿。v4 需要恢复余额后统一复评，不能免费推算新分数。','',
        f"来源可追溯性另作诊断：当前 {len(s['out_of_snapshot_citations'])} 份报告含快照外 URL，逐条保存于 final_analysis.json；该告警本身不是事实错误判定。公开题 EM/F1、裁判诊断通过率、正向条件表现和 M5 前后变化均暂无本轮实测结果，不用历史实验补位。",'',
        '## 资源与检查点','',
        f"实际最多 {s['max_rounds_observed']} 轮，Blue 最大单轮选择 {s['blue_max_selected']} 项，累计选择 {s['blue_selected']}、延后 {s['blue_deferred']}；三种修复合计共用预算。Red/Blue 占所选研究 Token 的 {pct(s['m5_share_of_research_tokens'])}。三轮五项限制控制操作规模，不是总 Token 硬上限。",'',
        '|用量类别|服务端记录Token|','|---|---:|']
    for name,value in s['resource'].items():lines.append(f'|{name}|{value}|')
    lines += ['',
        'selected_artifact_recorded_total 中复用研究只计一次；this_expansion_recorded_total 另加首次中断中未入选尝试和 pilot，各级小计不能再重复相加。中止在途请求未返回 Usage 的部分无法从本地确认，不承诺等于账单。','',
        f"本次新熔断遇到 HTTP402 后已保存当前尝试，停止排队并清理在途任务，后续模块与补充评审未启动。127 个离线用例通过；所有验证在 conda deepr，正式记忆库未作为实验库。生成源码 {len(s['primary_source_hashes'])} 个指纹，复用的 40 次保留原身份；相同题集和配置检查点可继续复用成功研究。",'',
        f"产物：[当前完整分析]({rel}/final_analysis.json)、[原始运行与轨迹检查]({rel}/analysis.json)、[产物审计]({rel}/artifact_audit_final.json)、[工程测试日志]({rel}/unit_tests_final.log)、[数据卡与许可](EVALUATION_DATASET_V3.md)。原始 v3 与将来的统一 v4 分开保存，不覆盖原报告。"]
    for example in s['mechanism_examples']:
        lines += ['',f"记忆证据传递案例：[对抗前正文]({example['before']})、[最终正文]({example['after']})、[执行轨迹]({example['trace']})。"]
    write_report(base,'\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--initial-only',action='store_true',help='Analyze saved initial scores without any new model calls')
    args=p.parse_args();run(args.base,args.initial_only)
