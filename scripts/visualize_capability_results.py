"""Static publication artifacts plus a local, filterable case browser."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.doc_paths import REPORT, rebase_links


def run(base):
    base=base.resolve();data=json.loads((base/'final_analysis.json').read_text())
    initial=data.get('evaluation_scope')=='v3_initial_only'
    scope='v3 initial review' if initial else 'uniform v4 review'
    folder=base/'figures';folder.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ('png','svg'):fig.savefig(folder/(name+'.'+ext),dpi=170)
        plt.close(fig)
    categories=data['by_category'];names=list(categories)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    scores=[categories[n]['quality_mean'] if categories[n]['quality_mean'] is not None else np.nan for n in names]
    ax.barh(names,scores,color='#437eac');ax.invert_yaxis();ax.set_xlim(0,10)
    for i,n in enumerate(names):
        label=f"{scores[i]:.2f} n={categories[n]['quality_n']}" if np.isfinite(scores[i]) else 'no valid review'
        ax.text(min(scores[i]+.06,9.25) if np.isfinite(scores[i]) else .1,i,label,va='center',fontsize=8)
    ax.set_xlabel(scope+' quality / 10, complete research with valid reviews')
    ax.set_title('Completed controlled cases; public questions not run yet' if initial else 'Full pipeline: controlled categories and adapted public questions')
    save(fig,'category_quality')
    pairs={n:x for n,x in data['paired_full_minus_control'].items() if data['ablations'][n]['complete']==12};names=list(pairs)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for i,n in enumerate(names):
        d=pairs[n];value=d['mean'];ci=d['ci95']
        if value is None:continue
        if ci is None:
            axes[0].plot(value,i,'o',color='#437eac');continue
        axes[0].errorbar(value,i,xerr=np.array([[max(0,value-ci[0])],[max(0,ci[1]-value)]]),fmt='o',color='#437eac',capsize=4)
    axes[0].set_yticks(range(len(names)),names);axes[0].invert_yaxis();axes[0].axvline(0,color='#8894a3',linestyle='--')
    axes[0].set_title('Full minus control: matched quality difference')
    axes[0].set_xlabel('Quality / 10; descriptive paired 95% bootstrap CI')
    variants={n:x for n,x in data['ablations'].items() if x['complete']==12};vnames=list(variants)
    axes[1].barh(vnames,[variants[n]['research_tokens']/1e6 for n in vnames],color='#8a70a7')
    axes[1].invert_yaxis();axes[1].set_xlabel('Research tokens, millions (12 matched anchors each)')
    axes[1].set_title('Observed resource tradeoffs, single runs')
    save(fig,'module_tradeoffs')
    pre=data['pre_post_m5']['pairs'];fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
    for i,r in enumerate(pre):
        ax.plot([r['before'],r['after']],[i,i],color='#bf6b44' if r['difference']<0 else '#50947b',linewidth=2)
    ax.scatter([r['before'] for r in pre],range(len(pre)),label='Before M5',color='#437eac',s=28,zorder=3)
    ax.scatter([r['after'] for r in pre],range(len(pre)),label='After M5',color='#c48848',marker='x',s=42,zorder=4)
    ax.set_yticks(range(len(pre)),[r['question_id'] for r in pre]);ax.invert_yaxis();ax.set_xlim(0,10)
    ax.set_xlabel('Uniform v4 quality / 10; exact saved before/after reports')
    ax.set_title('M5 can improve or degrade: inspect evidence propagation');ax.legend()
    if pre:save(fig,'m5_before_after')
    else:plt.close(fig)
    report=REPORT
    text=report.read_text().split('\n## 图表与案例浏览\n',1)[0]+'\n## 图表与案例浏览\n\n'
    for name,label in (('category_quality','各方向报告质量'),('module_tradeoffs','同题模块质量和成本'),('m5_before_after','同次研究的对抗前后变化')):
        if name=='m5_before_after' and not pre:continue
        text+=rebase_links(f'![{label}]({folder.relative_to(ROOT)}/{name}.png)\n\n',ROOT,report.parent)
    text+=rebase_links(f'[打开本地交互案例浏览器]({base.relative_to(ROOT)}/dashboard.html)，可按方向、模块配置和评审有效性筛选，并打开报告、对抗前正文和 Trace。\n',ROOT,report.parent)
    report.write_text(text)
    (base/'final_report.md').write_text(rebase_links(text,report.parent,base))
    review_folder='initial_report_analysis' if initial else 'uniform_report_v4'
    reviews=json.loads((base/review_folder/'results.json').read_text());rindex={(r['sample_id'],r['version']):r for r in reviews}
    cases=[]
    for stage in ('controlled','ablations','hotpot','positive_controls','positive_memory_control'):
        p=base/stage/'results.jsonl'
        if not p.exists():continue
        for line in p.read_text().splitlines():
            old=json.loads(line);r=rindex[(old['sample_id'],'final')];before=rindex.get((old['sample_id'],'before_m5'))
            sample=Path(old['report_path']).parent
            cases.append({'id':old['question_id'],'stage':stage,'variant':old['variant'],'category':old.get('category'),
                'status':old['status'],'valid':r['valid'],'initial_valid':r['initial_valid'],'quality':r['evaluation'].get('average'),
                'contract':r['contract']['constraint_pass'],'failures':[c['name'] for c in r['contract']['checks'] if not c['passed']],
                'facts':[{'topic':f['topic'],'verdict':f['verdict'],'quote':f.get('quote',''),'reason':f.get('reason','')} for f in r['evaluation'].get('facts',[])],
                'error':r['evaluation'].get('error'),'tokens':old.get('usage',{}).get('total_tokens',0),
                'before':before['evaluation'].get('average') if before else None,
                'report':str(Path(stage)/sample/'report_body.md'),'trace':str(Path(stage)/sample/'trace.json'),
                'pre_report':str(Path(stage)/sample/'report_before_adversarial.md') if (base/stage/sample/'report_before_adversarial.md').exists() else None})
    payload=json.dumps({'cases':cases,'summary':data['groups'],'qa':data['qa_summary']},ensure_ascii=False).replace('<','\\u003c')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DeepResearch 扩展测评案例</title>
<style>body{max-width:1250px;margin:35px auto;padding:0 18px;background:#f6f8fb;color:#193247;font:15px/1.6 system-ui}h1{font-size:27px}a{color:#256aa2}section{background:white;border:1px solid #dde5ee;border-radius:10px;padding:18px;margin:14px 0}select,input{padding:8px;margin:5px;border:1px solid #acbbcb;border-radius:5px}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;border-bottom:1px solid #e2e7ee;padding:9px;vertical-align:top}th{background:#e9f0f7}small{color:#536b80}summary{cursor:pointer}pre{white-space:pre-wrap;word-break:break-word;background:#f5f7fa;padding:10px} .scroll{overflow-x:auto}</style>
<h1>DeepResearch 扩展测评案例浏览</h1><p>本地离线浏览。质量采用统一 v4 证据口径，契约单列；无效评分显示“—”，不会填成零分。主实验与补充探针分别标记。<a href="final_report.md">完整报告</a> · <a href="final_analysis.json">原始分析</a></p>
<section id="stats"></section><section><label>方向<select id="category"><option value="">全部</option></select></label><label>配置<select id="variant"><option value="">全部</option></select></label><label>分组<select id="stage"><option value="">全部</option></select></label><input id="search" placeholder="按问题编号筛选"><label><input type="checkbox" id="invalid">仅看无效评审</label><label><input type="checkbox" id="low">仅看质量低于7</label><p id="count"></p><div class="scroll"><table><thead><tr><th>问题 / 方向</th><th>配置 / 分组</th><th>研究 / 质量</th><th>契约 / Token</th><th>证据与产物</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const $=id=>document.getElementById(id);
function elem(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e}
for(const key of ['category','variant','stage']){for(const value of [...new Set(data.cases.map(c=>c[key]))].sort()){const o=elem('option',value);o.value=value;$(key).append(o)}$(key).onchange=render}
for(const key of ['search','invalid','low'])$(key).oninput=render;
for(const [name,s] of Object.entries(data.summary))$('stats').append(elem('p',`${name}: 完整 ${s.complete}/${s.n}，合法完整评分 ${s.quality_n}，质量 ${s.quality_mean?.toFixed(3)??'—'}，契约通过 ${s.contract_pass}/${s.n}，验收 ${s.accepted}/${s.n}`));
function link(label,path){const a=elem('a',label);a.href=path;a.target='_blank';a.rel='noopener';return a}
function render(){const rows=data.cases.filter(c=>['category','variant','stage'].every(k=>!$(k).value||$(k).value===c[k])&&c.id.includes($('search').value)&&(!$('invalid').checked||!c.valid)&&(!$('low').checked||(typeof c.quality==='number'&&c.quality<7)));$('count').textContent=`显示 ${rows.length} / ${data.cases.length} 个研究案例`;$('rows').replaceChildren();for(const c of rows){const tr=elem('tr');const a=elem('td',c.id);a.append(elem('br'),elem('small',c.category));const b=elem('td',c.variant);b.append(elem('br'),elem('small',c.stage));const score=elem('td',`${c.status} / ${c.quality?.toFixed(3)??'—'}`);score.append(elem('br'),elem('small',`首评${c.initial_valid?'合法':'无效'}${c.before!=null?'；M5前 '+c.before.toFixed(3):''}`));const d=elem('td',`${c.contract?'通过':'未通过'} / ${c.tokens.toLocaleString()}`);d.append(elem('br'),elem('small',c.failures.join(', ')));const e=elem('td');e.append(link('报告',c.report),document.createTextNode(' · '),link('Trace',c.trace));if(c.pre_report)e.append(document.createTextNode(' · '),link('对抗前',c.pre_report));const details=elem('details');details.append(elem('summary','事实判定 / 错误详情'),elem('pre',JSON.stringify(c.error??c.facts,null,2)));e.append(details);tr.append(a,b,score,d,e);$('rows').append(tr)}}render();
</script></html>'''.replace('PAYLOAD',payload)
    if initial:
        html=html.replace('质量采用统一 v4 证据口径，契约单列','当前为 v3 首评＋v2 契约重算，统一 v4 尚未实跑。全量实验因余额不足中断')
    (base/'dashboard.html').write_text(html)
    print('Generated PNG/SVG figures and offline dashboard:',base/'dashboard.html')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);run(p.parse_args().base)
