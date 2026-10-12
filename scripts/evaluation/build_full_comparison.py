"""Write auditable before/after and historical-result tables; never use partial scores."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.data.submission_benchmarks import ROOT,REGISTRY,write_json,sha256


def percent(metrics):
    for key in ['strict_itt_accuracy','group_score','vqa_accuracy','accuracy']:
        if key in metrics:
            return 100*metrics[key]
    return None


def emit(out,name,columns,rows):
    out.mkdir(parents=True,exist_ok=True)
    def display(v):
        return '--' if v is None else f'{v:.2f}' if isinstance(v,float) else str(v)
    with (out/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows([[display(r.get(c)) for c in columns] for r in rows])
    text='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    text+=''.join('| '+' | '.join(display(r.get(c)).replace('|','/') for c in columns)+' |\n' for r in rows)
    (out/f'{name}.md').write_text(text,encoding='utf-8')
    def escape(v):
        return display(v).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    tex='\\begin{tabular}{'+'l'*len(columns)+'}\n\\hline\n'+' & '.join(escape(c) for c in columns)+r' \\'+'\n\\hline\n'
    tex+=''.join(' & '.join(escape(r.get(c)) for c in columns)+r' \\'+'\n' for r in rows)
    tex+='\\hline\n\\end{tabular}\n'
    (out/f'{name}.tex').write_text(tex,encoding='utf-8')


def build(data_root,result_root,out):
    primary=[]; coverage=[]; sources=[]; history=[]
    registry=dict(REGISTRY)
    for name in ['iconqa_choose_img','iconqa_fill_in_blank']:
        mp=data_root/name/'manifest.json'
        if mp.exists():registry[name]=dict(split='test')
    for name,spec in registry.items():
        mp=data_root/name/'manifest.json'
        m=json.loads(mp.read_text()) if mp.exists() else {}
        rp=result_root/name/'result.json'
        paired=json.loads(rp.read_text()) if rp.exists() else {}
        oldpath=ROOT/'experiments/results/submission_full'/name/'result.json'
        old=json.loads(oldpath.read_text()) if oldpath.exists() else {}
        native=after=None
        if paired.get('status')=='complete' and paired.get('N')==m.get('N') and paired.get('identity',{}).get('candidate_manifest_sha256')==m.get('candidate_manifest_sha256'):
            native=percent(paired['native_metrics']);after=percent(paired['jev_metrics'])
            sources.append(dict(table='before_after',dataset=name,result=str(rp),sha256=sha256(rp)))
        elif (old.get('status')=='complete' and old.get('protocol')=='full_benchmark'
              and old.get('metrics',{}).get('count')==m.get('N')
              and old.get('candidate_manifest_sha256')==m.get('candidate_manifest_sha256')):
            after=percent(old['metrics'])
            sources.append(dict(table='before_after',dataset=name,result=str(oldpath),sha256=sha256(oldpath)))
        task=m.get('task',name)
        primary.append(dict(Dataset=name,Task=task,Split=m.get('split',spec['split']),N=m.get('N'),
                            Native_pct=native,JEV_pct=after,Delta_pp=None if native is None or after is None else after-native))
        if after is not None:
            history.append(dict(Dataset=name,Protocol='complete released evaluation suite' if name=='sugarcrepe_pp' else f'official {m["split"]} / declared task',N=m['N'],Score_pct=after,
                                Source=str(rp if paired.get('status')=='complete' else oldpath)))
        sp=result_root/name/'status.json'
        status=json.loads(sp.read_text()) if sp.exists() else {}
        coverage.append(dict(dataset=name,N_source=m.get('N_source'),N_task=m.get('N'),task=task,
                             excluded_tasks=m.get('excluded',{}),access=m.get('status'),reason=m.get('reason'),
                             paired_run_status=status,paired_result_complete=paired.get('status')=='complete',
                             existing_jev_complete=after is not None,
                             followup_command=m.get('followup_command')))
    # Custom results are separate historical evidence, never same-split deltas.
    for name in ['aokvqa','scienceqa','iconqa','coco']:
        path=ROOT/'experiments/results/independent_datasets_full'/name/'result.json'
        if not path.exists():continue
        r=json.loads(path.read_text())
        metric=r.get('test',r.get('metrics',r.get('evaluation',{})))
        if not metric:
            for key in ['test_calibrated','test_uncalibrated']:
                if r.get(key):metric=r[key];break
        history.append(dict(Dataset=name,Protocol='custom independently trained checkpoint; NOT same official split',
                            N=metric.get('count'),Score_pct=percent(metric),Source=str(path)))
        sources.append(dict(table='historical',dataset=name,result=str(path),sha256=sha256(path)))
    for backbone in ['siglip2','internvl','llava']:
        path=ROOT/'experiments/results/backbone_generality'/backbone/'results.json'
        if not path.exists():continue
        r=json.loads(path.read_text());metric=r.get('test_uncalibrated',{})
        history.append(dict(Dataset='COCO',Protocol=f'capped custom backbone pilot / {backbone}; NOT official full test',
                            N=r.get('split_counts',{}).get('test'),Score_pct=percent(metric),Source=str(path)))
        sources.append(dict(table='historical',dataset='COCO',result=str(path),sha256=sha256(path)))
    emit(out,'before_after',['Dataset','Task','Split','N','Native_pct','JEV_pct','Delta_pp'],primary)
    emit(out,'historical_results',['Dataset','Protocol','N','Score_pct','Source'],history)
    write_json(out/'coverage.json',coverage)
    write_json(out/'table_sources.json',sources)
    write_json(out/'before_after.json',primary)
    text='# 完整评测与接入 JEV 前后对比\n\n'
    text+='原生对照：相同 Qwen 权重原始 LM head，在图像与问题条件下计算每个候选答案的平均 token log-likelihood。接入后：现有 controlled-COCO Visual-JEV checkpoint。两侧题目、候选集和顺序一致；这不是官方 Visual Jev SFT，也不是训练预算相同的因果比较。\n\n'
    text+='只有全量覆盖产生 result.json；进行中的分数不显示。`--` 表示尚未完成，不表示 0。\n\n'
    text+=(out/'before_after.md').read_text()+'\n## 任务覆盖与缺口\n\n'
    for c in coverage:
        text+=f'- {c["dataset"]}: source={c["N_source"]}, task={c["N_task"]}, excluded={c["excluded_tasks"]}; access={c["access"]}; run={c["paired_run_status"]}.\n'
        if c['reason']:text+=f'  原因：{c["reason"]}\n'
    text+='\n## 历史结果（不同 split，不能用于前后增益）\n\n'+(out/'historical_results.md').read_text()
    (out/'README.md').write_text(text,encoding='utf-8')
    return primary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data/submission_benchmarks')
    p.add_argument('--result-root',type=Path,default=ROOT/'experiments/results/complete_eval_20261007')
    p.add_argument('--output',type=Path,default=ROOT/'reports/full-benchmark-20261007')
    a=p.parse_args();build(a.data_root,a.result_root,a.output)
