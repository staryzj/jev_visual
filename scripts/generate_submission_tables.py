#!/usr/bin/env python3
"""Generate all submission tables exclusively from audited result.json files.

Legacy results are wrapped losslessly with source hashes and custom-protocol
labels. They cannot enter the Full Benchmark Generalization table.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.submission_benchmarks import ROOT,REGISTRY,sha256,write_json,read_jsonl

DISPLAY={'gqa':'GQA balanced','snli_ve':'SNLI-VE','textvqa':'TextVQA (OCR-constrained)','tallyqa':'TallyQA (0--15)','scienceqa':'ScienceQA (all)','aokvqa':'A-OKVQA','iconqa':'IconQA select-txt','ai2d':'AI2D','nlvr2':'NLVR2 test-P','vsr':'VSR random','winoground':'Winoground (group)','sugarcrepe_pp':'SugarCrepe++ (strict ITT)'}

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def pct(value): return '--' if value is None else f'{100*float(value):.2f}'
def num(value,digits=4): return '--' if value is None else f'{float(value):.{digits}f}'
def tex_escape(text):
    result=str(text)
    for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]: result=result.replace(a,b)
    return result

def latex_presentation(filename,headers,rows,caption):
    # Short display labels only: CSV/Markdown retain complete identifiers/boundaries.
    aliases={'Official Visual Jev answer-SFT':'Official Jev SFT',
             'Visual-JEV separately trained':'Visual-JEV (custom)',
             'same fixed checkpoint':'Fixed Visual-JEV',
             'Visual-JEV fixed checkpoint':'Fixed Visual-JEV',
             'scienceqa_image_only':'ScienceQA image',
             'coco_hard_negative':'COCO hard-neg',
             'iconqa_choice':'IconQA choice',
             'controlled/matched':'Controlled',
             'custom independent':'Custom',
             'matched Visual Jev':'Matched',
             'full benchmark':'Full',
             'sugarcrepe_pp':'SugarCrepe++ ITT',
             'evaluation (HF storage split=train)':'released suite (HF train)'}
    def short(value):
        value=str(value)
        for a,b in aliases.items(): value=value.replace(a,b)
        return value
    display=[[short(v) for v in row] for row in rows]
    if filename=='visual_dependency':
        headers=['Protocol','Model','Dataset','N','Orig. (%)','Blank (%)','Wrong (%)','Pairs','CF (%)','Both (%)','Flip (%)']
        display=[row for original,row in zip(rows,display) if not (original[0]=='matched Visual Jev' and original[7]=='--')]
        caption+=' Matched sets without recorded controls/pairs are not repeated here; their original accuracy appears in the matched comparison table. Complete row inventory is retained in CSV/Markdown.'
    if filename=='latency_params':
        for original,row in zip(rows,display):
            row[-1]='B1' if original[-1].startswith('official model-run') else ('B2' if original[-1].startswith('full decision') else 'B0')
        caption+=' B0: parameter provenance only; matched live timing unmeasured. B1: official model-run, group preparation excluded. B2: full decision with diagnostic control passes and repeated-image caching; no speed comparison across boundaries.'
    return headers,display,caption

def normalize_legacy(output):
    normalized=[]
    controlled=ROOT/'experiments/results/controlled_multiseed/seed-20260928/results/metrics.json'
    provenance=controlled.parent/'visual_jev_v3_provenance.json'
    if controlled.exists():
        d=load(controlled); prov=load(provenance)
        normalized.append(dict(schema='visual-jev-result-v1',protocol='matched_controlled',status='complete',dataset_name='COCO controlled',split='custom held-out',N=d['test_records'],seed=20260928,metrics=d,provenance=prov,source=str(controlled),source_sha256=sha256(controlled),source_provenance_sha256=sha256(provenance)))
        write_json(output/'controlled/result.json',normalized[-1])
    for name in ['coco','aokvqa','scienceqa','iconqa']:
        path=ROOT/f'experiments/results/independent_datasets_full/{name}/result.json'
        if not path.exists(): continue
        d=load(path)
        normalized.append({**d,'protocol':'matched_custom_independent','status':'complete','dataset_name':name,'split':'custom repartition of labelled source (not official test)','N':d['metrics']['count'],'source':str(path),'source_sha256':sha256(path)})
        write_json(output/f'independent_{name}/result.json',normalized[-1])
    for seed in [0,1,2]:
        base=ROOT/f'experiments/results/official_visual_jev_matched/seed-{seed}'
        for name in ['controlled','aokvqa','scienceqa_image_only','iconqa_choice','coco_hard_negative']:
            path=base/f'{name}.json'
            if not path.exists(): continue
            d=load(path)
            normalized.append({**d,'schema':'visual-jev-result-v1','protocol':'matched_visual_jev','status':'complete','dataset_name':name,'model':'Official Visual Jev answer-SFT','seed':seed,'split':'custom held-out','N':d['metrics']['count'],'source':str(path),'source_sha256':sha256(path),'official_provenance':load(base/'summary.json')})
            write_json(output/f'official_seed{seed}_{name}/result.json',normalized[-1])
    return normalized

def export_table(directory,filename,title,headers,rows,caption,label):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'{filename}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(rows)
    md=f'# {title}\n\n{caption}\n\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    md+=''.join('| '+' | '.join(str(v).replace('|','/') for v in row)+' |\n' for row in rows)
    (directory/f'{filename}.md').write_text(md,encoding='utf-8')
    headers,rows,caption=latex_presentation(filename,headers,rows,caption)
    columns='lp{4cm}rrrrl' if filename=='main_full_benchmark' else 'l'*len(headers)
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{'+tex_escape(caption)+'}',r'\label{'+label+'}',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{'+columns+'}',r'\toprule',' & '.join(tex_escape(v) for v in headers)+r' \\',r'\midrule']
    lines+=[' & '.join(tex_escape(v) for v in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}']
    (directory/f'{filename}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data-root',type=Path,default=ROOT/'data/submission_benchmarks'); p.add_argument('--result-root',type=Path,default=ROOT/'experiments/results/submission_full'); p.add_argument('--output',type=Path,default=ROOT/'experiments/tables/submission'); p.add_argument('--legacy-output',type=Path,default=ROOT/'experiments/results/submission_legacy'); args=p.parse_args()
    legacy=normalize_legacy(args.legacy_output); mainrows=[]; full=[]; sources=[]; gaps=[]
    for name,spec in REGISTRY.items():
        mp=args.data_root/name/'manifest.json'; m=load(mp) if mp.exists() else dict(status='not prepared',N=None,split=spec['split'])
        rp=args.result_root/name/'result.json'; r=load(rp) if rp.exists() else None
        status=m['status']; values={}
        if r is not None:
            if (r.get('protocol')!='full_benchmark' or r.get('status')!='complete'
                or not m.get('complete_split') or not m.get('public_labels')
                or r['metrics'].get('count')!=m.get('N')
                or r.get('checkpoint_sha256')!=m.get('checkpoint_sha256')
                or r.get('seed')!=m.get('seed')
                or r.get('split')!=m.get('split')
                or r.get('candidate_manifest_sha256')!=m.get('candidate_manifest_sha256')
                or sha256(m['candidate_manifest'])!=m['candidate_manifest_sha256']):
                raise ValueError(f'Invalid full result: {rp}')
            if sha256(args.result_root/name/'predictions.jsonl')!=r['predictions_sha256']: raise ValueError(f'Prediction hash mismatch {name}')
            status='complete'; values=r['metrics']; full.append(r)
            sources.append(dict(table='main_full_benchmark',dataset=name,result=str(rp),sha256=sha256(rp),json_pointer='/metrics',manifest=str(mp)))
        sp=args.result_root/name/'status.json'
        if r is None and sp.exists(): status=load(sp).get('status',status)
        accuracy=values.get('group_score',values.get('strict_itt_accuracy',values.get('vqa_accuracy',values.get('accuracy'))))
        mainrows.append([DISPLAY[name],m.get('split',spec['split']),m.get('N') if m.get('N') is not None else '--',pct(accuracy),pct(values.get('macro_f1')),num(values.get('nll')),status])
        if status!='complete': gaps.append(dict(dataset=name,status=status,N=m.get('N'),reason=m.get('reason','complete split evaluation not yet finished'),command=f'.venv/bin/python scripts/evaluate_submission_benchmarks.py --datasets {name} --controls',prepare_command=m.get('followup_command')))
    if len({r['checkpoint_sha256'] for r in full})>1:
        raise ValueError('Heterogeneous checkpoints cannot enter one fixed-checkpoint full benchmark table; use separate result/table roots')
    export_table(args.output,'main_full_benchmark','Main Full Benchmark Generalization',['Dataset/task','Official split','N','Primary score (%)','Macro-F1 (%)','NLL','Status'],mainrows,'Complete declared official splits/tasks, evaluated with the same fixed COCO checkpoint. Validation and test-dev are named explicitly. Open-answer and multi-image tasks use the disclosed candidate adaptation; these scores are not interchangeable with unrestricted leaderboard metrics. -- means no completed real result.json.','tab:submission-full')
    controlled=next((r for r in legacy if r['protocol']=='matched_controlled'),None)
    matched=[]; dependencies=[]; ablations=[]; latency=[]
    if controlled:
        d=controlled['metrics']
        for name,r in d['models'].items():
            c=r['classification']; cf=r.get('semantic_counterfactual',{}); v=r.get('visual_dependency',{})
            ablations.append([name,'controlled COCO',d['test_records'],pct(c.get('accuracy')),pct(c.get('macro_f1')),num(c.get('nll')),pct(cf.get('both_directions_accuracy')),pct(cf.get('prediction_flip_rate'))])
            dependencies.append(['controlled/matched',name,'COCO',d['test_records'],pct(v.get('original',{}).get('accuracy',c.get('accuracy'))),pct(v.get('blank',{}).get('accuracy')),pct(v.get('wrong',v.get('wrong_image',{})).get('accuracy')),cf.get('pair_count','--'),pct(cf.get('counterfactual_accuracy')),pct(cf.get('both_directions_accuracy')),pct(cf.get('prediction_flip_rate'))])
            if name=='v3_full': matched.append(['COCO controlled','custom 128',name,'20260928',d['test_records'],pct(c['accuracy']),pct(c['macro_f1']),num(c['nll'])])
        latency.append(['Visual-JEV V3','controlled checkpoint',controlled['provenance']['parameters']['full'],'--','parameter count from recorded provenance; timing pending common live harness'])
    for r in legacy:
        if r['protocol']=='matched_custom_independent':
            c=r['metrics']; matched.append([r['dataset_name'],'custom held-out','Visual-JEV separately trained','20260928',r['N'],pct(c['accuracy']),pct(c['macro_f1']),num(c['nll'])])
            v=r.get('visual_dependency',{}).get('overall',{})
            dependencies.append(['custom independent','Visual-JEV',r['dataset_name'],r['N'],pct(v.get('original',{}).get('accuracy')),pct(v.get('blank',{}).get('accuracy')),pct(v.get('wrong',{}).get('accuracy')),'--','--','--','--'])
        elif r['protocol']=='matched_visual_jev':
            c=r['metrics']; matched.append([r['dataset_name'],'custom held-out',r['model'],r['seed'],r['N'],pct(c['accuracy']),pct(c['macro_f1']),num(c['nll'])])
            cf=r.get('semantic_counterfactual',{})
            dependencies.append(['matched Visual Jev',r['model']+f' seed{r["seed"]}',r['dataset_name'],r['N'],pct(c['accuracy']),'--','--',cf.get('pair_count','--'),pct(cf.get('counterfactual_accuracy')),pct(cf.get('both_directions_accuracy')),pct(cf.get('prediction_flip_rate'))])
            timing=r.get('mean_model_seconds')
            latency.append([r['model']+f' seed{r["seed"]}',r['dataset_name'],'--',num(timing*1000 if timing is not None else None,2),'official model-run timing; prepare_group excluded; incomparable with full live decision'])
        sources.append(dict(table='legacy matched/ablation/dependency',source=r.get('source'),sha256=r.get('source_sha256'),protocol=r['protocol']))
    for r in full:
        v=r.get('visual_dependency',{}); c=r['metrics']
        wrong=v.get('wrong',{}).get('accuracy') if r.get('wrong_control_verified') else None
        dependencies.append(['full benchmark','same fixed checkpoint',r['dataset_name'],r['N'],pct(c.get('accuracy',c.get('group_score'))),pct(v.get('blank',{}).get('accuracy')),pct(wrong),'--','--','--','--'])
        latency.append(['Visual-JEV fixed checkpoint',r['dataset_name'],r['trainable_parameters'],num(r['latency_ms']['mean'],2),r['latency_boundary']])
    export_table(args.output,'matched_visual_jev','Matched Visual Jev Comparison',['Evaluation set','Split','Model','Seed/adapter','N','Accuracy (%)','Macro-F1 (%)','NLL'],matched,'Custom evaluation sets inherited from existing recorded runs. Separately trained Visual-JEV checkpoints and official answer-SFT adapters have different training provenance. This table does not establish superiority on official full benchmarks. Controlled and independent held-out sets are distinct. Official indices 0/1/2 load the released root/seed1/seed2 adapters with fixed evaluation seed 20260928; they do not estimate our model training uncertainty.','tab:submission-matched')
    export_table(args.output,'visual_dependency','Visual Dependency',['Protocol','Model','Dataset','N','Original (%)','Blank (%)','Wrong (%)','Pair N','Semantic CF (%)','Pair-both (%)','Pair flip (%)'],dependencies,'Original, blank and wrong use each declared task score: native-choice accuracy, SugarCrepe++ strict ITT or TextVQA consensus. Semantic CF is counterfactual-direction accuracy on verified image pairs. Pair-both requires both directions correct; pair flip is reported alongside it. Unavailable verified pair tests remain --. Generic wrong-image controls are diagnostic.','tab:submission-dependency')
    export_table(args.output,'latency_params','Latency/Params',['Model','Protocol/dataset','Trainable params','Measured mean ms','Timing boundary'],latency,'Parameter and timing values from recorded result.json/provenance only. Different timing boundaries are disclosed and must not be pooled into a speed comparison; controls and warm-cache policies are named explicitly. Recorded row timings exclude subsequent wrong-control repair passes.','tab:submission-latency')
    export_table(args.output,'ablation','Ablation',['Variant','Protocol','N','Accuracy (%)','Macro-F1 (%)','NLL','Pair-both (%)','Pair flip (%)'],ablations,'Recorded controlled COCO ablation, one seed. It assesses training-stage effects on the custom diagnostic and supplies no full official benchmark ablation claim.','tab:submission-ablation')
    write_json(args.output/'table_sources.json',sources); write_json(args.output/'pending_benchmarks.json',gaps)
    inventory=[]
    for path in (ROOT/'experiments/results').rglob('result.json'):
        if 'submission' in str(path): continue
        d=load(path); profile=d.get('profile','unknown')
        inventory.append(dict(path=str(path),sha256=sha256(path),profile=profile,protocol='custom_heldout' if 'full_independent' in profile else 'debug/controlled/unclassified',eligible_for_main=False,reason='No full official split manifest matching this result'))
    write_json(args.output/'legacy_result_audit.json',inventory)
    print(json.dumps(dict(full_complete=[r['dataset_name'] for r in full],pending=len(gaps),tables=str(args.output)),indent=2))

if __name__=='__main__': main()
