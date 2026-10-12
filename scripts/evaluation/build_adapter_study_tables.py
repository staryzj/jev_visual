"""Publish complete per-seed results and seed means/SDs; retain all signs."""
import csv
import json
import statistics
from pathlib import Path
from scripts.evaluation.jev_pair_protocol import sha256,write_json


def export(path,rows):
    columns=list(rows[0]) if rows else ['backbone','dataset','split','N']
    with path.with_suffix('.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    def cell(value): return f'{value:.2f}' if isinstance(value,float) else str(value)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines+=['| '+' | '.join(cell(row[c]) for c in columns)+' |' for row in rows]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    def escape(value): return cell(value).replace('_','\\_').replace('%','\\%').replace('&','\\&').replace('#','\\#')
    lines=['% Complete results only. SD is across training seeds, not questions.',
           '\\begin{tabular}{'+'l'*len(columns)+'}','\\hline',' & '.join(escape(c) for c in columns)+' \\\\','\\hline']
    lines+=[' & '.join(escape(row[c]) for c in columns)+' \\\\' for row in rows]
    lines+=['\\hline','\\end{tabular}']; path.with_suffix('.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def build(result_root,report_root,config,verify_predictions=False):
    report_root=Path(report_root); report_root.mkdir(parents=True,exist_ok=True)
    main=[]; controls=[]; raw=[]; parameters=[]; sources=[]; scorer_hashes=set()
    for path in sorted(Path(result_root).glob('*/*/result.json')):
        r=json.loads(path.read_text())
        if r.get('schema')!='adapter-study-full-result-v1' or r.get('status')!='complete': continue
        seeds=r['metrics']['seeds']
        if set(seeds)!={str(s) for s in config['seeds']} or r['N']!=r['metrics']['N']: raise ValueError('Incomplete seed/row denominator in paper source')
        pp=path.parent/'predictions.jsonl'
        if verify_predictions and sha256(pp)!=r['predictions_sha256']: raise ValueError('Prediction file changed')
        scorer_hashes.add(r['identity']['scorer_state_sha256']); sources.append(dict(path=str(path),sha256=sha256(path),predictions_sha256=r['predictions_sha256']))
        row=dict(backbone=r['backbone'],dataset=r['dataset_name'],split=r['split'],N=r['N'],metric=r['metrics']['metric'],training_seeds=len(seeds))
        for arm in ('linear_ce','mlp_ce','mlp_invalid'):
            values=[100*s[arm]['primary'] for s in seeds.values()]
            row[arm+'_mean_pct']=statistics.fmean(values); row[arm+'_seed_sd_pct']=statistics.stdev(values) if len(values)>1 else '--'
        delta=[100*(s['mlp_invalid']['primary']-s['linear_ce']['primary']) for s in seeds.values()]
        row['delta_pp']=statistics.fmean(delta); main.append(row)
        for seed,values in seeds.items():
            for arm in ('linear_ce','mlp_ce','mlp_invalid'):
                raw.append(dict(backbone=r['backbone'],dataset=r['dataset_name'],split=r['split'],seed=int(seed),arm=arm,N=r['N'],primary_pct=100*values[arm]['primary']))
        if all('visual_dependency' in s['mlp_invalid'] for s in seeds.values()):
            for arm in ('linear_ce','mlp_ce','mlp_invalid'):
                controls.append(dict(backbone=r['backbone'],dataset=r['dataset_name'],split=r['split'],N=r['N'],arm=arm,
                    original_pct=statistics.fmean(100*s[arm]['primary'] for s in seeds.values()),
                    blank_pct=statistics.fmean(100*s[arm]['visual_dependency']['blank']['primary'] for s in seeds.values()),
                    wrong_pct=statistics.fmean(100*s[arm]['visual_dependency']['wrong']['primary'] for s in seeds.values()),
                    blank_flip_pct=statistics.fmean(100*s[arm]['visual_dependency']['blank']['flip_rate'] for s in seeds.values()),
                    wrong_flip_pct=statistics.fmean(100*s[arm]['visual_dependency']['wrong']['flip_rate'] for s in seeds.values()),
                    semantic_counterfactual='--'))
        for arm in ('linear_ce','mlp_ce','mlp_invalid'):
            parameter_set={r['identity']['checkpoints'][str(s)][arm]['adapter_parameters'] for s in config['seeds']}
            if len(parameter_set)!=1: raise ValueError('Architecture parameter count changed across seeds')
            parameters.append(dict(backbone=r['backbone'],dataset=r['dataset_name'],split=r['split'],arm=arm,
                              adapter_parameters=next(iter(parameter_set)),scorer_frozen='yes',
                              cached_adapter_scorer_ms=statistics.fmean(s[arm]['adapter_scorer_ms_mean'] for s in seeds.values())))
    if len(scorer_hashes)>1: raise ValueError('Paper matrix contains different scorers')
    for name,rows in [('main_cross_backbone',main),('visual_dependency',controls),('per_seed',raw),('params_cached_scoring_latency',parameters)]: export(report_root/name,rows)
    write_json(report_root/'table_sources.json',dict(sources=sources,scorer_state_sha256=next(iter(scorer_hashes),None),
               seed_dispersion='sample SD across complete training seeds; no invented p-values/CIs',all_signs_retained=True))
    return dict(complete_backbone_dataset_cases=len(main),rows=main)
