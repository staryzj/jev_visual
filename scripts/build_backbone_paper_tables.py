"""Generate manuscript tables only from audited recorded cross-backbone results."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper',type=Path,required=True)
    args=p.parse_args()
    audit=json.loads((ROOT/'reports/backbone-adaptation-v6/audit.json').read_text())
    names={'siglip2':'SigLIP2','internvl':'InternVL3.5','llava':'LLaVA-OneVision'}
    rows=[]
    for identity in audit['rows']:
        result=json.loads(Path(identity['result_path']).read_text())
        assert result['base_checkpoint_sha256']==audit['base_checkpoint_sha256']
        assert result['fixed_decision_head'] and result['split_counts']=={'train':512,'validation':80,'calibration':80,'test':80}
        rows.append((names[result['backbone']],result))
    output=args.paper/'tables/submission'
    output.mkdir(parents=True,exist_ok=True)
    caption='Adapter-only cross-backbone study: the same frozen VisualJEVV3 scorer and cached Qwen candidate text features are reused. Target rows use 512/80/80/80 custom COCO train/validation/calibration/held-out examples, not official benchmark splits. Only the target adapter is trained. Qwen is the source architecture reference; no comparable 80-example result is recorded. Extraction timing excludes text encoding and is not end-to-end latency.'
    main_rows=[]
    csv_rows=[]
    for name,result in rows:
        b=result['backbone_extraction']; native='/'.join(str(shape[0]) for shape in b['native_shapes'])
        main_rows.append(f"{name} & {b['hidden_dim']} & {native} & 196 & {result['adapter_trainable_parameters']/1e6:.2f} & same/frozen & {100*result['test_uncalibrated']['accuracy']:.2f} & {b['mean_extraction_ms']:.2f} \\\\")
        v=result['visual_dependency']
        csv_rows.append({'backbone':name,'N':80,'input_dim':b['hidden_dim'],'native_tokens':native,'pooled_tokens':196,'adapter_parameters':result['adapter_trainable_parameters'],'same_frozen_head':True,'accuracy':result['test_uncalibrated']['accuracy'],'blank_accuracy':v['blank']['accuracy'],'noise_accuracy':v['noise']['accuracy'],'wrong_accuracy':v['wrong_image']['accuracy'],'wrong_flip':v['wrong_image']['flip_rate_from_original'],'wrong_js':v['wrong_image']['js_from_original'],'extract_ms':b['mean_extraction_ms'],'adapter_jev_ms':result['latency']['adapter_plus_jev_mean_ms'],'result_path':str(ROOT/'experiments/results/backbone_generality'/result['backbone']/'results.json')})
    tex='\n'.join([r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:backbone-adaptation}',r'\small',r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrlrr@{}}',r'\toprule',r'Visual backbone & $d_b$ & Native $M_b$ & Pooled $N_b$ & Adapter (M) & $G_\theta$ & Acc. (\%) & Extract (ms) \\',r'\midrule',r'Qwen3-VL (reference) & 1024 & variable & native & 3.68 & source/frozen & -- & -- \\',*main_rows,r'\bottomrule',r'\end{tabular*}',r'\end{table*}',''])
    (output/'backbone_adaptation.tex').write_text(tex)
    control_rows=[]
    for name,result in rows:
        v=result['visual_dependency']
        vals=[100*v[c]['accuracy'] for c in ('original','blank','noise','wrong_image')]
        control_rows.append(f"{name} & "+' & '.join(f'{value:.2f}' for value in vals)+f" & {100*v['wrong_image']['flip_rate_from_original']:.2f} & {v['wrong_image']['js_from_original']:.6f} & -- & -- & -- \\\\")
    controls='\n'.join([r'\begin{table*}[t]',r'\centering',r'\caption{Visual-use controls on the same target-backbone 80-example custom subset. Accuracy and flip are percentages; JS is mean Jensen--Shannon divergence of candidate distributions. Original-image accuracy is not above blank-image accuracy for any row. Wrong images are cyclically shifted cached features, not verified semantic counterfactuals; identical or answer-equivalent images were not excluded by this historical runner. Zero-token, random-adapter and trained-linear baselines have no recorded result for this exact protocol and remain --.}',r'\label{tab:backbone-controls}',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}',r'\toprule',r'Backbone & Original & Blank & Noise & Wrong & Wrong flip & Wrong JS & Zero & Random & Linear \\',r'\midrule',*control_rows,r'\bottomrule',r'\end{tabular*}',r'\end{table*}',''])
    (output/'backbone_controls.tex').write_text(controls)
    report=ROOT/'reports/backbone-adaptation-v6'
    with (report/'backbone_tables.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=csv_rows[0]);w.writeheader();w.writerows(csv_rows)
    md=['# Recorded cross-backbone results','',caption,'','| Backbone | Adapter M | Accuracy % | Blank % | Wrong flip % | Extract ms |','|---|---:|---:|---:|---:|---:|']
    for row in csv_rows: md.append(f"| {row['backbone']} | {row['adapter_parameters']/1e6:.2f} | {100*row['accuracy']:.2f} | {100*row['blank_accuracy']:.2f} | {100*row['wrong_flip']:.2f} | {row['extract_ms']:.2f} |")
    (report/'backbone_tables.md').write_text('\n'.join(md)+'\n')
    print('Generated two real-result tables; no missing baseline filled or official split substituted.')

if __name__=='__main__':main()
