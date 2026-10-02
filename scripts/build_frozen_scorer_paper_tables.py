"""Build V7 core tables from real frozen-scorer result records; no inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'siglip2': 'SigLIP2', 'internvl': 'InternVL3.5', 'llava': 'LLaVA-OneVision'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper', type=Path, required=True)
    args = p.parse_args()
    audit = json.loads((ROOT/'reports/backbone-adaptation-v6/audit.json').read_text())
    rows, records, sources, controls = [], [], [], []
    rows.append(r'Qwen3-VL (reference) & 1024 & variable & source (3.68) & yes & yes & -- & -- & -- & -- & -- \\')
    for key, name in NAMES.items():
        source = ROOT/'experiments/results/backbone_generality'/key/'results.json'
        d = json.loads(source.read_text())
        assert d['status'] == 'success' and d['fixed_decision_head']
        assert d['base_checkpoint_sha256'] == audit['base_checkpoint_sha256']
        assert d['split_counts'] == {'train':512, 'validation':80, 'calibration':80, 'test':80}
        file_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        # Public aggregate bytes differ from originals. Preserve the canonical
        # source identity declared by the export; its byte hash is separately
        # verified in PUBLICATION_MANIFEST.json.
        sha = d.get('public_export_metadata', {}).get('source_sha256', file_sha)
        sources.append({'path':source.relative_to(ROOT).as_posix(),'sha256':sha,'base_checkpoint_sha256':d['base_checkpoint_sha256']})
        native = '/'.join(str(s[0]) for s in d['backbone_extraction']['native_shapes'])
        display_native = native if key != 'llava' else r'variable$^{\dagger}$'
        accuracy = f"{100*d['test_uncalibrated']['accuracy']:.2f}"
        params = f"{d['adapter_trainable_parameters']/1e6:.2f}"
        rows.append(f"{name} & {d['backbone_extraction']['hidden_dim']} & {display_native} & {params} & yes & yes & {accuracy} & -- & -- & -- & -- " + r'\\')
        vd = d['visual_dependency']
        values = [100*vd[c]['accuracy'] for c in ('original','blank','noise','wrong_image')]
        controls.append(name+' & '+' & '.join(f'{v:.2f}' for v in values)+f" & {100*vd['wrong_image']['flip_rate_from_original']:.2f} & {vd['wrong_image']['js_from_original']:.6f} "+r'\\')
        records.append({'backbone':name,'input_width':d['backbone_extraction']['hidden_dim'],'raw_tokens':native,'adapter_parameters':d['adapter_trainable_parameters'],'scorer_frozen':True,'same_scorer':True,'N':80,'accuracy_pct':float(accuracy),'zero_visual':None,'random_adapter':None,'linear_adapter':None,'semantic_pair':None,'source':source.relative_to(ROOT).as_posix(),'source_sha256':sha})
    output = args.paper/'tables/submission'
    caption = ('Core adapter-only cross-backbone study. Each target row reuses the same frozen external scorer and common cached Qwen text features on 80 custom held-out COCO examples; only its adapter is trained. Qwen is the source architecture reference, not a measured fourth row on that subset. Adapter sizes are millions of trainable parameters. All four control columns are unmeasured and remain --. $^{\\dagger}$Observed LLaVA raw token counts: 1458/2187/3645; target pooling yields 196 tokens. These are not official full test results.')
    tex = '\n'.join([r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:backbone-adaptation}',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lrrrccrrrrr}',r'\toprule',r'Backbone & $d_b$ & Raw tokens & Adapter (M) & Frozen $G$? & Same $G$? & Acc. (\%) & Zero visual & Random & Linear & Semantic pair \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}}',r'\end{table*}',''])
    (output/'backbone_adaptation.tex').write_text(tex,encoding='utf-8')
    controls_caption = ('Visual-dependence controls for the same three target-backbone runs. Original, blank, noise, wrong and wrong flip are percentages; wrong JS is mean Jensen--Shannon divergence. The historical wrong control cyclically shifts cached features without verified content/answer changes, and is not a semantic pair. No target row exceeds its blank-image accuracy.')
    (output/'backbone_controls.tex').write_text('\n'.join([r'\begin{table*}[t]',r'\centering',r'\caption{'+controls_caption+'}',r'\label{tab:backbone-controls}',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',r'\toprule',r'Backbone & Original & Blank & Noise & Wrong & Wrong flip & Wrong JS \\',r'\midrule',*controls,r'\bottomrule',r'\end{tabular*}',r'\end{table*}','']),encoding='utf-8')
    report = ROOT/'reports/frozen-scorer-v7'
    report.mkdir(parents=True,exist_ok=True)
    with (report/'core_backbone_table.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    md=['# V7 core table source data','','Qwen reference accuracy and all zero/random/linear/semantic-pair cells are unmeasured. This file records only the three real target rows.','','| Backbone | N | Adapter M | Accuracy % |','|---|---:|---:|---:|']
    md += [f"| {r['backbone']} | {r['N']} | {r['adapter_parameters']/1e6:.2f} | {r['accuracy_pct']:.2f} |" for r in records]
    (report/'core_backbone_table.md').write_text('\n'.join(md)+'\n')
    (report/'table_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print('Generated real-result V7 core and visual-use tables; all missing baselines remain unmeasured.')

if __name__ == '__main__':
    main()
