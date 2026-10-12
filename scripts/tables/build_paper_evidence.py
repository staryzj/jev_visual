"""Audit full aggregate records and regenerate manuscript accuracy/cost tables.

No model inference or resampling. Raw result bytes and hashes remain unchanged.
Usage: python -m scripts.tables.build_paper_evidence --archive RESULTS.tar.gz
       --output reports/full_study_20261008 --paper /path/to/overleaf
Without --archive, verify and rebuild the published aggregate snapshot.
"""
import argparse
import csv
import hashlib
import json
import statistics as st
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
BACKBONES = ('siglip2', 'internvl', 'llava')
TASKS = ('aokvqa', 'scienceqa', 'iconqa', 'iconqa_choose_img',
         'iconqa_fill_in_blank', 'ai2d', 'vsr', 'sugarcrepe_pp',
         'textvqa', 'tallyqa', 'gqa', 'snli_ve')
LABELS = ('A-OKVQA', 'ScienceQA (all)', 'IconQA text', 'IconQA image',
          'IconQA fill', 'AI2D', 'VSR random', 'SugarCrepe++',
          'TextVQA', 'TallyQA', 'GQA balanced', 'SNLI-VE')
SPLITS = ('val.', 'test', 'test', 'test', 'test', 'test', 'test',
          'suite', 'val.', 'test', 'test-dev', 'test')
ARMS = ('linear_ce', 'mlp_ce', 'mlp_invalid')
SEEDS = {'20260928', '20260929', '20260930'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def export_snapshot(archive):
    """Allowlist result/identity files only, not images, weights or traces."""
    files = []
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            p = PurePosixPath(item.name)
            if not item.isfile() or p.is_absolute() or '..' in p.parts:
                continue
            if not item.name.startswith(('experiments/results/adapter_study_20261008/',
                                         'experiments/results/native_vs_jev_20261008/')):
                continue
            if p.name not in {'result.json', 'study_identity.json', 'run_manifest.json',
                               'execution_plan.json', 'parallel_execution_plan.json',
                               'resource_gaps.json', 'fallback_trigger.json'}:
                continue
            raw = tar.extractfile(item).read()
            json.loads(raw)
            target = ROOT / Path(*p.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != raw:
                raise ValueError('Refusing to overwrite distinct published evidence: ' + item.name)
            target.write_bytes(raw)
            files.append(dict(path=item.name, sha256=sha(raw), bytes=len(raw)))
    metadata = ROOT / 'release_metadata/FULL_STUDY_SNAPSHOT.json'
    metadata.write_text(json.dumps(dict(schema='full-study-snapshot-v1',
        archive_sha256=sha(archive.read_bytes()), files=files,
        scope='Exact aggregate result/identity bytes; excludes weights, images and predictions',
        provenance='Server archive supplied by author; no new training or evaluation'), indent=2)+'\n')


def verify_snapshot():
    meta = json.loads((ROOT/'release_metadata/FULL_STUDY_SNAPSHOT.json').read_text())
    for entry in meta['files']:
        assert sha((ROOT/entry['path']).read_bytes()) == entry['sha256'], entry['path']
    return meta


def table(caption, label, spec, header, rows, wide=True):
    env = 'table*' if wide else 'table'
    return '\n'.join([f'\\begin{{{env}}}[t]', r'\centering\small',
        r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.10}',
        '\\caption{'+caption+'}', '\\label{'+label+'}',
        '\\begin{tabular}{@{}'+spec+'@{}}', r'\toprule', header,
        r'\midrule', *rows, r'\bottomrule', r'\end{tabular}',
        f'\\end{{{env}}}', ''])


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/full_study_20261008')
    parser.add_argument('--paper', type=Path)
    args = parser.parse_args()
    if args.archive:
        export_snapshot(args.archive)
    snapshot = verify_snapshot()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}; native = {}; sources = []
    for b in BACKBONES:
        for task in TASKS:
            path = ROOT/f'experiments/results/adapter_study_20261008/evaluation/{b}/{task}/result.json'
            d = json.loads(path.read_bytes())
            assert d['status']=='complete' and d['no_sampling'] and d['public_labels']
            assert d['N']==d['metrics']['N']==d['identity']['N']
            assert set(d['metrics']['seeds'])==SEEDS
            for s in d['metrics']['seeds'].values():
                assert set(ARMS) <= set(s)
            records[b,task] = d
            sources.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path.read_bytes())))
            candidates = []
            for directory in ('native', 'native_uncached'):
                p = ROOT/f'experiments/results/native_vs_jev_20261008/{directory}/{b}/{task}/result.json'
                if not p.exists():
                    continue
                n = json.loads(p.read_bytes())
                if n.get('status')=='complete' and n.get('N')==d['N']:
                    assert n['split']==d['split']
                    assert n['identity']['candidate_manifest_sha256']==d['identity']['candidate_manifest_sha256']
                    assert n['identity']['model_files']==d['identity']['feature_identity']['visual_models'][b]
                    candidates.append((p,n))
            if candidates:
                # Uncached full reruns replace numerically rejected cached outputs.
                p,n = candidates[-1]
                native[b,task]=n
                sources.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p.read_bytes())))
    scorer_states={d['identity']['scorer_state_sha256'] for d in records.values()}
    assert len(scorer_states)==1

    def scores(b,t,arm='mlp_invalid'):
        return [100*s[arm]['primary'] for s in records[b,t]['metrics']['seeds'].values()]

    accuracy=[]; latency=[]; table_rows=[]; main_rows=[]
    for t,label,split in zip(TASKS,LABELS,SPLITS):
        cells=[label,split,f"{records['siglip2',t]['N']:,}"]
        complete=all((b,t) in native for b in BACKBONES)
        for b in BACKBONES:
            v=scores(b,t); mean=st.fmean(v); sd=st.stdev(v)
            nv=100*native[b,t]['native_metric']['score'] if (b,t) in native else None
            accuracy.append(dict(backbone=b,dataset=t,split=records[b,t]['split'],
                N=records[b,t]['N'],native_score_pct=nv,jev_mean_pct=mean,jev_seed_sd_pct=sd,
                delta_pp=None if nv is None else mean-nv))
            ns='--' if nv is None else f'{nv:.2f}'
            js=f'{mean:.2f}'
            if nv is not None:
                if mean>nv: js=r'\textbf{'+js+'}'
                elif nv>mean: ns=r'\textbf{'+ns+'}'
            cells += [ns,js+r'\,$\pm$\,'+f'{sd:.2f}']
            for arm in ARMS:
                vals=[s[arm]['adapter_scorer_ms_mean'] for s in records[b,t]['metrics']['seeds'].values()]
                assert all(v>0 for v in vals)
                params={r[arm]['adapter_parameters'] for r in records[b,t]['identity']['checkpoints'].values()}
                assert len(params)==1
                latency.append(dict(backbone=b,dataset=t,arm=arm,
                    adapter_parameters=params.pop(),N=records[b,t]['N'],
                    cached_mean_ms=st.fmean(vals),between_seed_sd_ms=st.stdev(vals),
                    scope='Synchronized cached adapter+scorer incl transfer/readout; excludes encoders/cache/controls'))
        row=' & '.join(cells)+r' \\'
        table_rows.append(row)
        if complete: main_rows.append(row)
    assert len(main_rows)==10 and len(records)==36 and len(native)==32
    wins=sum(v['delta_pp']>0 for v in accuracy if v['delta_pp'] is not None)
    assert wins==21
    assert sum(st.fmean(scores(b,t,'mlp_ce'))>st.fmean(scores(b,t,'linear_ce'))
               for b in BACKBONES for t in TASKS)==29
    assert sum(records['siglip2',t]['N'] for t in TASKS)==110983
    header=(r'Task & Split & $N$ & \multicolumn{2}{c}{SigLIP2} & \multicolumn{2}{c}{InternVL} & \multicolumn{2}{c}{LLaVA} \\'+'\n'+
        r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}'+'\n'+
        r' & & & Native & JEV & Native & JEV & Native & JEV \\')
    main_tex=table(r'Native and adapted JEV scores (\%) on ten complete paired tasks. JEV is MLP+Null mean $\pm$ three-seed SD; bold marks the higher score within each pair.',
        'tab:native-main',r'llrrr@{\hspace{6pt}}rr@{\hspace{6pt}}rr',header,main_rows)
    full_tex=table(r'Full twelve-task Native/JEV record (\%). JEV is mean $\pm$ three-seed SD; -- marks four incomplete native runs.',
        'tab:full-results',r'llrrr@{\hspace{6pt}}rr@{\hspace{6pt}}rr',header,table_rows)
    time_rows=[]
    for t,label in [('scienceqa','ScienceQA'),('textvqa','TextVQA'),
                    ('iconqa_fill_in_blank','IconQA fill'),('gqa','GQA')]:
        vals=[next(v for v in latency if v['backbone']==b and v['dataset']==t and v['arm']=='mlp_invalid') for b in BACKBONES]
        time_rows.append(' & '.join([label]+[f"{v['cached_mean_ms']:.2f}"+r'\,$\pm$\,'+f"{v['between_seed_sd_ms']:.2f}" for v in vals])+r' \\')
    time_tex=table(r'H100 cached adapter-plus-scorer time (ms/example): MLP+Null mean $\pm$ between-seed SD. Encoder time is excluded.',
        'tab:latency-main','lrrr',r'Task & SigLIP2 & InternVL & LLaVA \\',time_rows,False)
    all_time_rows=[]
    for t,label in zip(TASKS,LABELS):
        vals=[next(v for v in latency if v['backbone']==b and v['dataset']==t and v['arm']=='mlp_invalid') for b in BACKBONES]
        all_time_rows.append(' & '.join([label]+[f"{v['cached_mean_ms']:.2f}"+r'\,$\pm$\,'+f"{v['between_seed_sd_ms']:.2f}" for v in vals])+r' \\')
    all_time_tex=table(r'All-task cached adapter-plus-scorer latency (ms/example): MLP+Null mean $\pm$ between-seed SD.',
        'tab:latency-all','lrrr',r'Task & SigLIP2 & InternVL & LLaVA \\',all_time_rows,False)
    for name,content in [('native_main',main_tex),('full_results',full_tex),('latency_main',time_tex),('latency_all',all_time_tex)]:
        (args.output/(name+'.tex')).write_text(content,encoding='utf-8')
        if args.paper:
            (args.paper/'tables'/(name+'.tex')).write_text(content,encoding='utf-8')
    csv_write(args.output/'accuracy.csv',accuracy)
    csv_write(args.output/'cached_latency.csv',latency)
    md=['# Native / adapted JEV complete-split scores',
        '| Backbone | Task | N | Native (%) | JEV (%) | Delta (pp) |',
        '|---|---|---:|---:|---:|---:|']
    for v in accuracy:
        ns='--' if v['native_score_pct'] is None else f"{v['native_score_pct']:.2f}"
        ds='--' if v['delta_pp'] is None else f"{v['delta_pp']:+.2f}"
        md.append(f"| {v['backbone']} | {v['dataset']} | {v['N']} | {ns} | {v['jev_mean_pct']:.2f} ± {v['jev_seed_sd_pct']:.2f} | {ds} |")
    (args.output/'accuracy.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    audit=dict(archive_sha256=snapshot['archive_sha256'],jev_complete=36,native_complete=32,
        jev_improvements=wins,main_paired_tasks=10,rows_per_backbone=110983,
        scorer_state_sha256=next(iter(scorer_states)),sources=sources,
        latency_scope='Cached stage only; no native end-to-end speedup or generated-token measurement',
        seed_uncertainty='Sample SD over three adapters, not within-run latency variability',
        predictions_present=False)
    (args.output/'evidence_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='sources'},indent=2))


if __name__=='__main__':
    main()
