#!/usr/bin/env python3
"""Publish an evidence-only submission snapshot and optional paper table copies."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.submission_benchmarks import ROOT,REGISTRY,DEFAULT_CHECKPOINT,sha256,write_json

CODE_FILES=['scripts/submission_benchmarks.py','scripts/evaluate_submission_benchmarks.py',
            'scripts/audit_submission_sources.py',
            'scripts/vqa_official_scoring.py','scripts/generate_submission_tables.py',
            'scripts/generate_submission_report.py','scripts/run_submission_benchmarks.sh',
            'scripts/resume_submission_benchmark.py','scripts/run_submission_full_training.sh',
            'scripts/download_visual_benchmarks.py','tests/test_submission_benchmarks.py',
            'BENCHMARK_PROTOCOL.md']
PAPER_FILES=['sec/0_abstract.tex','sec/1_intro.tex','sec/2_formatting.tex',
             'sec/4_submission_experiments.tex','sec/3_finalcopy.tex',
             'sec/X_suppl.tex','preamble.tex','README.md','COMPILE_STATUS.md','REBUILD_PAPER.ps1']

def load(path): return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data/submission_benchmarks')
    p.add_argument('--result-root',type=Path,default=ROOT/'experiments/results/submission_full')
    p.add_argument('--tables',type=Path,default=ROOT/'experiments/tables/submission')
    p.add_argument('--output',type=Path,default=ROOT/'reports/SUBMISSION_EXPERIMENT_REPORT.md')
    p.add_argument('--paper-root',type=Path,action='append',default=[])
    args=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_submission_tables.py'),
                    '--data-root',str(args.data_root),'--result-root',str(args.result_root),
                    '--output',str(args.tables)],check=True)
    timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
    completed=[]; pending=[]; rows=[]
    for name,spec in REGISTRY.items():
        mp=args.data_root/name/'manifest.json'
        m=load(mp) if mp.exists() else dict(status='not prepared',N=None,split=spec['split'])
        rp=args.result_root/name/'result.json'
        r=load(rp) if rp.exists() else None
        state=m['status']; progress='--'; metric='--'; score='--'
        sp=args.result_root/name/'status.json'
        evaluation_reason=None
        if sp.exists():
            status=load(sp); state=status.get('status',state)
            evaluation_reason=status.get('reason')
            if status.get('N_complete') is not None: progress=f"{status['N_complete']}/{status['N_total']}"
        if r is not None:
            state='complete'; metrics=r['metrics']
            metric=next(k for k in ['group_score','strict_itt_accuracy','vqa_accuracy','accuracy'] if k in metrics)
            score=f"{100*metrics[metric]:.2f}"
            completed.append(dict(dataset=name,N=r['N'],metric=metric,value=metrics[metric],
                                  result=str(rp),sha256=sha256(rp),checkpoint_sha256=r['checkpoint_sha256']))
        else:
            pending.append(dict(dataset=name,status=state,N=m.get('N'),progress=progress,
                                reason=evaluation_reason or m.get('reason','Full-split evaluation has not completed.'),
                                prepare_command=m.get('followup_command'),
                                evaluation_command=f'.venv/bin/python scripts/evaluate_submission_benchmarks.py --datasets {name} --controls --repair-wrong-controls'))
        rows.append([name,m.get('split',spec['split']),m.get('N') or '--',state,progress,metric,score])
    lines=['# Visual-JEV submission experiment delivery','',f'Snapshot: {timestamp}.',
           '',f'Checkpoint: `{DEFAULT_CHECKPOINT}`.',f'SHA256: `{sha256(DEFAULT_CHECKPOINT)}`.',
           '', 'All main scores use one fixed pre-existing controlled-COCO checkpoint; no new evaluation fitting.',
           'No sample/debug or custom matched result enters the official full-split table.',
           'This is an experiment implementation and evidence snapshot, not a claim that every submission gap is closed.',
           '', '## Complete-split status and real results','',
           '| Dataset/task | Official split | Full N | Status | Progress (not a score) | Metric | Score (%) |',
           '|---|---|---:|---|---|---|---:|']
    lines += ['| '+' | '.join(map(str,row))+' |' for row in rows]
    queue_path=args.result_root/'active_queue.json'
    if queue_path.exists():
        queue=load(queue_path)
        lines += ['', '## Existing full-split execution queue','',
                  f"Worker PID: `{queue['pid']}`; queue state: `{queue['status']}`.",
                  'Order: '+' -> '.join(queue['datasets'])+'.',
                  f"Log: `{queue.get('log_file')}`.",
                  'Already started/queued full evaluations need not be launched again. Do not run a competing GPU evaluator.',
                  'Queue metadata is a snapshot; per-dataset status/result files are authoritative.']
    lines += ['', 'ScienceQA includes intrinsic text-only questions; IconQA is the complete native select_txt task.',
              'TextVQA and GQA are candidate-constrained adaptations, not unrestricted leaderboard scores.',
              'SugarCrepe++ is the entire released evaluation suite, not a true-test claim.',
              'Completed transfer scores do not establish broad benchmark competitiveness; blank/wrong controls must be read alongside accuracy.',
              '', '## Remaining exact commands','', 'Run commands from `~/projects/Open-Jev`. Do not launch another GPU evaluator while the current one is running.']
    for gap in pending:
        lines += ['',f"### {gap['dataset']} ({gap['status']})",'',gap['reason']]
        if gap['dataset']=='snli_ve': lines += ['Obtain authorized Flickr30K images first and place them in `data/flickr30k-images/`.']
        if gap['dataset']=='winoground': lines += ['Get approval on the official Hub dataset page, then authenticate using `.venv/bin/hf auth login`; never log a token.']
        if gap['dataset']=='nlvr2': lines += ['Satisfy official researcher registration/image terms before using `--nlvr2-authorized`.']
        if gap['dataset']=='gqa': lines += ['All 1833 training-only answer candidates are retained; full scoring is computationally expensive.']
        if gap['dataset']=='textvqa': lines += ['Current ready manifest uses the fixed/OCR candidate policy. The optional 5000-word training-vocabulary policy requires a fresh data/result root.']
        lines += ['', '```bash']
        if gap['prepare_command'] and gap['status']=='blocked': lines += [gap['prepare_command']]
        lines += [gap['evaluation_command'],'```']
        recovery=f'.venv/bin/python scripts/resume_submission_benchmark.py {gap["dataset"]}'
        if gap['dataset']=='snli_ve': recovery+=' --flickr30k-root data/flickr30k-images'
        if gap['dataset']=='nlvr2': recovery+=' --nlvr2-authorized'
        lines += ['', 'One-command preparation/evaluation/table refresh after satisfying the prerequisite:',
                  '', '```bash',recovery,'```']
    lines += ['', '## Reproducibility and table outputs','',
              f'CSV/Markdown/LaTeX: `{args.tables}`.',
              'Five table families: Main Full Benchmark Generalization; Matched Visual Jev Comparison;',
              'Visual Dependency; Latency/Params; Ablation. `table_sources.json` retains real JSON paths/hashes.',
              'Unverified wrong-image controls are withheld until content-SHA verification; semantic pairs unavailable on official splits remain `--`.',
              'Full timing includes diagnostic passes/cache reuse and is not pooled with official Visual Jev model-run timing.',
              '', '## Added/modified implementation files','']
    lines += [f'- `{path}`' for path in CODE_FILES]
    lines += ['', 'Paper source/layout changes:', '']
    lines += [f'- `{path}`' for path in PAPER_FILES]
    lines += ['', '## Verification and pending training','',
              'Focused checks: `.venv/bin/python -m pytest tests/test_submission_benchmarks.py -q`.',
              'Table generation validates complete manifest N, checkpoint/split identity, candidate and prediction file hashes.',
              'No oversized training run was shrunk or silently started. Existing full feature-manifest training can be reproduced with:',
              '', '```bash', '.venv/bin/python run_independent_datasets.py --train-cap 0 --result-root experiments/results/independent_datasets_full_reproduction --resume','```',
              '', 'That training uses the historical custom held-out protocol, not the official full benchmark protocol.',
              'New benchmark-specific training/model selection requires its own train/validation boundary and provenance.',
              'The frozen backbone pretraining exposure and controlled-COCO image overlap are not certified contamination-free.',
              '', '## Paper update','',
              'The current Experiments include `sec/4_submission_experiments.tex` from `sec/2_formatting.tex`.',
              'Abstract/introduction/conclusion no longer present historical sampled/custom counts as full benchmark evidence.',
              'Any PDF is a compilation-time snapshot. Regenerate tables and recompile after new result.json files arrive.',
              'Windows one-command rebuild: run `./REBUILD_PAPER.ps1` in the delivered paper project; it uses an existing compiler and installs nothing.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(args.output.with_suffix('.json'),dict(timestamp=timestamp,completed=completed,pending=pending,code_files=CODE_FILES,paper_files=PAPER_FILES))
    for paper in args.paper_root:
        if not (paper/'main.tex').is_file(): raise ValueError(f'Not a paper project: {paper}')
        dest=paper/'tables/submission'; dest.mkdir(parents=True,exist_ok=True)
        for path in args.tables.iterdir():
            if path.is_file(): shutil.copy2(path,dest/path.name)
        shutil.copy2(ROOT/'BENCHMARK_PROTOCOL.md',paper/'BENCHMARK_PROTOCOL.md')
        shutil.copy2(args.output,paper/'SUBMISSION_EXPERIMENT_REPORT.md')
        shutil.copy2(args.output.with_suffix('.json'),paper/'SUBMISSION_EXPERIMENT_REPORT.json')
        evidence=paper/'evidence'; evidence.mkdir(exist_ok=True)
        if queue_path.exists(): shutil.copy2(queue_path,evidence/'active_queue.json')
        for name in REGISTRY:
            for source,filename in [(args.data_root/name/'manifest.json',name+'_manifest.json'),
                                    (args.result_root/name/'result.json',name+'_result.json')]:
                if source.exists(): shutil.copy2(source,evidence/filename)
        for name in ['tallyqa','sugarcrepe_pp']:
            audit=args.data_root/name/'official_annotation_audit.json'
            if audit.exists(): shutil.copy2(audit,evidence/(name+'_official_annotation_audit.json'))
        archive=ROOT/'experiments/results/submission_legacy'
        for source in archive.glob('*/result.json'): shutil.copy2(source,evidence/(source.parent.name+'_result.json'))
        for source in CODE_FILES:
            if source=='BENCHMARK_PROTOCOL.md': continue
            target=paper/'reproducibility'/source; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/source,target)
    print(f'Report: {args.output}; complete={len(completed)}; pending={len(pending)}',flush=True)

if __name__=='__main__': main()
