#!/usr/bin/env python3
"""One-command complete-split recovery after the user resolves an access gap."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.submission_benchmarks import ROOT,REGISTRY,DEFAULT_CHECKPOINT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=list(REGISTRY))
    p.add_argument('--flickr30k-root',type=Path)
    p.add_argument('--nlvr2-authorized',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT/'data/submission_benchmarks')
    p.add_argument('--result-root',type=Path,default=ROOT/'experiments/results/submission_full')
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--paper-root',type=Path,action='append',default=[])
    args=p.parse_args()
    cmd=[sys.executable,str(ROOT/'scripts/submission_benchmarks.py'),'prepare',
         '--datasets',args.dataset,'--refresh','--fetch-images','--root',str(args.root),
         '--checkpoint',str(args.checkpoint)]
    if args.flickr30k_root: cmd.extend(['--flickr30k-root',str(args.flickr30k_root)])
    if args.nlvr2_authorized: cmd.append('--nlvr2-authorized')
    subprocess.run(cmd,check=True)
    manifest=json.loads((args.root/args.dataset/'manifest.json').read_text())
    finished=False
    if manifest['status']=='ready':
        cmd=[sys.executable,str(ROOT/'scripts/evaluate_submission_benchmarks.py'),
             '--datasets',args.dataset,'--root',str(args.root),'--output-root',str(args.result_root),
             '--checkpoint',str(args.checkpoint),'--controls','--repair-wrong-controls']
        if args.dataset=='winoground': cmd.remove('--controls'); cmd.remove('--repair-wrong-controls')
        subprocess.run(cmd,check=True)
        status=args.result_root/args.dataset/'status.json'
        finished=status.exists() and json.loads(status.read_text()).get('status')=='complete'
    else: print(f"BLOCKED: {manifest.get('reason')}",flush=True)
    cmd=[sys.executable,str(ROOT/'scripts/generate_submission_report.py'),
         '--data-root',str(args.root),'--result-root',str(args.result_root)]
    for paper in args.paper_root: cmd.extend(['--paper-root',str(paper)])
    subprocess.run(cmd,check=True)
    if not finished: raise SystemExit(2)

if __name__=='__main__': main()
