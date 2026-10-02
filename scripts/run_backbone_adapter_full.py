"""Run every matching row of each declared custom manifest; never truncate it.

This is full CUSTOM-manifest adapter transfer, not an official benchmark test.
It launches the existing adapter-only runner with the exact manifest counts.
Large training is explicit and never silently replaced by the old 80-row pilot.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',choices=['siglip2','internvl','llava'],required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--manifest-root',type=Path,default=ROOT/'data/benchmark_v1_full/manifests')
    p.add_argument('--qwen-feature-root',type=Path,default=ROOT/'experiments/benchmark_v1_full/features')
    p.add_argument('--dataset-substr',default='coco')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=20260928)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    counts,identity={},{}
    for split in ['train','validation','calibration','test']:
        path=a.manifest_root/f'{split}.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        selected=[r for r in rows if a.dataset_substr.lower() in r['dataset'].lower()]
        if not selected: raise ValueError(f'No matching rows in {split}')
        counts[split]=len(selected)
        identity[split]={'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'N':len(selected),'ids':[r['id'] for r in selected]}
    command=[sys.executable,str(ROOT/'scripts/run_backbone_adapter_fast.py'),'--backbone',a.backbone,'--model',str(a.model.resolve()),'--base-checkpoint',str(a.base_checkpoint.resolve()),'--manifest-root',str(a.manifest_root.resolve()),'--qwen-feature-root',str(a.qwen_feature_root.resolve()),'--dataset-substr',a.dataset_substr,'--output',str(a.output.resolve()),'--device',a.device,'--seed',str(a.seed),'--epochs',str(a.epochs)]
    for split,n in counts.items():command.extend([f'--{split}-limit',str(n)])
    print(json.dumps({'protocol':'full_custom_manifest_adapter_transfer','counts':counts,'command':command,'official_test':False},indent=2))
    if a.plan_only:return
    if a.output.exists(): raise FileExistsError('Use a fresh output directory; no stale cache or result overwrite is permitted.')
    for split in counts:
        shards=a.qwen_feature_root/f'{split}-shards'
        rows=[json.loads(line) for line in (a.manifest_root/f'{split}.jsonl').read_text().splitlines() if line.strip()]
        missing=[i for i,r in enumerate(rows) if a.dataset_substr.lower() in r['dataset'].lower() and not (shards/f'{i:06d}.pt').is_file()]
        if missing:raise FileNotFoundError(f'{split}: {len(missing)} Qwen text-feature shards missing; rebuild features, do not reduce coverage.')
    subprocess.run(command,cwd=ROOT,check=True)
    (a.output/'full_protocol_identity.json').write_text(json.dumps({'protocol':'full_custom_manifest_adapter_transfer','official_test':False,'membership':identity},indent=2)+'\n')

if __name__=='__main__':main()
