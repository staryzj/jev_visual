"""Audit recorded adapter-only runs without training or changing original results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/backbone-adaptation-v6/audit.json')
    args = parser.parse_args()
    base = ROOT/'experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt'
    payload = torch.load(base, map_location='cpu', weights_only=True)
    base_hash = sha(base)
    decision = {k: v for k,v in payload['state_dict'].items() if k.startswith('decision.')}
    digest = hashlib.sha256()
    for k,v in sorted(decision.items()):
        digest.update(k.encode()); digest.update(str(v.dtype).encode()); digest.update(str(tuple(v.shape)).encode())
        digest.update(v.contiguous().numpy().tobytes())
    rows = []
    for name in ('siglip2','internvl','llava'):
        directory = ROOT/'experiments/results/backbone_generality'/name
        result = json.loads((directory/'results.json').read_text())
        adapter_path = directory/'adapter.pt'
        adapter = torch.load(adapter_path, map_location='cpu', weights_only=True)
        assert result['status']=='success' and result['fixed_decision_head']
        assert result['base_checkpoint_sha256']==adapter['fixed_decision_checkpoint_sha256']==base_hash
        assert result['checkpoint_sha256']==sha(adapter_path)
        assert sum(v.numel() for v in adapter['state_dict'].values())==result['adapter_trainable_parameters']
        assert not any(k.startswith('decision.') for k in adapter['state_dict'])
        rows.append({'backbone':name,'result_path':str(directory/'results.json'),'result_sha256':sha(directory/'results.json'), 'base_sha256':base_hash,'adapter_sha256':sha(adapter_path),'adapter_parameters':result['adapter_trainable_parameters'],'alignment_config':adapter['alignment_config'],'counts':result['split_counts']})
    manifests = {}
    selected = {}
    for split,limit in {'train':512,'validation':80,'calibration':80,'test':80}.items():
        path=ROOT/'data/benchmark_v1_full/manifests'/f'{split}.jsonl'
        candidates=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows_split=[row for row in candidates if 'coco' in row['dataset'].lower()][:limit]
        assert len(rows_split)==limit
        manifests[split]={'file_sha256':sha(path),'selected_ids':[r['id'] for r in rows_split],'candidate_counts':sorted({len(r['candidates']) for r in rows_split}),'selection':'first matching rows in current manifest order; historical subset, NOT full official split'}
        selected[split]={r['id'] for r in rows_split}
    overlaps={f'{left}:{right}':len(selected[left]&selected[right]) for i,left in enumerate(selected) for right in list(selected)[i+1:]}
    audit={'status':'verified_record_identity','base_checkpoint':str(base),'base_checkpoint_sha256':base_hash,'decision_state_sha256':digest.hexdigest(),'decision_parameters':sum(v.numel() for v in decision.values()),'source_alignment_config':payload['alignment_config'],'rows':rows,'current_manifest_reconstruction':manifests,'cross_split_id_overlap':overlaps,'training_source_sha256':sha(ROOT/'scripts/run_backbone_adapter_fast.py'),'training_source_checks':{'decision_requires_grad_false':True,'optimizer_adapter_only':True,'source':'manually checked run_backbone_adapter_fast.py'},'limitations':['Historical run did not save prediction-level IDs or original manifest hashes. Current manifest reconstruction is not proof of historical cache identity.','Shared head identity is supported by identical base-checkpoint hashes, adapter metadata and the frozen-head training implementation; no post-training full scorer snapshot was saved.','Three target runs are controlled subsets. Qwen has no recorded comparable 80-example result here.','No trained linear, random-adapter or zero-aligned-token result was recorded for this exact backbone protocol. Blank-image features are not zero aligned tokens.']}
    audit['source_adapter_parameters']=sum(v.numel() for k,v in payload['state_dict'].items() if k.startswith('alignment.'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['status','base_checkpoint_sha256','decision_state_sha256','decision_parameters','source_alignment_config','cross_split_id_overlap']},indent=2))

if __name__=='__main__':
    main()
