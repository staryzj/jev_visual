#!/usr/bin/env python3
"""Evaluate every row of a complete official split with one frozen checkpoint.

Progress predictions are resumable. A result.json is emitted only after exact
UID coverage is verified. Candidate construction occurs before scoring and
never depends on this checkpoint or evaluation reference answers.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.submission_benchmarks import ROOT, DEFAULT_CHECKPOINT, REGISTRY, read_jsonl, sha256, write_json
from scripts.run_official_visual_jev_matched import metrics
from jev.serving import load_predictor
from train_visual_jev_v2 import format_candidate
from visual_jev_v2 import Qwen3VLFeatureExtractor
from visual_jev_v3_pipeline import ThreeStageVisualJEV

def load_image(row):
    if not row['images']:
        if not row['metadata'].get('intrinsic_no_image'): raise ValueError('missing image')
        return Image.new('RGB',(224,224),'white')
    images=[]
    for path in row['images']:
        with Image.open(path) as im: images.append(im.convert('RGB'))
    if len(images)==1: return images[0]
    h=max(im.height for im in images)
    canvas=Image.new('RGB',(sum(im.width for im in images)+16*(len(images)-1),h),(127,127,127))
    x=0
    for im in images: canvas.paste(im,(x,0)); x+=im.width+16
    return canvas

def normalize_answer(value):
    # Exact official VQA normalizer, when installed; fallback is explicitly marked.
    from scripts.vqa_official_scoring import normalize
    return normalize(str(value))

def vqa_score(prediction, answers):
    answers=[normalize_answer(a) for a in answers]; p=normalize_answer(prediction)
    if not answers: raise ValueError('hidden labels cannot be scored')
    return statistics.fmean(min(sum(p==a for j,a in enumerate(answers) if j!=i)/3,1) for i in range(len(answers)))

def result_metrics(rows, predictions):
    if len(rows)!=len(predictions): raise ValueError('incomplete split')
    for row,prediction in zip(rows,predictions):
        values=[v for scores in prediction['scores_2x2'] for v in scores] if row['dataset']=='winoground' else prediction['scores']
        if not values or not all(math.isfinite(float(v)) for v in values): raise ValueError('non-finite/empty prediction scores')
        if row['dataset']!='winoground' and 'candidates' in row and len(prediction['scores'])!=len(row['candidates']): raise ValueError('candidate/score coverage mismatch')
    if rows[0]['dataset']=='winoground':
        text=[]; image=[]; group=[]
        for p in predictions:
            a,b=p['scores_2x2']
            t=a[0]>a[1] and b[1]>b[0]; im=a[0]>b[0] and b[1]>a[1]
            text.append(t); image.append(im); group.append(t and im)
        return dict(count=len(rows),text_score=statistics.fmean(text),image_score=statistics.fmean(image),group_score=statistics.fmean(group))
    if rows[0]['dataset']=='sugarcrepe_pp':
        good=[p['scores'][0]>p['scores'][2] and p['scores'][1]>p['scores'][2] for p in predictions]
        categories={}
        for r,ok in zip(rows,good): categories.setdefault(r['metadata']['category'],[]).append(ok)
        return dict(count=len(rows),strict_itt_accuracy=statistics.fmean(good),accuracy=statistics.fmean(good),categories={k:dict(count=len(v),strict_itt_accuracy=statistics.fmean(v)) for k,v in categories.items()})
    if rows[0]['dataset']=='textvqa':
        scores=[vqa_score(r['candidates'][p['prediction']],r['reference_answers']) for r,p in zip(rows,predictions)]
        return dict(count=len(rows),vqa_accuracy=statistics.fmean(scores),accuracy=statistics.fmean(scores),metric='official VQA leave-one-out consensus; candidate-constrained adaptation',candidate_oracle_upper_bound=statistics.fmean(max(vqa_score(c,r['reference_answers']) for c in r['candidates']) for r in rows))
    valid=[(r,p) for r,p in zip(rows,predictions) if r['label'] is not None]
    m=metrics([p['scores'] for r,p in valid],[r['label'] for r,p in valid]) if valid else {}
    m['count']=len(rows)
    # OOV targets stay in the full denominator as incorrect; conditional metrics are named.
    if len(valid)!=len(rows):
        m={**{f'in_vocabulary_{k}':v for k,v in m.items() if k!='count'},'count':len(rows),'oov_count':len(rows)-len(valid),'accuracy':sum(p['prediction']==r['label'] for r,p in valid)/len(rows)}
    for flag,label in [(True,'image_bearing'),(False,'intrinsic_text_only')]:
        if rows[0]['dataset']=='scienceqa':
            pairs=[(r,p) for r,p in zip(rows,predictions) if bool(r['images'])==flag]
            m[label]=dict(count=len(pairs),accuracy=sum(r['label']==p['prediction'] for r,p in pairs)/len(pairs))
    if rows[0]['dataset']=='tallyqa':
        for flag,label in [(True,'simple'),(False,'complex')]:
            pairs=[(r,p) for r,p in zip(rows,predictions) if bool(r['metadata']['is_simple'])==flag]
            m[label]=dict(count=len(pairs),accuracy=sum(r['label']==p['prediction'] for r,p in pairs)/len(pairs))
    return m

def valid_complete_result(result,manifest,checkpoint_hash):
    return (result.get('status')=='complete' and result.get('protocol')=='full_benchmark'
            and result.get('checkpoint_sha256')==checkpoint_hash
            and result.get('candidate_manifest_sha256')==manifest['candidate_manifest_sha256']
            and result.get('metrics',{}).get('count')==manifest['N'])

def image_signatures(rows):
    hashes={}
    signatures=[]
    for row in rows:
        values=[]
        for path in row['images']:
            if path not in hashes: hashes[path]=sha256(path)
            values.append(hashes[path])
        signatures.append(tuple(values) or ('intrinsic_blank',))
    return signatures

def different_image_index(index,signatures):
    j=(index+1)%len(signatures)
    while signatures[j]==signatures[index] and j!=index: j=(j+1)%len(signatures)
    return None if j==index else j

@torch.inference_mode()
def evaluate(name,args,extractor,model,checkpoint_hash,payload):
    directory=args.root/name; manifest=json.loads((directory/'manifest.json').read_text())
    out=args.output_root/name; out.mkdir(parents=True,exist_ok=True)
    if manifest.get('status')!='ready' or not manifest.get('complete_split'):
        write_json(out/'status.json',dict(status='blocked',dataset_name=name,reason=manifest.get('reason'),followup_command=manifest.get('followup_command'))); return
    if manifest['checkpoint_sha256']!=checkpoint_hash: raise ValueError('Manifest checkpoint differs: prepare --refresh --checkpoint with the intended file')
    if manifest['seed']!=args.seed: raise ValueError('Manifest seed differs: prepare --refresh --seed with the intended value')
    if sha256(manifest['candidate_manifest'])!=manifest['candidate_manifest_sha256']: raise ValueError('candidate manifest changed')
    result_path=out/'result.json'
    if result_path.exists():
        existing=json.loads(result_path.read_text())
        if (valid_complete_result(existing,manifest,checkpoint_hash)
            and existing.get('backbone')==args.model
            and (not args.controls or existing.get('visual_dependency'))
            and (not args.repair_wrong_controls or existing.get('wrong_control_verified'))):
            print(f'{name}: complete result reused',flush=True); return
    rows=list(read_jsonl(manifest['candidate_manifest'])); expected=[r['uid'] for r in rows]
    if len(rows)!=manifest['N'] or len(set(expected))!=len(expected): raise ValueError('N/UID coverage failure')
    identity=dict(checkpoint_sha256=checkpoint_hash,candidate_manifest_sha256=manifest['candidate_manifest_sha256'],backbone=args.model,controls=args.controls)
    id_path=out/'resume_identity.json'; records_path=out/'predictions.jsonl'
    if id_path.exists() and json.loads(id_path.read_text())!=identity: raise ValueError(f'{name}: resume identity changed; choose a new --output-root')
    write_json(id_path,identity)
    saved=list(read_jsonl(records_path)) if records_path.exists() else []
    if [r['uid'] for r in saved]!=expected[:len(saved)]: raise ValueError('resume sequence mismatch')
    signatures=image_signatures(rows) if args.controls and name!='winoground' else None
    timings=[]; image_cache={}; blank_cache={}
    with records_path.open('a',encoding='utf-8') as f:
        for i in range(len(saved),len(rows)):
            row=rows[i]; start=time.perf_counter(); image=load_image(row)
            image_key=tuple(row['images']) or ('intrinsic_blank',)
            if image_key not in image_cache:
                image_cache.clear(); image_cache[image_key]=extractor.encode_image_pre_merger(image)
            pre=image_cache[image_key]
            text=torch.cat([extractor.encode_candidates([format_candidate(row['question'],c) for c in row['candidates'][j:j+args.candidate_batch]]) for j in range(0,len(row['candidates']),args.candidate_batch)])
            def score(v): return model(v,text).scores.detach().float().cpu().tolist()
            scores=score(pre)
            record=dict(uid=row['uid'],scores=scores,prediction=int(np.argmax(scores)),label=row['label'])
            if name=='winoground':
                with Image.open(row['images'][0]) as im: first=extractor.encode_image_pre_merger(im.convert('RGB'))
                with Image.open(row['images'][1]) as im: second=extractor.encode_image_pre_merger(im.convert('RGB'))
                record['scores_2x2']=[score(first),score(second)]
            if args.controls and name!='winoground':
                bk=image.size
                if bk not in blank_cache:
                    blank_cache.clear(); blank_cache[bk]=extractor.encode_image_pre_merger(Image.new('RGB',bk,'white'))
                record['blank_scores']=score(blank_cache[bk])
                # Stable cyclic shift chooses a genuinely different image, never the same source.
                j=different_image_index(i,signatures)
                if j is not None:
                    wrong=extractor.encode_image_pre_merger(load_image(rows[j])); record['wrong_scores']=score(wrong); record['wrong_uid']=rows[j]['uid']
            torch.cuda.synchronize() if args.device.startswith('cuda') else None
            record['full_decision_ms']=(time.perf_counter()-start)*1000
            f.write(json.dumps(record)+'\n'); f.flush(); saved.append(record)
            if (i+1)%100==0:
                write_json(out/'status.json',dict(status='running',N_complete=i+1,N_total=len(rows),protocol='full_benchmark'))
                print(f'{name}: {i+1}/{len(rows)}',flush=True)
    if [p['uid'] for p in saved]!=expected: raise ValueError('exact full coverage failed')
    repaired=0
    if signatures is not None:
        uid_index={uid:i for i,uid in enumerate(expected)}
        for i,record in enumerate(saved):
            old_index=uid_index.get(record.get('wrong_uid'))
            if old_index is not None and signatures[i]!=signatures[old_index]: continue
            j=different_image_index(i,signatures)
            if j is None: raise ValueError('No distinct wrong image exists; cannot claim wrong-image evaluation')
            row=rows[i]
            text=torch.cat([extractor.encode_candidates([format_candidate(row['question'],c) for c in row['candidates'][k:k+args.candidate_batch]]) for k in range(0,len(row['candidates']),args.candidate_batch)])
            wrong=extractor.encode_image_pre_merger(load_image(rows[j]))
            record['wrong_scores']=model(wrong,text).scores.detach().float().cpu().tolist(); record['wrong_uid']=rows[j]['uid']; repaired+=1
        if repaired:
            import shutil
            backup=out/'predictions_before_wrong_repair.jsonl'
            if not backup.exists(): shutil.copy2(records_path,backup)
            temp=out/'predictions.jsonl.tmp'
            with temp.open('w',encoding='utf-8') as f:
                for record in saved: f.write(json.dumps(record)+'\n')
            temp.replace(records_path)
    result=dict(schema='visual-jev-result-v1',status='complete',protocol='full_benchmark',dataset_name=name,split=manifest['split'],is_true_test=manifest['is_true_test'],public_labels=True,N=len(rows),checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=checkpoint_hash,checkpoint_metadata=payload.get('metadata',{}),seed=args.seed,revision=manifest['revision'],fingerprint=manifest['fingerprint'],candidate_manifest_sha256=manifest['candidate_manifest_sha256'],predictions_sha256=sha256(records_path),manifest=manifest,metrics=result_metrics(rows,saved),trainable_parameters=model.trainable_parameter_count,backbone=args.model,latency_boundary='full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark',latency_ms=dict(mean=statistics.fmean(p['full_decision_ms'] for p in saved)),comparison_scope='single fixed checkpoint; native multiple-choice or explicitly named candidate-constrained task; no evaluation tuning')
    if args.controls:
        controls={}
        for kind in ['blank','wrong']:
            if all(kind+'_scores' in p for p in saved):
                ps=[{**p,'scores':p[kind+'_scores'],'prediction':int(np.argmax(p[kind+'_scores']))} for p in saved]
                controls[kind]={**result_metrics(rows,ps),'flip_rate':statistics.fmean(p['prediction']!=q['prediction'] for p,q in zip(saved,ps))}
        result['visual_dependency']=dict(original=result['metrics'],**controls,semantic_counterfactual=None,pair_both=None,flip=None,reason='Verified semantic counterfactual pairs not provided for this official split; generic wrong-image shifts are diagnostic.')
        result['wrong_control_verified']=signatures is not None
        result['wrong_control_repair_count']=repaired
        result['wrong_image_policy']='cyclic shift to a different image SHA256, not merely a different filename'
    write_json(result_path,result); write_json(out/'status.json',dict(status='complete',N_complete=len(rows),N_total=len(rows)))
    print(f'{name}: COMPLETE {result["metrics"]}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'data/submission_benchmarks')
    p.add_argument('--output-root',type=Path,default=ROOT/'experiments/results/submission_full')
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--datasets',nargs='+',choices=list(REGISTRY),default=['ai2d','aokvqa','scienceqa','vsr','sugarcrepe_pp','iconqa','nlvr2','tallyqa','snli_ve','winoground','textvqa','gqa'])
    p.add_argument('--model',default=str(ROOT/'models/Qwen3-VL-4B-Instruct')); p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=20260928); p.add_argument('--candidate-batch',type=int,default=8); p.add_argument('--controls',action='store_true')
    p.add_argument('--repair-wrong-controls',action='store_true',help='Verify and repair duplicate-content wrong-image controls in existing complete predictions')
    p.add_argument('--paper-root',type=Path,action='append',default=[],help='Publish audited table/report snapshots to this existing paper project after each dataset')
    p.add_argument('--log-file',type=Path,help='Record the externally redirected log location in active_queue.json')
    args=p.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    checkpoint_hash=sha256(args.checkpoint)
    ready=[n for n in args.datasets if (args.root/n/'manifest.json').exists() and json.loads((args.root/n/'manifest.json').read_text()).get('status')=='ready']
    if not ready: raise SystemExit('No complete ready split. Run preparation first.')
    import os
    queue=dict(status='running',pid=os.getpid(),started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),datasets=args.datasets,
               checkpoint_sha256=checkpoint_hash,evaluator_sha256=sha256(__file__),log_file=str(args.log_file.resolve()) if args.log_file else None)
    write_json(args.output_root/'active_queue.json',queue)
    for name in ready[1:]:
        out=args.output_root/name
        if not (out/'result.json').exists() and not (out/'predictions.jsonl').exists():
            manifest=json.loads((args.root/name/'manifest.json').read_text())
            write_json(out/'status.json',dict(status='queued',N_complete=0,N_total=manifest['N'],protocol='full_benchmark'))
    predictor=load_predictor(model_id=args.model,device=args.device,max_length=512,batch_size=args.candidate_batch,vision=True,image_root=ROOT)
    extractor=Qwen3VLFeatureExtractor(predictor.scorer.model)
    model,payload=ThreeStageVisualJEV.from_checkpoint(args.checkpoint,map_location='cpu'); model=model.to(args.device).eval()
    for name in args.datasets:
        if not (args.root/name/'manifest.json').exists(): continue
        queue['current_dataset']=name; write_json(args.output_root/'active_queue.json',queue)
        try: evaluate(name,args,extractor,model,checkpoint_hash,payload)
        except Exception as e:
            write_json(args.output_root/name/'status.json',dict(status='failed',reason=f'{type(e).__name__}: {e}',protocol='full_benchmark'))
            print(f'{name}: FAILED {type(e).__name__}: {e}',flush=True)
            if isinstance(e,torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
        if args.paper_root:
            import subprocess
            command=[sys.executable,str(ROOT/'scripts/generate_submission_report.py'),
                     '--data-root',str(args.root),'--result-root',str(args.output_root)]
            for paper in args.paper_root: command.extend(['--paper-root',str(paper)])
            subprocess.run(command,check=True)
    queue['status']='finished'; queue['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    write_json(args.output_root/'active_queue.json',queue)

if __name__=='__main__': main()
