"""Evaluate all trained seeds/arms together on every row of one full split."""
import itertools
import json
import math
import os
import time
from pathlib import Path
import torch
from jev.adapter_study import ARMS,load
from scripts.evaluation.jev_pair_protocol import credit,primary_metric,sha256,write_json
from scripts.evaluation.run_jev_checkpoint_comparison import state_hash
from scripts.workflows.adapter_study_cache import FrozenEncoders,resolve_rows
from scripts.workflows.progress_eta import ProgressETA
from scripts.training.train_adapter_study_matrix import score


def new_totals(config):
    return {str(s):{a:{c:dict(sum=0.,flip=0,head_ms=0.) for c in ('original','blank','wrong')} for a in ARMS} for s in config['seeds']}


def add_row(row,record,totals,changes,controls):
    if set(record['runs'])!=set(totals): raise ValueError('Incomplete seed coverage for this evaluation UID')
    for seed,arms in record['runs'].items():
        if seed not in totals or set(arms)!=set(ARMS): raise ValueError('Missing seed/arm prediction')
        for arm,value in arms.items():
            if not math.isfinite(value['head_ms']) or value['head_ms']<0: raise ValueError('Invalid measured scoring time')
            for condition in ('original','blank','wrong') if controls else ('original',):
                prediction=value[condition]
                totals[seed][arm][condition]['sum']+=credit(row,prediction)
                if condition!='original': totals[seed][arm][condition]['flip']+=prediction['prediction']!=value['original']['prediction']
            totals[seed][arm]['original']['head_ms']+=value['head_ms']
        before=credit(row,arms['linear_ce']['original']); after=credit(row,arms['mlp_invalid']['original'])
        change=after-before; changes[seed]['improved' if change>1e-12 else 'worsened' if change< -1e-12 else 'unchanged']+=1


def finish(name,manifest,totals,changes,n,controls):
    if n!=manifest['N']: raise ValueError('Full evaluation denominator differs')
    values={}
    for seed,arms in totals.items():
        values[seed]={}
        for arm,conditions in arms.items():
            values[seed][arm]=dict(primary=conditions['original']['sum']/n,
                adapter_scorer_ms_mean=conditions['original']['head_ms']/n)
            if controls:
                values[seed][arm]['visual_dependency']={c:dict(primary=conditions[c]['sum']/n,
                    flip_rate=conditions[c]['flip']/n) for c in ('blank','wrong')}
        values[seed]['paired_changes']=changes[seed]
    return dict(metric=primary_metric(name),seeds=values,N=n)


def evaluate(runtime,config,name,manifest,rows,backbone,training_results,pool,output,device):
    output=Path(output)/backbone/name; output.mkdir(parents=True,exist_ok=True)
    models={}; cp_identity={}; source_hash=None
    for training in training_results:
        seed=str(training['identity']['seed']); models[seed]={}; cp_identity[seed]={}
        for arm,checkpoint in training['checkpoints'].items():
            path=Path(checkpoint['path'])
            if sha256(path)!=checkpoint['sha256']: raise ValueError('Trained adapter checkpoint changed')
            model,payload=load(path,device); model.eval().requires_grad_(False)
            current=state_hash(model.decision)
            if source_hash is None: source_hash=current
            if current!=source_hash: raise ValueError('Different scorers in the backbone/seed/arm matrix')
            models[seed][arm]=model; cp_identity[seed][arm]=checkpoint
    if set(models)!={str(s) for s in config['seeds']}: raise ValueError('Missing full-training seed')
    controls=config['controls'] and name!='winoground'
    identity=dict(candidate_manifest_sha256=manifest['candidate_manifest_sha256'],N=manifest['N'],
                  backbone=backbone,scorer_state_sha256=source_hash,checkpoints=cp_identity,
                  feature_identity=pool.identity,controls=controls,code_sha256=sha256(__file__))
    ip=output/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity: raise ValueError('Evaluation resume identity differs')
    write_json(ip,identity)
    write_json(output/'run_manifest.json',dict(schema='adapter-study-matrix-manifest-v1',
               dataset_name=name,split=manifest['split'],N=manifest['N'],repo=manifest.get('repo'),
               revision=manifest.get('revision'),fingerprint=manifest.get('fingerprint'),public_labels=True,
               is_true_test=manifest['is_true_test'],candidate_manifest_sha256=manifest['candidate_manifest_sha256'],
               checkpoint_sha256={s:{a:r['sha256'] for a,r in arms.items()} for s,arms in cp_identity.items()},
               scorer_state_sha256=source_hash,seeds=config['seeds'],backbone=backbone,sampling='none',
               legacy_source_manifest_checkpoint_sha256=manifest.get('checkpoint_sha256')))
    rp=output/'result.json'; pp=output/'predictions.jsonl'
    if rp.exists():
        previous=json.loads(rp.read_text())
        if previous['status']=='complete' and previous['identity']==identity and sha256(pp)==previous['predictions_sha256']:
            return previous
        raise ValueError('Existing complete evaluation changed; preserve it and choose a new output')
    rows=list(resolve_rows(rows,runtime)); signatures=None
    if controls:
        from scripts.evaluation.evaluate_submission_benchmarks import image_signatures
        signatures=image_signatures(rows)
    totals=new_totals(config); changes={str(s):dict(improved=0,worsened=0,unchanged=0) for s in config['seeds']}; n=0
    # Recover only an uncommitted final fragment, preserving its bytes. Full
    # records are never removed or accepted without ordered UID/score checks.
    if pp.exists():
        end=0
        with pp.open('rb') as handle:
            for line in handle:
                if not line.endswith(b'\n'):
                    backup=output/f'uncommitted-tail-{time.time_ns()}.bin'; backup.write_bytes(line)
                    break
                record=json.loads(line)
                if n>=len(rows) or record['uid']!=rows[n]['uid']: raise ValueError('Resume UID order differs')
                add_row(rows[n],record,totals,changes,controls); n+=1; end=handle.tell()
        if pp.stat().st_size!=end:
            with pp.open('r+b') as handle: handle.truncate(end)
    encoder=None; progress=ProgressETA({backbone+'/'+name:len(rows)},{backbone+'/'+name:sum(len(r['candidates']) for r in rows)/len(rows)},initial={backbone+'/'+name:n})
    def encode_features(row,need_controls):
        nonlocal encoder
        # Cache hits do not need a loaded foundation model. Lazily load only
        # when a required tensor has not yet been computed.
        class LazyEncoder:
            def encode_text(self,r):
                nonlocal encoder
                if encoder is None: encoder=FrozenEncoders(runtime,config,backbone,device)
                return encoder.encode_text(r)
            def encode_image(self,image):
                nonlocal encoder
                if encoder is None: encoder=FrozenEncoders(runtime,config,backbone,device)
                return encoder.encode_image(image)
        return pool.get(row,LazyEncoder(),backbone,controls=need_controls)[0]
    def run_score(model,visual,text,uid):
        if device.startswith('cuda'): torch.cuda.synchronize(device)
        start=time.perf_counter(); logits=score(model,visual.to(device).float(),text.to(device).float(),config['head_batch'])
        if device.startswith('cuda'): torch.cuda.synchronize(device)
        values=logits.detach().float().cpu().tolist()
        if len(values)!=len(text) or not all(math.isfinite(v) for v in values): raise ValueError('Candidate scores missing/nonfinite')
        return dict(uid=uid,scores=values,prediction=max(range(len(values)),key=values.__getitem__)),(time.perf_counter()-start)*1000
    try:
        with torch.inference_mode(),pp.open('a',encoding='utf-8') as handle:
            for i in range(n,len(rows)):
                row=rows[i]; started=time.perf_counter(); features=encode_features(row,controls); wrong=None; wrong_uid=None
                if controls:
                    from scripts.evaluation.evaluate_submission_benchmarks import different_image_index
                    j=different_image_index(i,signatures)
                    if j is None: raise ValueError('No genuinely distinct wrong image; cannot fabricate this condition')
                    other={**row,'images':rows[j]['images'],'metadata':{**row.get('metadata',{}),'intrinsic_no_image':not rows[j]['images']}}
                    wrong=encode_features(other,False)['visual']; wrong_uid=rows[j]['uid']
                pairs=None
                if name=='winoground': pairs=[encode_features({**row,'images':[path]},False)['visual'] for path in row['images']]
                record=dict(uid=row['uid'],runs={},wrong_uid=wrong_uid)
                for seed,arms in models.items():
                    record['runs'][seed]={}
                    for arm,model in arms.items():
                        original,ms=run_score(model,features['visual'],features['text'],row['uid'])
                        if pairs is not None: original['scores_2x2']=[run_score(model,v,features['text'],row['uid'])[0]['scores'] for v in pairs]
                        value=dict(original=original,head_ms=ms)
                        if controls:
                            value['blank']=run_score(model,features['blank'],features['text'],row['uid'])[0]
                            value['wrong']=run_score(model,wrong,features['text'],row['uid'])[0]
                        record['runs'][seed][arm]=value
                add_row(row,record,totals,changes,controls)
                record['elapsed_seconds']=time.perf_counter()-started
                handle.write(json.dumps(record,ensure_ascii=False)+'\n'); handle.flush(); n=i+1
                if n%10==0 or n==len(rows):
                    status=dict(status='running',N_complete=n,N_total=len(rows),backbone=backbone,dataset=name,pid=os.getpid())
                    write_json(output/'status.json',status); progress.update(backbone+'/'+name,status,force=n==len(rows))
        result=dict(schema='adapter-study-full-result-v1',status='complete',dataset_name=name,backbone=backbone,
                    split=manifest['split'],N=n,is_true_test=manifest['is_true_test'],public_labels=True,
                    revision=manifest.get('revision'),fingerprint=manifest.get('fingerprint'),
                    identity=identity,predictions_sha256=sha256(pp),metrics=finish(name,manifest,totals,changes,n,controls),
                    no_sampling=True,training_scope='One fixed adapter per backbone/seed/arm for all evaluation datasets; no dataset-specific refitting',
                    latency_scope='Synchronized cached-feature adapter+scorer time only; excludes encoder/cache/controls',
                    semantic_counterfactual=None,semantic_pair_status='Verified semantic edit pairs not supplied; blank/wrong are separately named diagnostics',
                    OOV_count=sum(r['label'] is None for r in rows) if name not in {'textvqa','winoground','sugarcrepe_pp'} else None)
        write_json(rp,result); write_json(output/'status.json',dict(status='complete',N_complete=n,N_total=n)); return result
    finally:
        if encoder is not None: encoder.close()
        del models; torch.cuda.empty_cache()
