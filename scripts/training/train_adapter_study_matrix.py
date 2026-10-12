"""Matched full-row training of linear / nonlinear / nonlinear+blank arms.

Consumes only frozen training features. Training is resumable at optimizer
boundaries; epoch permutations are deterministic and every source row is used.
"""
import copy
import json
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import jev.adapter_study as model_module
from jev.adapter_study import ARMS,fresh_arms
from scripts.evaluation.jev_pair_protocol import sha256,write_json
from scripts.evaluation.run_jev_checkpoint_comparison import state_hash
from scripts.workflows.adapter_study_cache import load_feature
from scripts.workflows.progress_eta import ProgressETA
from visualjev.models.visual_jev_v3 import uniformity_loss


def score(model,visual,text,batch):
    aligned=model.alignment(visual)
    return torch.cat([model.score_aligned(aligned,chunk).scores for chunk in text.split(batch)])


def train(runtime,config,manifest,rows,references,pool,source,backbone,output,device):
    expected=manifest['N']; epochs=config['epochs']; accumulation=config['gradient_accumulation']
    source_hash=state_hash(source.decision); input_dim=load_feature(pool,references[0])['visual'].shape[-1]
    results=[]
    for seed in config['seeds']:
        directory=Path(output)/backbone/f'seed_{seed}'; directory.mkdir(parents=True,exist_ok=True)
        identity=dict(protocol='matched_adapter_architecture_and_regularizer',backbone=backbone,seed=seed,N_train=expected,
                      candidate_sha256=manifest['candidate_manifest_sha256'],cache_identity=pool.identity,
                      scorer_state_sha256=source_hash,epochs=epochs,gradient_accumulation=accumulation,
                      lr=config['learning_rate'],invalid_weight=config['invalid_weight'],
                      code_sha256=sha256(__file__),model_code_sha256=sha256(model_module.__file__))
        ip=directory/'identity.json'
        if ip.exists() and json.loads(ip.read_text())!=identity: raise ValueError('Training identity changed; choose a new study root')
        write_json(ip,identity)
        completed=directory/'result.json'
        if completed.exists():
            value=json.loads(completed.read_text())
            if value['status']!='complete' or value['identity']!=identity: raise ValueError('Invalid old training completion')
            for arm in ARMS:
                if sha256(directory/arm/'model.pt')!=value['checkpoints'][arm]['sha256']: raise ValueError('Completed model changed')
            results.append(value); continue
        models={name:model.to(device) for name,model in fresh_arms(source,input_dim,seed).items()}
        if state_hash(models['mlp_ce'].alignment)!=state_hash(models['mlp_invalid'].alignment): raise ValueError('MLP ablation initialization differs')
        for model in models.values(): model.alignment.train(); model.decision.eval().requires_grad_(False)
        optimizers={arm:torch.optim.AdamW(model.alignment.parameters(),lr=config['learning_rate'],weight_decay=.01) for arm,model in models.items()}
        epoch=1; offset=0; visits=0; updates=0; history=[]; sums={arm:0. for arm in ARMS}
        resume=directory/'training_resume.pt'
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            if saved['identity']!=identity: raise ValueError('Training resume provenance differs')
            epoch,offset,visits,updates,history,sums=(saved[k] for k in ['epoch','offset','visits','updates','history','sums'])
            for arm in ARMS:
                models[arm].load_state_dict(saved['models'][arm]); optimizers[arm].load_state_dict(saved['optimizers'][arm])
                for state in optimizers[arm].state.values():
                    for k,v in state.items():
                        if torch.is_tensor(v) and k!='step': state[k]=v.to(device)
                if state_hash(models[arm].decision)!=source_hash: raise ValueError('Resumed scorer changed')
        progress=ProgressETA({f'{backbone}/seed_{seed}':expected*epochs},{f'{backbone}/seed_{seed}':sum(len(r['candidates']) for r in rows)/len(rows)},initial={f'{backbone}/seed_{seed}':visits})
        started=time.perf_counter()
        def save_resume():
            for model in models.values():
                if state_hash(model.decision)!=source_hash: raise ValueError('Frozen scorer was updated')
            temp=resume.with_suffix('.tmp')
            torch.save(dict(identity=identity,epoch=epoch,offset=offset,visits=visits,updates=updates,history=history,sums=sums,
                       models={a:m.state_dict() for a,m in models.items()},optimizers={a:o.state_dict() for a,o in optimizers.items()}),temp)
            temp.replace(resume)
        while epoch<=epochs:
            order=list(range(expected)); random.Random(seed+epoch).shuffle(order)
            for start in range(offset,expected,accumulation):
                group=order[start:start+accumulation]
                for optimizer in optimizers.values(): optimizer.zero_grad(set_to_none=True)
                for index in group:
                    features=load_feature(pool,references[index]); row=rows[index]
                    visual=features['visual'].to(device).float(); text=features['text'].to(device).float(); blank=features['blank'].to(device).float()
                    for arm,model in models.items():
                        logits=score(model,visual,text,config['head_batch'])
                        loss=F.cross_entropy(logits[None],torch.tensor([row['label']],device=device))
                        if arm=='mlp_invalid' and not row.get('metadata',{}).get('intrinsic_no_image'):
                            loss=loss+config['invalid_weight']*uniformity_loss(score(model,blank,text,config['head_batch']))
                        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                        (loss/len(group)).backward(); sums[arm]+=float(loss.detach())
                    visits+=1
                for arm,optimizer in optimizers.items():
                    torch.nn.utils.clip_grad_norm_(models[arm].alignment.parameters(),1.); optimizer.step()
                updates+=1; offset=start+len(group)
                if updates%25==0 or offset==expected:
                    status=dict(status='training',seed=seed,epoch=epoch,N_complete=visits,N_total=expected*epochs,
                                optimizer_updates_per_arm=updates,full_source_rows_per_epoch=expected)
                    write_json(directory/'status.json',status); progress.update(f'{backbone}/seed_{seed}',status,force=offset==expected)
                if updates%config['resume_every_updates']==0: save_resume()
            history.append(dict(epoch=epoch,N_train_visited=expected,losses={a:v/expected for a,v in sums.items()}))
            epoch+=1; offset=0; sums={arm:0. for arm in ARMS}; save_resume()
        if visits!=expected*epochs or updates!=((expected+accumulation-1)//accumulation)*epochs:
            raise ValueError('Full training visits/update budget not completed')
        checkpoints={}
        for arm,model in models.items():
            target=directory/arm; target.mkdir(exist_ok=True)
            if state_hash(model.decision)!=source_hash: raise ValueError('Scorer tensor identity differs')
            metadata=dict(identity=identity,arm=arm,full_declared_train=True,N_train=expected,source_membership=manifest['source_membership_audit'],
                          train_visits=visits,optimizer_updates=updates,scorer_frozen=True,scorer_state_sha256=source_hash,
                          adapter_parameters=sum(p.numel() for p in model.alignment.parameters()),history=history,
                          selection='Prespecified epochs; no benchmark-label selection/calibration',
                          regularizer='White blank uniformity; intrinsic text-only rows use supervised CE only')
            model.eval(); model.save(target/'model.pt',metadata)
            checkpoints[arm]=dict(path=str((target/'model.pt').resolve()),sha256=sha256(target/'model.pt'),adapter_parameters=metadata['adapter_parameters'])
            write_json(target/'training_provenance.json',{**metadata,'checkpoint_sha256':checkpoints[arm]['sha256']})
        value=dict(status='complete',identity=identity,checkpoints=checkpoints,history=history,N_train=expected,
                   train_visits_per_arm=visits,updates_per_arm=updates,elapsed_this_session_seconds=time.perf_counter()-started)
        write_json(completed,value); write_json(directory/'status.json',dict(status='complete',N_complete=visits,N_total=expected*epochs))
        results.append(value); del models,optimizers; torch.cuda.empty_cache()
    return results
