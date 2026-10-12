"""One server entry: complete train preparation -> cache -> matched training -> full evaluation -> tables."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.evaluation.jev_pair_protocol import audit_label_use,load_complete_manifest,sha256,write_json


def read_config(path):
    config=json.loads(Path(path).read_text())
    if config['protocol']!='full_cross_backbone_matched_adapter_study' or min(config[k] for k in ['epochs','gradient_accumulation','candidate_batch','head_batch','token_count','resume_every_updates'])<1:
        raise ValueError('Invalid full-study configuration')
    if len(set(config['seeds']))!=len(config['seeds']) or len(set(config['backbones']))!=len(config['backbones']): raise ValueError('Duplicate seed/backbone')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,default=ROOT/'configs/adapter_study_20261008.json')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--check-only',action='store_true'); a=p.parse_args()
    config=read_config(a.config); output=ROOT/config['output']; output.mkdir(parents=True,exist_ok=True)
    # OS lock releases on crashes, making automatic resume safe on the Linux server.
    import fcntl
    lock=(output/'study.lock').open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('This study already has a running worker; no duplicate GPU load')
    identity=dict(config=config,config_sha256=sha256(a.config),workflow_sha256=sha256(__file__))
    ip=output/'study_identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity: raise ValueError('Study configuration changed; choose a new output/cache root')
    write_json(ip,identity)
    status_path=output/'pipeline_status.json'
    def status(phase,**extra): write_json(status_path,dict(status='running',phase=phase,pid=os.getpid(),**extra))
    try:
        inventory=json.loads((ROOT/config['training_inventory']).read_text())
        from scripts.data.prepare_adapter_study_training import inspect_sources,prepare
        status('checking_train_sources'); source=inspect_sources(inventory,ROOT)
        model_inventory=inventory['model_files']
        for name,record in model_inventory.items():
            path=ROOT/name
            if not path.is_file() or path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']: raise ValueError('Target visual model transfer mismatch: '+name)
        if not (ROOT/config['source_checkpoint']).is_file() or not (ROOT/config['text_model']).is_dir(): raise FileNotFoundError('Extract the original H100 model bundle first')
        inputs={}; gaps={}
        for name in config['datasets']:
            path=ROOT/config['evaluation_root']/name/'manifest.json'
            if not path.is_file(): gaps[name]='Missing full evaluation manifest'; continue
            manifest=json.loads(path.read_text())
            if manifest.get('status')!='ready' or not manifest.get('complete_split'):
                gaps[name]=manifest.get('reason','Required resources incomplete'); continue
            inputs[name]=load_complete_manifest(path)[:2]
        if not inputs: raise ValueError('No complete evaluation split available')
        import torch
        if not a.check_only and not torch.cuda.is_available(): raise RuntimeError('CUDA-enabled PyTorch environment required; no silent CPU fallback')
        free=shutil.disk_usage(ROOT).free/1024**3
        plan=dict(status='checked_not_trained',training_N=source['N_train'],backbones=config['backbones'],seeds=config['seeds'],arms=['linear_ce','mlp_ce','mlp_invalid'],
                  checkpoints_planned=len(config['backbones'])*len(config['seeds'])*3,
                  training_visits_per_arm=source['N_train']*config['epochs'],
                  full_evaluation_tasks={n:m['N'] for n,(m,_) in inputs.items()},evaluation_access_gaps=gaps,free_disk_gib=free,
                  source_checkpoint_sha256=sha256(ROOT/config['source_checkpoint']),
                  training_scope='Whole A-OKVQA + ScienceQA + all three IconQA TRAIN tasks; each trained adapter is frozen across all declared full evaluations')
        write_json(output/'execution_plan.json',plan); write_json(output/'resource_gaps.json',gaps)
        print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
        if a.check_only:
            write_json(status_path,{**plan,'phase':'input_check_complete'})
            return
        if free<config['minimum_free_gib']: raise RuntimeError(f'Full feature caches need planned disk capacity: {free:.1f} GiB free; configured minimum {config["minimum_free_gib"]}. Data is not reduced.')
        status('materializing_whole_train'); manifest=prepare(inventory,ROOT,ROOT/config['training_root'])
        manifest,rows,_=load_complete_manifest(ROOT/config['training_root']/'manifest.json',training=True)
        source_labels=dict(used_label_manifests=[dict(path=manifest['candidate_manifest'],sha256=manifest['candidate_manifest_sha256'],N=manifest['N'],purpose='train')])
        for _,eval_rows in inputs.values(): audit_label_use(source_labels,eval_rows)
        from scripts.workflows.adapter_study_cache import FeaturePool,build_training_cache,resolve_rows
        from scripts.training.train_adapter_study_matrix import train
        from scripts.evaluation.evaluate_adapter_study_matrix import evaluate
        from scripts.evaluation.build_adapter_study_tables import build
        from scripts.evaluation.run_full_before_after import model_identity
        from visualjev.models.visual_jev_v3_pipeline import ThreeStageVisualJEV
        rows=list(resolve_rows(rows,ROOT)); source_model,_=ThreeStageVisualJEV.from_checkpoint(ROOT/config['source_checkpoint'],map_location='cpu')
        source_model.eval().requires_grad_(False)
        cache_identity=dict(text_model=model_identity(ROOT/config['text_model']),visual_models={b:model_identity(ROOT/config['visual_models'][b]) for b in config['backbones']},
                            token_count=config['token_count'],candidate_batch=config['candidate_batch'],torch_version=torch.__version__,
                            gpu=torch.cuda.get_device_name(a.device),encoder_code={n:sha256(ROOT/n) for n in ['visual_jev_v2.py','train_visual_jev_v2.py','jev/hf_vision_backends.py','scripts/evaluation/cached_candidate_features.py','scripts/workflows/adapter_study_cache.py']})
        train_pool=FeaturePool(ROOT/config['feature_cache']/'train',{**cache_identity,'scope':'train','membership_sha256':manifest['candidate_manifest_sha256']})
        eval_pool=FeaturePool(ROOT/config['feature_cache']/'evaluation',{**cache_identity,'scope':'evaluation_only_frozen_encoder_features'})
        for backbone in config['backbones']:
            status('cache_whole_train',backbone=backbone)
            refs=build_training_cache(ROOT,config,rows,backbone,train_pool,train_pool.root/f'index_{backbone}.json',a.device)
            status('train_three_arms_all_seeds',backbone=backbone)
            training=train(ROOT,config,manifest,rows,refs,train_pool,source_model,backbone,output/'training',a.device)
            for name,(m,eval_rows) in inputs.items():
                status('full_split_evaluation',backbone=backbone,dataset=name)
                evaluate(ROOT,config,name,m,eval_rows,backbone,training,eval_pool,output/'evaluation',a.device)
                build(output/'evaluation',ROOT/config['reports'],config)
        status('final_prediction_hash_audit')
        summary=build(output/'evaluation',ROOT/config['reports'],config,verify_predictions=True)
        expected=len(config['backbones'])*len(inputs)
        if summary['complete_backbone_dataset_cases']!=expected: raise ValueError('Full accessible matrix incomplete')
        final=dict(status='accessible_matrix_complete_access_gaps_remain' if gaps else 'complete',full_accessible_matrix_complete=True,
                   all_declared_datasets_complete=not gaps,complete_cases=expected,source_train_N=manifest['N'],
                   resource_gaps=gaps,report_path=str((ROOT/config['reports']).resolve()))
        write_json(status_path,final); print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)
        if gaps: raise SystemExit(2)
    except Exception as error:
        write_json(status_path,dict(status='failed',reason=f'{type(error).__name__}: {error}',full_matrix_complete=False)); raise
    finally: lock.close()


if __name__=='__main__': main()
