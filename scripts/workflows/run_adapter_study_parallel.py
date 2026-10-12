"""Whole-split study: batch frozen vision, prefetch CPU features, schedule GPUs.

One backbone cache / backbone-seed training / backbone-dataset evaluation per
worker. Never DDP: training accumulation, epoch order, seeds and loss stay fixed.
The original identity-bound source files are NOT edited by this launcher.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.jev_pair_protocol import sha256, write_json, load_complete_manifest, audit_label_use
from scripts.workflows.adapter_study_acceleration import file_lock, LockedFeaturePool, accelerated_training_cache, FeaturePrefetch


def jobs_for_stage(stage, config, datasets):
    if stage == 'cache': return [(b,) for b in config['backbones']]
    if stage == 'train': return [(b, s) for b in config['backbones'] for s in config['seeds']]
    if stage == 'evaluate': return [(b, d) for b in config['backbones'] for d in datasets]
    raise ValueError(stage)


def training_order(config, output, backbone, seed, n):
    import torch
    directory = output / 'training' / backbone / f'seed_{seed}'
    if (directory / 'result.json').exists(): return [0]  # original trainer only inspects input_dim
    epoch, offset = 1, 0
    resume = directory / 'training_resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        epoch, offset = saved['epoch'], saved['offset']
    order = [0]  # original trainer reads one feature before entering seed loop
    for ep in range(epoch, config['epochs']+1):
        permutation = list(range(n)); random.Random(seed+ep).shuffle(permutation)
        order.extend(permutation[offset if ep == epoch else 0:])
    return order


def worker(args, plan):
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    config = plan['config']; output = ROOT / config['output']
    train_pool = LockedFeaturePool(ROOT / config['feature_cache'] / 'train', plan['train_identity'])
    eval_pool = LockedFeaturePool(ROOT / config['feature_cache'] / 'evaluation', plan['evaluation_identity'])
    from scripts.workflows.adapter_study_cache import resolve_rows
    if args.worker in {'cache', 'train'}:
        manifest, rows, _ = load_complete_manifest(ROOT / config['training_root'] / 'manifest.json', training=True)
        rows = list(resolve_rows(rows, ROOT))
        index = train_pool.root / f'index_{args.backbone}.json'
        if args.worker == 'cache':
            accelerated_training_cache(ROOT, config, rows, args.backbone, train_pool, index, args.device,
                                       args.vision_batch, args.decode_workers)
            return
        payload = json.loads(index.read_text())
        if payload['identity'] != train_pool.identity or [r['uid'] for r in payload['rows']] != [r['uid'] for r in rows]:
            raise ValueError('Training cache does not cover the complete ordered train split')
        from visualjev.models.visual_jev_v3_pipeline import ThreeStageVisualJEV
        import scripts.training.train_adapter_study_matrix as trainer
        source, _ = ThreeStageVisualJEV.from_checkpoint(ROOT / config['source_checkpoint'], map_location='cpu')
        source.eval().requires_grad_(False)
        order = training_order(config, output, args.backbone, args.seed, len(rows))
        prefetch = FeaturePrefetch(train_pool, payload['rows'], order, args.prefetch)
        write_json(output/'training'/args.backbone/f'seed_{args.seed}'/'parallel_execution_context.json',
                   dict(execution=plan['execution'],device=args.device,
                        existing_complete_returned=(output/'training'/args.backbone/f'seed_{args.seed}'/'result.json').exists()))
        previous = trainer.load_feature
        trainer.load_feature = prefetch
        try:
            trainer.train(ROOT, {**config, 'seeds': [args.seed]}, manifest, rows, payload['rows'], train_pool,
                          source, args.backbone, output / 'training', args.device)
        finally:
            trainer.load_feature = previous; prefetch.close()
    else:
        from scripts.evaluation.evaluate_adapter_study_matrix import evaluate
        manifest, rows, _ = load_complete_manifest(ROOT / config['evaluation_root'] / args.dataset / 'manifest.json')
        training = [json.loads((output / 'training' / args.backbone / f'seed_{s}' / 'result.json').read_text())
                    for s in config['seeds']]
        write_json(output/'evaluation'/args.backbone/args.dataset/'parallel_execution_context.json',
                   dict(execution=plan['execution'],device=args.device,
                        existing_complete_returned=(output/'evaluation'/args.backbone/args.dataset/'result.json').exists(),
                        latency_scope='Cached-head timing under parallel execution; not isolated end-to-end latency'))
        evaluate(ROOT, config, args.dataset, manifest, rows, args.backbone, training, eval_pool,
                 output / 'evaluation', args.device)


def schedule(stage, config, datasets, args, plan_path, status_path):
    queue = list(jobs_for_stage(stage, config, datasets)); total = len(queue)
    active = {}; complete = 0; started = time.monotonic()
    measured_rates = {}
    log_root = ROOT / 'logs/adapter-study-parallel'; log_root.mkdir(parents=True, exist_ok=True)
    last_display = 0
    try:
        while queue or active:
            for gpu in args.gpus:
                if gpu in active or not queue: continue
                job = queue.pop(0)
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', stage,
                       '--plan', str(plan_path), '--device', f'cuda:{gpu}', '--backbone', job[0],
                       '--vision-batch', str(args.vision_batch), '--decode-workers', str(args.decode_workers),
                       '--cpu-threads', str(args.cpu_threads), '--prefetch', str(args.prefetch)]
                if stage == 'train': cmd += ['--seed', str(job[1])]
                if stage == 'evaluate': cmd += ['--dataset', job[1]]
                logfile = log_root / (stage + '_' + '_'.join(map(str, job)) + '.log')
                handle = logfile.open('a', buffering=1)
                env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TZ': 'Asia/Shanghai',
                       'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': str(args.cpu_threads),
                       'MKL_NUM_THREADS': str(args.cpu_threads), 'OPENBLAS_NUM_THREADS': str(args.cpu_threads)}
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active[gpu] = (process, handle, job, logfile)
                print(f'GPU{gpu}: {stage} {job} | log={logfile}', flush=True)
            for gpu, (process, handle, job, logfile) in list(active.items()):
                code = process.poll()
                if code is None: continue
                handle.close(); del active[gpu]
                if code: raise RuntimeError(f'{stage} {job} failed (exit={code}); inspect {logfile}; existing artifacts retained')
                complete += 1
                print(f'{stage}: completed {complete}/{total} | {job}', flush=True)
            now = time.monotonic()
            if now-last_display >= 10:
                from scripts.workflows.progress_eta import duration
                output = ROOT / config['output']
                progress = {}
                for gpu, (_, _, job, _) in active.items():
                    path = (ROOT/config['feature_cache']/'train'/f'parallel_cache_{job[0]}.json' if stage=='cache' else
                            output/'training'/job[0]/f'seed_{job[1]}'/'status.json' if stage=='train' else
                            output/'evaluation'/job[0]/job[1]/'status.json')
                    if path.exists():
                        value = json.loads(path.read_text())
                        progress[str(gpu)] = value
                        n, total_rows = value.get('N_complete',0), value.get('N_total',0)
                        rate_key = (gpu, tuple(job))
                        previous = measured_rates.get(rate_key)
                        rate = previous[2] if previous else None
                        if previous and n > previous[0]:
                            recent = (n-previous[0])/(now-previous[1])
                            rate = recent if rate is None else .5*rate+.5*recent
                        measured_rates[rate_key] = (n,now,rate)
                        eta_seconds = (max(0,total_rows-n)/rate if rate else value.get('task_eta_seconds'))
                        eta = duration(eta_seconds)
                        speed = f'{rate:.2f} rows/s' if rate else 'measuring'
                        print(f'  GPU{gpu} {job}: {n:,}/{total_rows:,} | {speed} | current-task ETA {eta}',flush=True)
                state = dict(status='running',phase=stage,completed_jobs=complete,total_jobs=total,
                             active={str(g):dict(job=list(v[2]),pid=v[0].pid,log=str(v[3])) for g,v in active.items()},
                             progress=progress,
                             elapsed_stage_seconds=now-started,
                             eta_note='Job durations vary by candidate count; per-worker logs provide measured row ETA. Not an overall completion estimate.')
                write_json(status_path, state)
                print(f'{stage}: {complete}/{total} completed, {len(active)} GPUs active; per-task progress: {log_root}', flush=True)
                last_display = now
            time.sleep(1)
    finally:
        # Stop only children started by this scheduler; never kill other Python jobs.
        for process, handle, _, _ in active.values():
            if process.poll() is None: process.terminate()
        for process, handle, _, _ in active.values():
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/adapter_study_20261008.json')
    p.add_argument('--gpus', type=int, nargs='+', default=[0])
    p.add_argument('--vision-batch', type=int, default=32)
    p.add_argument('--decode-workers', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--prefetch', type=int, default=8)
    p.add_argument('--worker', choices=['cache','train','evaluate'])
    p.add_argument('--plan', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--backbone'); p.add_argument('--seed', type=int); p.add_argument('--dataset')
    args = p.parse_args()
    if min(args.vision_batch,args.decode_workers,args.cpu_threads,args.prefetch) < 1:
        raise ValueError('Execution batch/worker/thread/prefetch sizes must be positive')
    if args.worker: return worker(args, json.loads(args.plan.read_text()))
    from scripts.workflows.run_adapter_study import read_config
    config = read_config(args.config); output = ROOT / config['output']; output.mkdir(parents=True,exist_ok=True)
    try:
        with file_lock(output / 'study.lock', nonblocking=True):
            run_controller(args, config, output)
    except BlockingIOError:
        raise SystemExit('Original study is still running. Ctrl+C that foreground process once, then retry. Do not run duplicate writers.')


def run_controller(args, config, output):
    status_path = output / 'pipeline_status.json'
    try:
        old = output / 'study_identity.json'
        expected = dict(config=config,config_sha256=sha256(args.config),
                        workflow_sha256=sha256(ROOT / 'scripts/workflows/run_adapter_study.py'))
        if old.exists() and json.loads(old.read_text()) != expected:
            raise ValueError('Original study configuration/code changed; no unsafe reuse')
        if not old.exists(): write_json(old, expected)
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        if len(set(args.gpus)) != len(args.gpus) or any(g < 0 or g >= torch.cuda.device_count() for g in args.gpus):
            raise ValueError('Invalid or duplicate GPU IDs')
        if len({torch.cuda.get_device_name(g) for g in args.gpus}) != 1:
            raise ValueError('Cache reuse requires identical declared GPU models')
        if shutil.disk_usage(ROOT).free / 1024**3 < config['minimum_free_gib']:
            raise RuntimeError('Insufficient full-cache disk capacity; no data reduction')
        from scripts.data.prepare_adapter_study_training import inspect_sources, prepare
        print('Preflight: checking complete train membership and model checksums (CPU/storage stage)',flush=True)
        inventory = json.loads((ROOT / config['training_inventory']).read_text())
        source = inspect_sources(inventory, ROOT)
        for name, entry in inventory['model_files'].items():
            path = ROOT / name
            if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
                raise ValueError('Model transfer mismatch: ' + name)
        prepare(inventory, ROOT, ROOT / config['training_root'])
        manifest, rows, _ = load_complete_manifest(ROOT / config['training_root'] / 'manifest.json', training=True)
        if manifest['N'] != source['N_train']: raise ValueError('Full train coverage differs')
        label_sources = dict(used_label_manifests=[dict(path=manifest['candidate_manifest'],sha256=manifest['candidate_manifest_sha256'],N=manifest['N'],purpose='train')])
        datasets, gaps = {}, {}
        for name in config['datasets']:
            path = ROOT / config['evaluation_root'] / name / 'manifest.json'
            entry = json.loads(path.read_text()) if path.exists() else {}
            if entry.get('status') != 'ready' or not entry.get('complete_split'):
                gaps[name] = entry.get('reason','Missing complete public labeled split'); continue
            m, eval_rows, _ = load_complete_manifest(path)
            audit_label_use(label_sources, eval_rows); datasets[name] = m['N']
        if not datasets: raise ValueError('No complete labeled evaluation available')
        from scripts.evaluation.run_full_before_after import model_identity
        identity = dict(text_model=model_identity(ROOT/config['text_model']),
                        visual_models={b:model_identity(ROOT/config['visual_models'][b]) for b in config['backbones']},
                        token_count=config['token_count'],candidate_batch=config['candidate_batch'],torch_version=torch.__version__,
                        gpu=torch.cuda.get_device_name(args.gpus[0]),encoder_code={n:sha256(ROOT/n) for n in
                        ['visual_jev_v2.py','train_visual_jev_v2.py','jev/hf_vision_backends.py',
                         'scripts/evaluation/cached_candidate_features.py','scripts/workflows/adapter_study_cache.py']})
        plan = dict(config=config,train_identity={**identity,'scope':'train','membership_sha256':manifest['candidate_manifest_sha256']},
                    evaluation_identity={**identity,'scope':'evaluation_only_frozen_encoder_features'},
                    training_N=manifest['N'],datasets=datasets,resource_gaps=gaps,
                    source_checkpoint_sha256=sha256(ROOT/config['source_checkpoint']),
                    execution=dict(gpus=args.gpus,vision_batch=args.vision_batch,decode_workers=args.decode_workers,
                                   cpu_threads=args.cpu_threads,prefetch=args.prefetch,
                                   code_sha256=sha256(__file__),backend_sha256=sha256(ROOT/'scripts/workflows/adapter_study_acceleration.py'),
                                   unchanged=['N','candidate_sets','epochs','seeds','losses','gradient_accumulation','scorer'],
                                   latency_scope='Cached head timing under parallel execution; not isolated end-to-end latency'))
        # Constructor verifies compatibility with existing serial cache bytes.
        LockedFeaturePool(ROOT/config['feature_cache']/'train',plan['train_identity'])
        LockedFeaturePool(ROOT/config['feature_cache']/'evaluation',plan['evaluation_identity'])
        plan_path = output / 'parallel_execution_plan.json'; write_json(plan_path, plan)
        print(json.dumps(plan['execution'],ensure_ascii=False,indent=2),flush=True)
        print(f'Full train N={manifest["N"]}; full evaluation tasks={datasets}; gaps={list(gaps)}',flush=True)
        write_json(output/'resource_gaps.json',gaps)
        for stage in ('cache','train','evaluate'):
            schedule(stage,config,datasets,args,plan_path,status_path)
        from scripts.evaluation.build_adapter_study_tables import build
        summary = build(output/'evaluation',ROOT/config['reports'],config,verify_predictions=True)
        expected = len(config['backbones'])*len(datasets)
        if summary['complete_backbone_dataset_cases'] != expected: raise ValueError('Full accessible evaluation matrix incomplete')
        final = dict(status='accessible_matrix_complete_access_gaps_remain' if gaps else 'complete',
                     full_accessible_matrix_complete=True,all_declared_datasets_complete=not gaps,complete_cases=expected,
                     source_train_N=manifest['N'],resource_gaps=gaps,report_path=str((ROOT/config['reports']).resolve()),
                     execution_plan=str(plan_path.resolve()))
        write_json(status_path,final); print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)
        if gaps: raise SystemExit(2)
    except BaseException as error:
        if isinstance(error,SystemExit) and error.code == 2: raise
        write_json(status_path,dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',reason=str(error),
                                    artifacts_preserved=True,full_accessible_matrix_complete=False))
        raise


if __name__ == '__main__': main()
