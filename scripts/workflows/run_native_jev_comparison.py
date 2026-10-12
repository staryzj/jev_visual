"""Wait for existing study, run full native baselines, join native -> JEV tables.

Adds no training and never modifies the running study's identity-bound code.
One complete backbone/dataset per GPU worker. Own lock prevents duplicate runs;
original study lock prevents sharing GPUs with a still-active adapter workflow.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.evaluation.jev_pair_protocol import sha256,write_json
from scripts.workflows.adapter_study_acceleration import file_lock


def acquire_adapter_lock(config,handle,status_path):
    import fcntl
    output=ROOT/config['output']; last=0
    while True:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=json.loads((output/'pipeline_status.json').read_text())
            if not state.get('full_accessible_matrix_complete'):
                raise RuntimeError('Adapter study is not running and its full accessible evaluation is incomplete; finish/resume that workflow first')
            return
        except BlockingIOError:
            state=json.loads((output/'pipeline_status.json').read_text()) if (output/'pipeline_status.json').exists() else {}
            if time.monotonic()-last>30:
                print(f'Waiting without GPU use: adapter study phase={state.get("phase",state.get("status"))}. Do not stop current training.',flush=True)
                write_json(status_path,dict(status='waiting_for_adapter_study',gpu_workers_started=0,adapter_state=state))
                last=time.monotonic()
            time.sleep(10)


def schedule(config,plan,args,output):
    queue=[(b,d) for b in config['backbones'] for d in plan['datasets']]; total=len(queue); active={}; completed=0
    logs=ROOT/'logs/native-vs-jev'; logs.mkdir(parents=True,exist_ok=True); last=0
    try:
        while queue or active:
            for gpu in args.gpus:
                if gpu in active or not queue: continue
                b,d=queue.pop(0); log=logs/f'native_{b}_{d}.log'; handle=log.open('a',buffering=1)
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--plan',str(output/'execution_plan.json'),
                     '--backbone',b,'--dataset',d,'--device',f'cuda:{gpu}','--native-batch',str(args.native_batch),
                     '--cpu-threads',str(args.cpu_threads)]
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                     'OMP_NUM_THREADS':str(args.cpu_threads),'MKL_NUM_THREADS':str(args.cpu_threads),'OPENBLAS_NUM_THREADS':str(args.cpu_threads),'TZ':'Asia/Shanghai'}
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                active[gpu]=(process,handle,b,d,log)
                print(f'GPU{gpu}: native {b}/{d}; full N={plan["datasets"][d]}; log={log}',flush=True)
            for gpu,(process,handle,b,d,log) in list(active.items()):
                code=process.poll()
                if code is None: continue
                handle.close(); del active[gpu]
                if code: raise RuntimeError(f'Native {b}/{d} failed; full data remains pending, no sample replacement. Inspect {log}')
                completed+=1; print(f'Native completed {completed}/{total}: {b}/{d}',flush=True)
            if time.monotonic()-last>10:
                from scripts.workflows.progress_eta import duration
                progress={}
                for gpu,(_,_,b,d,_) in active.items():
                    sp=output/'native'/b/d/'status.json'
                    if sp.exists():
                        state=json.loads(sp.read_text()); progress[str(gpu)]=dict(backbone=b,dataset=d,**state)
                        print(f'GPU{gpu} native {b}/{d}: {state.get("N_complete",0):,}/{state.get("N_total",0):,}; task ETA {duration(state.get("task_eta_seconds"))}',flush=True)
                write_json(output/'pipeline_status.json',dict(status='running',phase='full_native_evaluation_and_paired_join',
                           completed_jobs=completed,total_jobs=total,active_gpus=list(active),progress=progress))
                last=time.monotonic()
            time.sleep(1)
    finally:
        for process,handle,_,_,_ in active.values():
            if process.poll() is None: process.terminate()
        for process,handle,_,_,_ in active.values():
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            handle.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/adapter_study_20261008.json')
    p.add_argument('--gpus',type=int,nargs='+',default=[0]); p.add_argument('--native-batch',type=int,default=32)
    p.add_argument('--cpu-threads',type=int,default=2); p.add_argument('--tables-only',action='store_true')
    p.add_argument('--worker',action='store_true'); p.add_argument('--plan',type=Path)
    p.add_argument('--backbone'); p.add_argument('--dataset'); p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if min(a.native_batch,a.cpu_threads)<1: raise ValueError('Positive execution batch/thread counts required')
    if a.worker:
        import torch
        torch.set_num_threads(a.cpu_threads); torch.set_num_interop_threads(1); torch.cuda.set_device(a.device)
        plan=json.loads(a.plan.read_text()); config=plan['config']
        from scripts.evaluation.native_jev_comparison import evaluate_native
        evaluate_native(ROOT,config,a.backbone,a.dataset,plan['models'][a.backbone],Path(plan['output'])/'native',a.device,a.native_batch)
        return
    config=json.loads(a.config.read_text()); output=ROOT/'experiments/results/native_vs_jev_20261008'; output.mkdir(parents=True,exist_ok=True)
    if config['protocol']!='full_cross_backbone_matched_adapter_study': raise ValueError('Wrong study configuration')
    try:
        with file_lock(output/'native.lock',nonblocking=True):
            study=ROOT/config['output']; study.mkdir(parents=True,exist_ok=True)
            with (study/'study.lock').open('a+') as lock:
                acquire_adapter_lock(config,lock,output/'pipeline_status.json')
                from scripts.evaluation.native_jev_comparison import compare_case,build_tables
                datasets={}; gaps={}
                for name in config['datasets']:
                    path=ROOT/config['evaluation_root']/name/'manifest.json'
                    m=json.loads(path.read_text()) if path.exists() else {}
                    if m.get('status')!='ready' or not m.get('complete_split'):
                        gaps[name]=m.get('reason','No complete public labeled split'); continue
                    datasets[name]=m['N']
                if not datasets: raise ValueError('No ready full evaluation datasets')
                if not a.tables_only:
                    import torch
                    if not torch.cuda.is_available() or len(set(a.gpus))!=len(a.gpus) or any(g<0 or g>=torch.cuda.device_count() for g in a.gpus):
                        raise ValueError('Invalid GPU selection')
                    from scripts.evaluation.run_full_before_after import model_identity
                    models={b:model_identity(ROOT/config['visual_models'][b]) for b in config['backbones']}
                    oldplan=json.loads((study/'parallel_execution_plan.json').read_text())
                    if oldplan['config']!=config: raise ValueError('Adapter training config changed')
                    if oldplan['train_identity']['visual_models']!=models: raise ValueError('Native model bytes differ from the trained study')
                    plan=dict(config=config,config_sha256=sha256(a.config),output=str(output),models=models,datasets=datasets,
                              access_gaps=gaps,code_sha256=sha256(__file__),native_batch=a.native_batch,
                              role='Released native model vs existing trained adapter+same fixed JEV; no new training')
                    previous=output/'execution_plan.json'
                    if previous.exists() and json.loads(previous.read_text())!=plan: raise ValueError('Native comparison plan changed; preserve previous results')
                    write_json(previous,plan); schedule(config,plan,a,output)
                pending=[]
                for b in config['backbones']:
                    for d in datasets:
                        if not (output/'native'/b/d/'result.json').exists(): pending.append(dict(backbone=b,dataset=d,reason='Native full result missing')); continue
                        write_json(output/'pipeline_status.json',dict(status='running',phase='paired_evidence_audit',backbone=b,dataset=d,
                                                                     gpu_workers_started=0))
                        compare_case(ROOT,config,b,d,output/'native',study/'evaluation',output/'comparisons')
                summary=build_tables(output/'comparisons',ROOT/'reports/native_vs_jev_20261008')
                expected=len(config['backbones'])*len(datasets)
                if not pending and summary['complete_cases']!=expected: raise ValueError('Complete native/JEV comparison count differs')
                final=dict(status='incomplete' if pending else 'accessible_comparisons_complete_access_gaps_remain' if gaps else 'complete',
                           full_accessible_comparisons_complete=not pending,complete_cases=summary['complete_cases'],expected_cases=expected,
                           pending=pending,resource_gaps=gaps,reports=str(ROOT/'reports/native_vs_jev_20261008'))
                write_json(output/'pipeline_status.json',final); print(json.dumps(final,indent=2,ensure_ascii=False),flush=True)
                if pending or gaps: raise SystemExit(2)
    except BlockingIOError: raise SystemExit('A native comparison controller is already running; do not launch duplicates')
    except BaseException as error:
        if isinstance(error,SystemExit) and error.code==2: raise
        write_json(output/'pipeline_status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',reason=str(error),artifacts_preserved=True))
        raise


if __name__=='__main__': main()
