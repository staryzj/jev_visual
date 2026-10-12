"""Full native baseline results and exact-UID native -> trained-JEV comparisons."""
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path

from scripts.evaluation.jev_pair_protocol import (
    sha256,write_json,read_jsonl,load_complete_manifest,credit,primary_metric,validated_prediction,
)
from scripts.evaluation.build_adapter_study_tables import export
from scripts.workflows.adapter_study_cache import resolve_rows
from scripts.workflows.progress_eta import ProgressETA


def recover_prefix(path,rows):
    """Retain committed rows and back up only an incomplete final JSON fragment."""
    path=Path(path); records=[]; end=0
    if not path.exists(): return records
    with path.open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n'):
                backup=path.parent/f'uncommitted-native-tail-{time.time_ns()}.bin'; backup.write_bytes(line)
                break
            record=json.loads(line); index=len(records)
            if index>=len(rows) or record['uid']!=rows[index]['uid']: raise ValueError('Native resume ordered UID differs')
            validated_prediction(rows[index],record['native'])
            records.append(record); end=handle.tell()
    if path.stat().st_size!=end:
        with path.open('r+b') as handle: handle.truncate(end)
    return records


def evaluate_native(runtime,config,backbone,dataset,model_files,output,device='cuda:0',batch_size=32,baseline=None):
    """No labels reach the native scorer. Every candidate/row is evaluated."""
    from scripts.evaluation.evaluate_submission_benchmarks import load_image
    import torch,transformers
    manifest,rows,_=load_complete_manifest(Path(runtime)/config['evaluation_root']/dataset/'manifest.json')
    rows=list(resolve_rows(rows,Path(runtime)))
    directory=Path(output)/backbone/dataset; directory.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol='full_native_without_jev_v1',backbone=backbone,dataset=dataset,N=manifest['N'],
                  split=manifest['split'],candidate_manifest_sha256=manifest['candidate_manifest_sha256'],model_files=model_files,
                  evaluator_sha256=sha256(__file__),baseline_sha256=sha256(Path(__file__).resolve().parents[2]/'jev/native_visual_baselines.py'),
                  metric_code_sha256=sha256(Path(__file__).with_name('jev_pair_protocol.py')),
                  torch_version=torch.__version__,transformers_version=transformers.__version__,
                  native_batch_size=batch_size,labels_used_for_inference=False)
    ip=directory/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity: raise ValueError('Native resume protocol/model changed; choose a new output root')
    write_json(ip,identity)
    rp=directory/'result.json'; pp=directory/'predictions.jsonl'
    if rp.exists():
        old=json.loads(rp.read_text())
        if old.get('status')!='complete' or old['identity']!=identity or old['N']!=manifest['N'] or sha256(pp)!=old['predictions_sha256']:
            raise ValueError('Existing complete native result changed')
        return old
    saved=recover_prefix(pp,rows); n=len(saved)
    total=sum(credit(row,r['native']) for row,r in zip(rows,saved))
    windowed=sum(r.get('windowed_candidate_texts',0) for r in saved)
    own=baseline is None
    if own:
        from jev.native_visual_baselines import NativeVisualBaseline
        baseline=NativeVisualBaseline(backbone,Path(runtime)/config['visual_models'][backbone],device,batch_size)
    protocol=baseline.protocol
    write_json(directory/'run_manifest.json',dict(identity=identity,repo=manifest.get('repo'),revision=manifest.get('revision'),
               fingerprint=manifest.get('fingerprint'),N=manifest['N'],split=manifest['split'],is_true_test=manifest['is_true_test'],
               public_labels=True,checkpoint_files_sha256=model_files,sampling='none',native_protocol=protocol))
    progress=ProgressETA({backbone+'/'+dataset:len(rows)},{backbone+'/'+dataset:sum(len(r['candidates']) for r in rows)/len(rows)},
                         initial={backbone+'/'+dataset:n})
    try:
        with pp.open('a',encoding='utf-8') as handle:
            for index in range(n,len(rows)):
                row=rows[index]; image=load_image(row); start=time.perf_counter()
                previous_windows=baseline.windowed_texts
                try:
                    scores=baseline.scores(image,row['question'],row['candidates'])
                    prediction=dict(uid=row['uid'],scores=scores,prediction=max(range(len(scores)),key=scores.__getitem__))
                    if dataset=='winoground':
                        from PIL import Image
                        matrix=[]
                        for path in row['images']:
                            with Image.open(path) as im: matrix.append(baseline.scores(im.convert('RGB'),row['question'],row['candidates']))
                        prediction['scores_2x2']=matrix
                finally: image.close()
                value=credit(row,prediction); total+=value; n=index+1
                windows=baseline.windowed_texts-previous_windows; windowed+=windows
                record=dict(uid=row['uid'],native=prediction,windowed_candidate_texts=windows,elapsed_seconds=time.perf_counter()-start)
                handle.write(json.dumps(record,ensure_ascii=False)+'\n'); handle.flush()
                if baseline.last_check: write_json(directory/'native_cache_numerical_check.json',baseline.last_check)
                if n%10==0 or n==len(rows):
                    measured=progress.update(backbone+'/'+dataset,dict(N_complete=n),force=n==len(rows)) or {}
                    write_json(directory/'status.json',dict(status='running',N_complete=n,N_total=len(rows),pid=os.getpid(),
                                                           **{k:v for k,v in measured.items() if k not in {'N_complete','N_total'}}))
        if n!=manifest['N']: raise ValueError('Native full denominator not completed')
        result=dict(schema='native-visual-full-result-v1',status='complete',backbone=backbone,dataset_name=dataset,N=n,
                    split=manifest['split'],is_true_test=manifest['is_true_test'],public_labels=True,identity=identity,
                    revision=manifest.get('revision'),fingerprint=manifest.get('fingerprint'),no_sampling=True,
                    native_protocol=protocol,native_training='Released pretrained model only; no adapter/JEV training',
                    native_metric=dict(name=primary_metric(dataset),score=total/n),windowed_candidate_texts=windowed,
                    predictions_sha256=sha256(pp))
        write_json(rp,result); write_json(directory/'status.json',dict(status='complete',N_complete=n,N_total=n))
        return result
    finally:
        if own: baseline.close()


def compare_case(runtime,config,backbone,dataset,native_root,adapter_root,comparison_root):
    """Recompute both metrics and improvements from full per-question evidence."""
    manifest,rows,_=load_complete_manifest(Path(runtime)/config['evaluation_root']/dataset/'manifest.json')
    nd=Path(native_root)/backbone/dataset; ad=Path(adapter_root)/backbone/dataset
    native=json.loads((nd/'result.json').read_text()); adapted=json.loads((ad/'result.json').read_text())
    for r,schema in [(native,'native-visual-full-result-v1'),(adapted,'adapter-study-full-result-v1')]:
        if r.get('schema')!=schema or r.get('status')!='complete' or r['N']!=manifest['N'] or r['split']!=manifest['split']:
            raise ValueError('Native/JEV comparison requires both complete identical splits')
        if r['identity']['candidate_manifest_sha256']!=manifest['candidate_manifest_sha256']:
            raise ValueError('Native/JEV candidate/question manifest differs')
        if r['backbone']!=backbone or r['dataset_name']!=dataset: raise ValueError('Comparison backbone/dataset differs')
    for d,r in [(nd,native),(ad,adapted)]:
        if sha256(d/'predictions.jsonl')!=r['predictions_sha256']: raise ValueError('Comparison prediction SHA differs')
    if set(adapted['metrics']['seeds'])!={str(s) for s in config['seeds']}: raise ValueError('Missing trained seed')
    plan_path=Path(runtime)/config['output']/'parallel_execution_plan.json' if 'output' in config else None
    if plan_path and plan_path.exists():
        visual=json.loads(plan_path.read_text())['train_identity']['visual_models'][backbone]
        if visual!=native['identity']['model_files']: raise ValueError('Native and JEV use different pretrained backbone bytes')
    for seed,arms in adapted['identity']['checkpoints'].items():
        for arm,cp in arms.items():
            if sha256(cp['path'])!=cp['sha256']: raise ValueError('Trained JEV checkpoint changed')
    directory=Path(comparison_root)/backbone/dataset; directory.mkdir(parents=True,exist_ok=True)
    sources=[dict(path=str(d/'result.json'),sha256=sha256(d/'result.json'),predictions_sha256=r['predictions_sha256']) for d,r in [(nd,native),(ad,adapted)]]
    completed=directory/'result.json'
    if completed.exists():
        old=json.loads(completed.read_text())
        if old.get('status')=='complete' and old.get('schema')=='native-to-jev-full-comparison-v1' and old.get('source_results')==sources and old.get('comparison_code_sha256')==sha256(__file__):
            if sha256(directory/'paired_changes.jsonl')!=old['paired_changes_sha256']: raise ValueError('Existing paired evidence changed')
            return old
    native_total=0.; n=0
    totals={str(seed):{arm:dict(after=0.,improved=0,worsened=0,unchanged=0) for arm in ('linear_ce','mlp_ce','mlp_invalid')} for seed in config['seeds']}
    paired=directory/'paired_changes.jsonl'; temp=paired.with_suffix('.jsonl.tmp')
    progress=ProgressETA({'paired/'+backbone+'/'+dataset:len(rows)},
                         {'paired/'+backbone+'/'+dataset:sum(len(r['candidates']) for r in rows)/len(rows)})
    try:
        with temp.open('w',encoding='utf-8') as handle:
            for row,before,after in itertools.zip_longest(rows,read_jsonl(nd/'predictions.jsonl'),read_jsonl(ad/'predictions.jsonl')):
                if row is None or before is None or after is None: raise ValueError('Full comparison prediction coverage differs')
                if before['uid']!=row['uid'] or after['uid']!=row['uid']: raise ValueError('Comparison ordered UID differs')
                if set(after['runs'])!=set(totals): raise ValueError('Comparison per-row seed coverage differs')
                b=credit(row,before['native']); native_total+=b; n+=1; changes={}
                for seed,arms in totals.items():
                    if set(after['runs'][seed])!=set(arms): raise ValueError('Comparison per-row arm coverage differs')
                    changes[seed]={}
                    for arm,total in arms.items():
                        prediction=after['runs'][seed][arm]['original']; a=credit(row,prediction)
                        total['after']+=a; delta=a-b
                        state='improved' if delta>1e-12 else 'worsened' if delta< -1e-12 else 'unchanged'
                        total[state]+=1; changes[seed][arm]=dict(after_credit=a,delta=delta,change=state,prediction=prediction['prediction'])
                handle.write(json.dumps(dict(uid=row['uid'],native_credit=b,native_prediction=before['native']['prediction'],changes=changes))+'\n')
                if n%100==0 or n==len(rows):
                    progress.update('paired/'+backbone+'/'+dataset,dict(N_complete=n),force=n==len(rows))
        if n!=manifest['N']: raise ValueError('Comparison full N not completed')
        temp.replace(paired)
    except BaseException:
        # Keep an incomplete join separate; never name it a complete result.
        raise
    before_score=native_total/n
    if abs(before_score-native['native_metric']['score'])>1e-10: raise ValueError('Native aggregate differs from raw evidence')
    values={}
    for seed,arms in totals.items():
        values[seed]={}
        for arm,t in arms.items():
            after_score=t['after']/n
            if abs(after_score-adapted['metrics']['seeds'][seed][arm]['primary'])>1e-10:
                raise ValueError('JEV aggregate differs from raw evidence')
            values[seed][arm]=dict(after=after_score,delta_pp=100*(after_score-before_score),
                                  **{k:t[k] for k in ('improved','worsened','unchanged')})
    result=dict(schema='native-to-jev-full-comparison-v1',status='complete',backbone=backbone,dataset_name=dataset,
                split=manifest['split'],N=n,is_true_test=manifest['is_true_test'],metric=primary_metric(dataset),
                candidate_manifest_sha256=manifest['candidate_manifest_sha256'],native=before_score,seeds=values,
                native_protocol=native['native_protocol'],native_training=native['native_training'],
                adapted_training='Adapters trained on complete A-OKVQA/ScienceQA/IconQA train corpus; same fixed JEV scorer',
                inference_scope='System comparison; released native zero-shot vs trained adapter+fixed JEV. Adapter-only causal contrast remains in matched three-arm study.',
                source_results=sources,
                checkpoint_sha256={s:{a:cp['sha256'] for a,cp in arms.items()} for s,arms in adapted['identity']['checkpoints'].items()},
                checkpoints=adapted['identity']['checkpoints'],scorer_state_sha256=adapted['identity']['scorer_state_sha256'],
                paired_changes_sha256=sha256(paired),comparison_code_sha256=sha256(__file__),sampling='none')
    write_json(directory/'result.json',result); return result


def build_tables(comparison_root,report_root):
    main=[]; raw=[]; sources=[]; compact=[]; scorer_hashes=set()
    for path in sorted(Path(comparison_root).glob('*/*/result.json')):
        r=json.loads(path.read_text())
        if r.get('schema')!='native-to-jev-full-comparison-v1' or r.get('status')!='complete': continue
        scorer_hashes.add(r['scorer_state_sha256'])
        if sha256(path.parent/'paired_changes.jsonl')!=r['paired_changes_sha256']: raise ValueError('Paired evidence changed')
        row=dict(backbone=r['backbone'],dataset=r['dataset_name'],split=r['split'],N=r['N'],metric=r['metric'],native_pct=100*r['native'],
                 native_protocol=r['native_protocol']['name'],training_seeds=len(r['seeds']))
        for arm in ('linear_ce','mlp_ce','mlp_invalid'):
            values=[100*s[arm]['after'] for s in r['seeds'].values()]
            deltas=[s[arm]['delta_pp'] for s in r['seeds'].values()]
            row[arm+'_mean_pct']=statistics.fmean(values)
            row[arm+'_seed_sd_pct']=statistics.stdev(values) if len(values)>1 else '--'
            row[arm+'_delta_pp']=statistics.fmean(deltas)
        main.append(row)
        values=[100*s['mlp_invalid']['after'] for s in r['seeds'].values()]
        compact.append(dict(backbone=r['backbone'],dataset=r['dataset_name'],split=r['split'],N=r['N'],metric=r['metric'],
                            native_pct=100*r['native'],jev_mean_pct=statistics.fmean(values),
                            jev_seed_sd_pct=statistics.stdev(values) if len(values)>1 else '--',
                            delta_pp=statistics.fmean(s['mlp_invalid']['delta_pp'] for s in r['seeds'].values())))
        for seed,arms in r['seeds'].items():
            for arm,value in arms.items():
                raw.append(dict(backbone=r['backbone'],dataset=r['dataset_name'],split=r['split'],N=r['N'],seed=int(seed),arm=arm,
                                native_pct=100*r['native'],jev_pct=100*value['after'],delta_pp=value['delta_pp'],
                                **{k:value[k] for k in ('improved','worsened','unchanged')}))
        sources.append(dict(path=str(path),sha256=sha256(path),paired_changes_sha256=r['paired_changes_sha256']))
    if len(scorer_hashes)>1: raise ValueError('Native/JEV comparison table mixes different JEV scorers')
    report_root=Path(report_root); report_root.mkdir(parents=True,exist_ok=True)
    export(report_root/'native_vs_jev',main); export(report_root/'native_vs_jev_per_seed',raw)
    export(report_root/'native_vs_jev_main',compact)
    write_json(report_root/'native_vs_jev_sources.json',dict(sources=sources,all_signs_retained=True,
               native_seed_note='One fixed released native model; native score is not misrepresented as three independently trained seeds'))
    return dict(complete_cases=len(main),rows=main)
