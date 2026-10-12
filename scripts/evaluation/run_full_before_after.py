"""Complete-split paired native-Qwen / Visual-JEV evaluation, with exact resume.

Native baseline = mean token log-likelihood of each candidate answer given the
image and question, with the released pretrained LM head. This is an explicitly
candidate-constrained baseline, NOT a published leaderboard score. Both sides
use identical ordered UIDs, images, questions and candidate sets. No sampling.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from scripts.data.submission_benchmarks import ROOT, DEFAULT_CHECKPOINT, REGISTRY, read_jsonl, sha256, write_json
from scripts.evaluation.evaluate_submission_benchmarks import load_image, result_metrics
from scripts.training.train_visual_jev_v2 import format_candidate
from visualjev.models.visual_jev_v2 import Qwen3VLFeatureExtractor
from visualjev.models.visual_jev_v3_pipeline import ThreeStageVisualJEV


def require_coverage(rows, predictions):
    expected = [r['uid'] for r in rows]
    if len(set(expected)) != len(expected) or [p['uid'] for p in predictions] != expected:
        raise ValueError('Exact complete ordered UID coverage required')


def paired_changes(rows, before, after):
    require_coverage(rows, before)
    require_coverage(rows, after)
    if rows[0]['dataset'] == 'textvqa':
        from scripts.evaluation.evaluate_submission_benchmarks import vqa_score
        values = [[vqa_score(r['candidates'][p['prediction']], r['reference_answers']) for r, p in zip(rows, side)] for side in [before, after]]
        return dict(N=len(rows), mean_consensus_difference=sum(b-a for a,b in zip(*values))/len(rows),
                    improved=sum(b>a for a,b in zip(*values)), worsened=sum(b<a for a,b in zip(*values)),
                    unchanged=sum(b==a for a,b in zip(*values)))
    if rows[0]['dataset'] in {'sugarcrepe_pp','winoground'}:
        return dict(N=len(rows), note='Structured metric differences; no binary candidate-index gain claim')
    b = [r['label'] is not None and p['prediction'] == r['label'] for r,p in zip(rows,before)]
    a = [r['label'] is not None and p['prediction'] == r['label'] for r,p in zip(rows,after)]
    return dict(N=len(rows), both_correct=sum(x and y for x,y in zip(b,a)),
                native_only_correct=sum(x and not y for x,y in zip(b,a)),
                jev_only_correct=sum(not x and y for x,y in zip(b,a)),
                both_wrong=sum(not x and not y for x,y in zip(b,a)),
                accuracy_difference=sum(int(y)-int(x) for x,y in zip(b,a))/len(rows))


@torch.inference_mode()
def native_scores(full, processor, image, row, candidate_batch):
    # No target labels/reference answers are supplied to the model.
    conversation = [[dict(role='user', content=[dict(type='image',image=image),
                    dict(type='text',text=row['question']+'\nAnswer the question with a short answer.')])]]
    encoded = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True,
                processor_kwargs={'padding':True}, return_dict=True, return_tensors='pt').to(full.device)
    prefix = full(**encoded, use_cache=True, logits_to_keep=1, return_dict=True)
    first = prefix.logits[0,-1].float().log_softmax(-1)
    cache = prefix.past_key_values
    # An empty native option is retained, not filtered; its answer is immediate EOS.
    tokens = [processor.tokenizer.encode(c, add_special_tokens=False) or [processor.tokenizer.eos_token_id] for c in row['candidates']]
    scores = []
    i = 0
    batch = candidate_batch
    while i < len(tokens):
        chunk = tokens[i:i+batch]
        maximum = max(len(t) for t in chunk)
        try:
            sums = torch.stack([first[t[0]] for t in chunk])
            if maximum > 1:
                ids = torch.full((len(chunk),maximum-1), processor.tokenizer.pad_token_id,
                                 dtype=torch.long, device=full.device)
                mask = torch.zeros_like(ids)
                for j,t in enumerate(chunk):
                    if len(t)>1:
                        ids[j,:len(t)-1] = torch.tensor(t[:-1],device=full.device)
                        mask[j,:len(t)-1] = 1
                repeated = copy.deepcopy(cache)
                repeated.batch_repeat_interleave(len(chunk))
                attention = torch.cat([encoded['attention_mask'].expand(len(chunk),-1),mask],dim=1)
                # transformers 5.10 computes cached M-RoPE positions over the whole
                # attention mask; provide only the current suffix positions explicitly.
                positions = (attention.long().cumsum(-1)-1)[:,-ids.shape[1]:]
                positions = positions.unsqueeze(0).expand(3,-1,-1)
                positions = positions + prefix.rope_deltas.repeat_interleave(len(chunk),dim=0).unsqueeze(0)
                suffix = full(input_ids=ids, attention_mask=attention, past_key_values=repeated,
                              position_ids=positions,use_cache=True, logits_to_keep=0, return_dict=True)
                for j,t in enumerate(chunk):
                    if len(t)>1:
                        lp = suffix.logits[j,:len(t)-1].float().log_softmax(-1)
                        sums[j] += lp.gather(1,torch.tensor(t[1:],device=full.device).unsqueeze(1)).sum()
                del suffix, repeated
            scores.extend(float(value)/len(t) for value,t in zip(sums,chunk))
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1,batch//2)
    if not all(math.isfinite(s) for s in scores):
        raise ValueError('nonfinite native scores')
    return scores


@torch.inference_mode()
def jev_scores(extractor, jev, image, row, batch):
    visual = extractor.encode_image_pre_merger(image)
    aligned = jev.alignment(visual)
    scores = []
    i = 0
    while i < len(row['candidates']):
        try:
            chunk = row['candidates'][i:i+batch]
            text = extractor.encode_candidates([format_candidate(row['question'],c) for c in chunk])
            scores.extend(jev.score_aligned(aligned,text).scores.detach().float().cpu().tolist())
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch = max(1,batch//2)
    return scores


def verified_old_native(rows,manifest,weight_identity,root):
    path=root/manifest['dataset_name']
    rp=path/'result.json'
    if not rp.exists():return {}
    r=json.loads(rp.read_text())
    identity=r.get('identity',{})
    if (r.get('status')!='complete' or r.get('protocol')!='full_benchmark_before_after'
        or identity.get('native_model_files')!=weight_identity
        or identity.get('native_score')!='mean answer-token conditional log-likelihood; empty option=immediate EOS'
        or identity.get('candidate_manifest_sha256')!=manifest['candidate_manifest_sha256']):return {}
    records=list(read_jsonl(path/'predictions.jsonl'))
    require_coverage(rows,records)
    if sha256(path/'predictions.jsonl')!=r['predictions_sha256']:raise ValueError('Existing native records hash mismatch')
    return {p['uid']:p for p in records}


def verified_old(rows, manifest, checkpoint_hash, root):
    path = root/manifest['dataset_name']
    result_file = path/'result.json'
    if not result_file.exists():
        return {}
    result = json.loads(result_file.read_text())
    if (result.get('status') != 'complete' or result.get('protocol') != 'full_benchmark'
        or result.get('checkpoint_sha256') != checkpoint_hash
        or result.get('candidate_manifest_sha256') != manifest['candidate_manifest_sha256']):
        return {}
    records = list(read_jsonl(path/'predictions.jsonl'))
    require_coverage(rows,records)
    if result.get('predictions_sha256') != sha256(path/'predictions.jsonl'):
        raise ValueError('historical predictions hash mismatch')
    return {p['uid']:p for p in records}


def model_identity(directory):
    # Include exact local tensor/config/tokenizer file bytes in published provenance.
    files = [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in {'.json','.safetensors','.jinja'}]
    return {p.name:sha256(p) for p in files}


def evaluate(name,args,full,processor,extractor,jev,weight_identity):
    data = args.root/name
    manifest = json.loads((data/'manifest.json').read_text())
    out = args.output_root/name
    if manifest.get('status') != 'ready' or not manifest.get('complete_split'):
        write_json(out/'status.json',dict(status='blocked',reason=manifest.get('reason'),N_total=manifest.get('N')))
        return
    audit_file = data/'image_decode_audit.json'
    if not audit_file.exists():
        from scripts.data.audit_full_images import audit
        audit(data)
    audit = json.loads(audit_file.read_text())
    if audit['status'] != 'complete' or audit['candidate_manifest_sha256'] != manifest['candidate_manifest_sha256']:
        raise ValueError('Full image decode audit must complete before scoring')
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != manifest['checkpoint_sha256'] or sha256(manifest['candidate_manifest']) != manifest['candidate_manifest_sha256']:
        raise ValueError('manifest/checkpoint identity differs')
    rows = list(read_jsonl(manifest['candidate_manifest']))
    if len(rows) != manifest['N']:
        raise ValueError('split count differs')
    identity = dict(protocol='full_benchmark_before_after',N=manifest['N'],checkpoint_sha256=checkpoint_hash,
                    candidate_manifest_sha256=manifest['candidate_manifest_sha256'],
                    native_model_files=weight_identity, native_score='mean answer-token conditional log-likelihood; empty option=immediate EOS',
                    evaluator_sha256=sha256(__file__), seed=args.seed)
    out.mkdir(parents=True,exist_ok=True)
    ip = out/'resume_identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Resume identity differs; use a fresh output root')
    write_json(ip,identity)
    if (out/'result.json').exists():
        result = json.loads((out/'result.json').read_text())
        if result.get('status') == 'complete' and result.get('identity') == identity:
            print(f'{name}: complete paired result reused',flush=True)
            return
    pp = out/'predictions.jsonl'
    saved = list(read_jsonl(pp)) if pp.exists() else []
    if [p['uid'] for p in saved] != [r['uid'] for r in rows[:len(saved)]]:
        raise ValueError('invalid resume UID prefix')
    old = verified_old(rows,manifest,checkpoint_hash,args.existing_jev_root)
    old_native=verified_old_native(rows,manifest,weight_identity,args.existing_native_root)
    write_json(out/'status.json',dict(status='running',N_complete=len(saved),N_total=len(rows),pid=os.getpid()))
    with pp.open('a',encoding='utf-8') as handle:
        for i in range(len(saved),len(rows)):
            row = rows[i]
            start = time.perf_counter()
            image = load_image(row) if row['uid'] not in old or row['uid'] not in old_native else None
            before = old_native[row['uid']]['native_scores'] if row['uid'] in old_native else native_scores(full,processor,image,row,args.candidate_batch)
            after = old[row['uid']]['scores'] if row['uid'] in old else jev_scores(extractor,jev,image,row,args.jev_candidate_batch)
            record = dict(uid=row['uid'],native_scores=before,jev_scores=after,
                          native_prediction=max(range(len(before)),key=before.__getitem__),
                          jev_prediction=max(range(len(after)),key=after.__getitem__),
                          jev_reused=row['uid'] in old,native_reused=row['uid'] in old_native, elapsed_seconds=time.perf_counter()-start)
            if name == 'winoground':
                from PIL import Image
                pairs = []
                for path in row['images']:
                    with Image.open(path) as im:
                        pairs.append(native_scores(full,processor,im.convert('RGB'),row,args.candidate_batch))
                record['native_scores_2x2'] = pairs
                if row['uid'] in old:
                    record['jev_scores_2x2'] = old[row['uid']]['scores_2x2']
                else:
                    pairs = []
                    for path in row['images']:
                        with Image.open(path) as im:
                            pairs.append(jev_scores(extractor,jev,im.convert('RGB'),row,args.jev_candidate_batch))
                    record['jev_scores_2x2'] = pairs
            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
            handle.flush()
            saved.append(record)
            if (i+1)%10 == 0 or i+1 == len(rows):
                write_json(out/'status.json',dict(status='running',N_complete=i+1,N_total=len(rows),pid=os.getpid(),last_uid=row['uid']))
                print(f'{name}: {i+1}/{len(rows)}; last={record["elapsed_seconds"]:.2f}s',flush=True)
    before = [dict(uid=p['uid'],scores=p['native_scores'],prediction=p['native_prediction'],**({'scores_2x2':p['native_scores_2x2']} if name=='winoground' else {})) for p in saved]
    after = [dict(uid=p['uid'],scores=p['jev_scores'],prediction=p['jev_prediction'],**({'scores_2x2':p['jev_scores_2x2']} if name=='winoground' else {})) for p in saved]
    require_coverage(rows,before)
    require_coverage(rows,after)
    result = dict(schema='visual-jev-before-after-v1',status='complete',dataset_name=name,split=manifest['split'],
                  protocol='full_benchmark_before_after',N=len(rows),identity=identity,manifest=manifest,
                  native_metrics=result_metrics(rows,before),jev_metrics=result_metrics(rows,after),
                  paired_changes=paired_changes(rows,before,after),predictions_sha256=sha256(pp),
                  native_reused_source=dict(path=str(args.existing_native_root/name/'result.json'),sha256=sha256(args.existing_native_root/name/'result.json')) if old_native else None,
                  comparison_scope='Same examples/candidates; pretrained native candidate likelihood versus controlled-COCO-trained external Visual-JEV. Not official Visual Jev SFT, not an equal-training causal study.',
                  no_sampling=True)
    write_json(out/'result.json',result)
    write_json(out/'status.json',dict(status='complete',N_complete=len(rows),N_total=len(rows)))
    print(f'{name}: COMPLETE native={result["native_metrics"]} jev={result["jev_metrics"]}',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'data/submission_benchmarks')
    p.add_argument('--output-root',type=Path,default=ROOT/'experiments/results/full_before_after_20261007')
    p.add_argument('--existing-jev-root',type=Path,default=ROOT/'experiments/results/submission_full')
    p.add_argument('--existing-native-root',type=Path,default=ROOT/'experiments/results/full_before_after_20261007_v3')
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--model',type=Path,default=ROOT/'models/Qwen3-VL-4B-Instruct')
    p.add_argument('--datasets',nargs='+',choices=list(REGISTRY),default=['aokvqa','ai2d','scienceqa','vsr','sugarcrepe_pp','iconqa','textvqa','tallyqa','gqa','snli_ve','winoground','nlvr2'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--candidate-batch',type=int,default=32)
    p.add_argument('--jev-candidate-batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=20260928)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    weight_identity = model_identity(args.model)
    processor = AutoProcessor.from_pretrained(args.model,min_pixels=256*28*28,max_pixels=512*28*28)
    processor.tokenizer.padding_side = 'right'
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    full = AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':args.device}).eval()
    full.requires_grad_(False)
    extractor = Qwen3VLFeatureExtractor(SimpleNamespace(processor=processor,backbone=full.model))
    jev,_ = ThreeStageVisualJEV.from_checkpoint(args.checkpoint,map_location='cpu')
    jev = jev.to(args.device).eval()
    queue = dict(status='running',pid=os.getpid(),datasets=args.datasets,failures={},started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(args.output_root/'active_queue.json',queue)
    for name in args.datasets:
        queue['current_dataset'] = name
        write_json(args.output_root/'active_queue.json',queue)
        try:
            evaluate(name,args,full,processor,extractor,jev,weight_identity)
        except Exception as error:
            import traceback
            reason = f'{type(error).__name__}: {error}'
            queue['failures'][name] = reason
            write_json(args.output_root/name/'status.json',dict(status='failed',reason=reason))
            traceback.print_exc()
            torch.cuda.empty_cache()
        # Always update evidence tables, but never turn partial progress into a score.
        from scripts.evaluation.build_full_comparison import build
        build(args.root,args.output_root,ROOT/'reports/full-benchmark-20261007')
    queue.update(status='finished_with_failures' if queue['failures'] else 'finished',finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(args.output_root/'active_queue.json',queue)


if __name__ == '__main__':
    main()
