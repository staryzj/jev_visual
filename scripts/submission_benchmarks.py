#!/usr/bin/env python3
"""Pinned, complete-split Visual-JEV submission data pipeline (no sample caps)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset, load_from_disk, concatenate_datasets, Image as DatasetImage
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'experiments/visual_jev_v3_paper/checkpoints/v3_full.pt'
REGISTRY = {
    'gqa': dict(repo='lmms-lab-encoder/GQA', split='testdev', prefix='testdev_balanced_instructions/', source='https://cs.stanford.edu/people/dorarad/gqa/', true_test=False, note='Complete official balanced test-dev; hidden test is not scored. Fixed train-only answer vocabulary required.'),
    'snli_ve': dict(repo='HuggingFaceM4/SNLI-VE', split='test', file='snli_ve_test.jsonl', source='https://github.com/necla-ml/SNLI-VE', true_test=True, note='Original SNLI-VE, including neutral; separately authorised Flickr30K images required.'),
    'textvqa': dict(repo='lmms-lab-encoder/TextVQA', split='validation', prefix='data/validation-', source='https://textvqa.org/', true_test=False, note='Official validation; test labels hidden. OCR plus an optional train-only vocabulary; never inject reference answers.'),
    'tallyqa': dict(repo='vikhyatk/tallyqa-test', split='test', prefix='data/test-', source='https://github.com/manoja328/TallyQA_dataset', true_test=True, note='All questions across all test image rows. Fixed integer candidates 0..15 (declared task range).'),
    'scienceqa': dict(repo='derek-thomas/ScienceQA', split='test', prefix='data/test-', source='https://github.com/lupantech/ScienceQA', true_test=True, note='Whole official test, including intrinsic text-only items as blank input; image-bearing stratum reported separately. No lectures/solutions.'),
    'aokvqa': dict(repo='HuggingFaceM4/A-OKVQA', split='validation', prefix='data/validation-', source='https://github.com/allenai/aokvqa', true_test=False, note='Complete official validation, native multiple choice; official test has no public labels.'),
    'iconqa': dict(repo='lmms-lab-encoder/ICON-QA', split='test', prefix='data/test-', source='https://iconqa.github.io/', true_test=True, note='Official test select_txt task (all native text-choice items). select_img and fill_in are separate tasks, not silently relabelled.'),
    'ai2d': dict(repo='lmms-lab-encoder/ai2d', split='test', prefix='data/test-', source='https://allenai.org/data/diagrams', true_test=True, note='Complete public labelled test, native options.'),
    'nlvr2': dict(repo='lmms-lab/NLVR2', split='unbalanced_test_public', prefix='data/unbalanced_test_public-', source='https://lil.nlp.cornell.edu/nlvr/', true_test=True, note='Complete official test-P; both images concatenated in left/right order, binary candidates; this adaptation is disclosed.'),
    'vsr': dict(repo='cambridgeltl/vsr_random', split='test', file='test.jsonl', source='https://github.com/cambridgeltl/visual-spatial-reasoning', true_test=True, note='Complete official random test; binary true/false; COCO image references resolved locally or downloaded.'),
    'winoground': dict(repo='facebook/winoground', split='test', prefix='data/test-', source='https://huggingface.co/datasets/facebook/winoground', true_test=True, note='400 groups; official strict text/image/group metrics from all four scores; gated access is never bypassed.'),
    'sugarcrepe_pp': dict(repo='Aman-J/SugarCrepe_pp', split='evaluation (HF storage split=train)', prefix='data/', source='https://github.com/Sri-Harsha/scpp', true_test=False, note='Entire released evaluation suite, all five categories; strict ITT: both positive captions must beat negative. HF train name is storage metadata, not a training split.'),
}
EXPECTED_SOURCE_COUNTS = {'gqa':12578,'snli_ve':17901,'textvqa':5000,
                         'tallyqa':26451,'scienceqa':4241,'aokvqa':1145,
                         'iconqa':21489,'ai2d':3088,'vsr':2195,
                         'winoground':400,'sugarcrepe_pp':4757}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def parse_list(value):
    if isinstance(value, list): return value
    if not value: return []
    try: return json.loads(value)
    except (ValueError, TypeError):
        try: return ast.literal_eval(value)
        except (ValueError, SyntaxError): return [str(value)]

def resolve_local_image(filename, extra=None):
    filename = Path(str(filename)).name
    roots = [extra] if extra else []
    roots += [ROOT/'data/visual-jev-v2/coco/val2017', ROOT/'data/visual-jev-v2/coco/train2017', ROOT/'data/visual-jev-v2/coco/images/val2017', ROOT/'data/visual-jev-v2/coco/images/train2017', ROOT/'data/flickr30k-images']
    for directory in roots:
        if directory and (Path(directory)/filename).is_file(): return str((Path(directory)/filename).resolve())
    return None

def materialize_image(value, out, key):
    if value is None: return None
    if isinstance(value, str):
        return resolve_local_image(value)
    path = out/'images'/f'{key}.png'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        value.convert('RGB').save(path)
    return str(path.resolve())

def pinned_load(name, out, spec, revision):
    dest = out/'dataset'
    if dest.exists() and not (out/'download.json').exists():
        raise ValueError(f'Existing unpinned dataset at {dest}; preserve it and choose a fresh --root')
    if dest.exists() and (out/'download.json').exists():
        previous = json.loads((out/'download.json').read_text())
        if previous.get('revision') == revision: return load_from_disk(str(dest))
        raise ValueError(f'Pinned revision changed at {dest}; preserve existing data and choose a fresh --root')
    info = HfApi().dataset_info(spec['repo'], revision=revision, timeout=20)
    if spec.get('file'):
        files = [spec['file']]; fmt = 'json'
    else:
        files = [f.rfilename for f in info.siblings if f.rfilename.startswith(spec['prefix']) and f.rfilename.endswith(('.parquet', '.json'))]
        fmt = 'parquet' if any(f.endswith('.parquet') for f in files) else 'json'
    if not files: raise RuntimeError(f'No split files match {spec.get("prefix")} at pinned revision')
    local = [hf_hub_download(spec['repo'], f, repo_type='dataset', revision=revision) for f in files]
    if name == 'sugarcrepe_pp':
        parts = []
        for remote, path in zip(files, local):
            payload = json.loads(Path(path).read_text())
            rows = list(payload.values()) if isinstance(payload, dict) else payload
            from datasets import Dataset
            rows = [{**r, 'category':Path(remote).stem} for r in rows]
            parts.append(Dataset.from_list(rows))
        ds = concatenate_datasets(parts)
    else:
        ds = load_dataset(fmt, data_files={spec['split']:local}, split=spec['split'])
    ds.save_to_disk(str(dest))
    write_json(out/'download.json', dict(revision=revision, files=[dict(path=f, sha256=sha256(p)) for f,p in zip(files,local)], N=len(ds), fingerprint=ds._fingerprint))
    return ds

def make_row(name, r, i, out, args, image_index=None, vocab=None):
    images=[]; extra={}; question=r.get('question',r.get('caption',r.get('sentence2','')))
    key=str(r.get('question_id',r.get('id',i)))
    candidates=[]; label=None; answers=[]
    if name in {'scienceqa','aokvqa','ai2d','iconqa'}:
        if name == 'iconqa' and r.get('ques_type') not in {'choose_txt','select_txt'}: return [], 'outside_native_select_txt_task'
        candidates=[str(c) for c in parse_list(r.get('choices',r.get('options')))]
        if name=='iconqa': candidates=[c.strip() for c in r['choices'].split(',')]
        value=r.get('correct_choice_idx',r.get('answer'))
        if value is not None:
            if name=='iconqa': label=candidates.index(str(value))
            elif isinstance(value,str) and value.upper() in ['A','B','C','D','E'] and value not in candidates: label=ord(value.upper())-65
            else: label=int(value)
        if name=='scienceqa' and r.get('hint'): question += '\nContext: '+r['hint']
        image=materialize_image(r.get('query_image',r.get('image')),out,f'{i}')
        images=[image] if image else []
        extra['intrinsic_no_image']=name=='scienceqa' and r.get('image') is None
    elif name=='tallyqa':
        image=materialize_image(r['image'],out,f'{i}'); result=[]
        for j,qa in enumerate(r['qa']):
            cs=[str(n) for n in range(16)]; a=str(qa['answer'])
            result.append(dict(uid=f'{name}:{i}:{j}',dataset=name,images=[image] if image else [],question=qa['question'],candidates=cs,label=cs.index(a) if a in cs else None,reference_answers=[a],metadata={'is_simple':qa['is_simple'],'data_source':qa['data_source'],'candidate_policy':'fixed integers 0..15'}))
        return result,None
    elif name=='nlvr2':
        images=[materialize_image(r.get(k),out,f'{i}_{k}') for k in ['left_image','right_image']]
        candidates=['false','true']; label=candidates.index(str(r['answer']).lower())
        extra['multi_image_policy']='left/right concatenation with 16px neutral separator'
    elif name=='vsr':
        image=(image_index or {}).get(r['image']) or resolve_local_image(r['image'])
        if image is None and args.fetch_images and r['image'] not in (image_index or {}):
            import requests
            path=out/'images'/Path(r['image']).name; path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                response=requests.get(r['image_link'],timeout=15); response.raise_for_status(); path.write_bytes(response.content)
            with Image.open(path) as im: im.verify()
            image=str(path.resolve())
        images=[image] if image else []; candidates=['false','true']; label=int(r['label'])
    elif name=='snli_ve':
        image=resolve_local_image(str(r['Flickr30K_ID'])+'.jpg',args.flickr30k_root)
        images=[image] if image else []; candidates=['entailment','neutral','contradiction']; label=candidates.index(r['gold_label'])
        question='What is the relation of the image to this hypothesis? '+r['sentence2']
    elif name=='gqa':
        image=(image_index or {}).get(str(r['imageId'])); images=[image] if image else []
        candidates=vocab or []; answers=[str(r['answer'])]; label=candidates.index(answers[0]) if answers[0] in candidates else None
        extra['candidate_policy']='fixed train-only answer vocabulary; OOV scored incorrect'
    elif name=='textvqa':
        image=materialize_image(r.get('image'),out,f'{i}'); images=[image] if image else []
        ocr=parse_list(r.get('ocr_tokens',[]))
        # References are intentionally not used to construct choices.
        fixed=(vocab or ['yes','no','none','unknown',*[str(n) for n in range(16)]])
        ocr_phrases=[' '.join(str(t) for t in ocr[j:j+k]) for k in [1,2,3] for j in range(len(ocr)-k+1)]
        candidates=list(dict.fromkeys([str(x).strip() for x in fixed+ocr_phrases if str(x).strip()]))
        answers=[str(a) for a in parse_list(r.get('answers',[]))]
        extra['candidate_policy']='train-only vocabulary or fixed yes/no/none/unknown/0..15, plus source OCR 1..3-grams; no gold insertion'
    elif name=='winoground':
        images=[materialize_image(r[k],out,f'{i}_{k}') for k in ['image_0','image_1']]
        candidates=[r['caption_0'],r['caption_1']]; label=0
        extra['task']='winoground_2x2'
    elif name=='sugarcrepe_pp':
        image=resolve_local_image(r['filename']); images=[image] if image else []
        candidates=[r['caption'],r['caption2'],r['negative_caption']]; label=0
        question='Which description is supported by the image?'; key=f'{r["category"]}:{i}'
        extra.update(task='sugarcrepe_pp_strict',category=r['category'],positive_indices=[0,1])
    return [dict(uid=f'{name}:{key}',dataset=name,images=images,question=question,candidates=candidates,label=label,reference_answers=answers,metadata=extra)],None

def prepare_one(name,args):
    spec=REGISTRY[name]; out=args.root/name; out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {}
    if name=='nlvr2' and not args.nlvr2_authorized:
        manifest={**previous,'dataset_name':name,'repo':spec['repo'],'source':spec['source'],'split':spec['split'],'N':previous.get('N'),'N_source':previous.get('N_source'),'revision':previous.get('revision'),'fingerprint':previous.get('fingerprint'),'is_true_test':True,'public_labels':True,'seed':args.seed,'checkpoint_sha256':sha256(args.checkpoint),'protocol':'full_benchmark','status':'blocked','complete_split':False,'reason':'Official NLVR2 image terms require researcher registration/permission; public mirror does not prove authorization.','license_source':'https://lil.nlp.cornell.edu/nlvr/','followup_command':'.venv/bin/python scripts/submission_benchmarks.py prepare --datasets nlvr2 --refresh --nlvr2-authorized'}
        try:
            info=HfApi().dataset_info(spec['repo'],timeout=20)
            manifest.update(revision=info.sha,gated=info.gated,license=(info.card_data or {}).get('license','see original image terms'))
        except Exception as e: manifest['metadata_access_reason']=f'{type(e).__name__}: {e}'
        manifest.update(schema='visual-jev-benchmark-manifest-v1',sampling='none; complete declared official split/task',note=spec['note'],excluded=previous.get('excluded',{}))
        write_json(out/'manifest.json',manifest); print('nlvr2: blocked (image authorization required)',flush=True); return manifest
    if previous.get('status')=='ready' and not args.refresh:
        if previous.get('checkpoint_sha256')!=sha256(args.checkpoint) or previous.get('seed')!=args.seed:
            raise ValueError(f'{name}: existing manifest uses another checkpoint/seed; refresh deliberately or choose a new root')
        print(f'{name}: ready N={previous["N"]}',flush=True); return previous
    manifest=dict(schema='visual-jev-benchmark-manifest-v1',dataset_name=name,repo=spec['repo'],source=spec['source'],split=spec['split'],N=None,N_source=None,revision=None,fingerprint=None,is_true_test=spec['true_test'],public_labels=True,seed=args.seed,checkpoint_sha256=sha256(args.checkpoint),protocol='full_benchmark',sampling='none; complete declared official split/task',note=spec['note'],status='preparing',excluded={},created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(out/'manifest.json',manifest)
    try:
        info=HfApi().dataset_info(spec['repo'],timeout=20)
        manifest.update(revision=info.sha,gated=info.gated,license=(info.card_data or {}).get('license','not declared by mirror; see original source'))
        # Access is checked by pinned download; gated without authorization fails normally.
        ds=pinned_load(name,out,spec,info.sha)
        for column,feature in ds.features.items():
            if isinstance(feature,dict) and set(feature)=={'bytes','path'}:
                ds=ds.cast_column(column,DatasetImage())
        manifest.update(N_source=len(ds),fingerprint=ds._fingerprint)
        if len(ds)!=EXPECTED_SOURCE_COUNTS.get(name,len(ds)):
            raise ValueError(f'Unexpected official source count for {name}: {len(ds)}; do not score a reduced mirror')
        vocab=None
        vp=args.vocab_root/f'{name}.json'
        if vp.exists():
            payload=json.loads(vp.read_text()); provenance=payload.get('provenance',{})
            if provenance.get('split')!='train': raise ValueError('Answer vocabulary must prove train-only provenance')
            candidates=payload['candidates']; vocab=[str(c) for c in candidates]
            if len(set(vocab))!=len(vocab): raise ValueError('duplicate vocabulary entries')
            manifest['candidate_vocab']=dict(path=str(vp.resolve()),sha256=sha256(vp),provenance=provenance)
        image_index={}
        if name=='gqa':
            img_spec={**spec,'prefix':'testdev_balanced_images/'}
            image_out=out/'source_images'; image_out.mkdir(exist_ok=True)
            ids=pinned_load(name,image_out,img_spec,info.sha)
            for i,r in enumerate(ids):
                key=str(r.get('id',r.get('imageId',i))); image_index[key]=materialize_image(r.get('image'),out,'gqa_'+key)
        if name=='vsr' and args.fetch_images:
            import requests
            urls=dict(zip(ds['image'],ds['image_link'])); failures={}
            def fetch_asset(filename,url):
                existing=resolve_local_image(filename)
                if existing: return filename,existing,None
                path=out/'images'/Path(filename).name; path.parent.mkdir(parents=True,exist_ok=True)
                try:
                    if not path.exists():
                        response=requests.get(url,timeout=15); response.raise_for_status()
                        path.write_bytes(response.content)
                    with Image.open(path) as im: im.verify()
                    return filename,str(path.resolve()),None
                except Exception as e: return filename,None,f'{type(e).__name__}: {e}'
            with ThreadPoolExecutor(max_workers=8) as pool:
                tasks=[pool.submit(fetch_asset,k,v) for k,v in urls.items()]
                for f in as_completed(tasks):
                    filename,image,error=f.result(); image_index[filename]=image
                    if error: failures[filename]=error
            write_json(out/'image_download_failures.json',failures)
        missing=[]; invalid=[]; intrinsic=0; total=0; excluded={}; uids=set()
        target=out/'candidates.jsonl'; temp=out/'candidates.jsonl.tmp'
        with temp.open('w',encoding='utf-8') as f:
            for i,r in enumerate(ds):
                try: rows,reason=make_row(name,r,i,out,args,image_index,vocab)
                except Exception as e:
                    invalid.append(dict(source_index=i,reason=f'{type(e).__name__}: {e}')); continue
                if reason: excluded[reason]=excluded.get(reason,0)+1
                for row in rows:
                    total+=1
                    if row['uid'] in uids: invalid.append(dict(uid=row['uid'],reason='duplicate uid'))
                    uids.add(row['uid'])
                    if row['metadata'].get('intrinsic_no_image'): intrinsic+=1
                    elif not row['images'] or any(not p or not Path(p).is_file() for p in row['images']): missing.append(row['uid'])
                    if len(row['candidates'])<2: invalid.append(dict(uid=row['uid'],reason='insufficient answer candidates'))
                    if name not in {'textvqa','gqa'} and row['label'] is None: invalid.append(dict(uid=row['uid'],reason='hidden/invalid label'))
                    if row['label'] is not None and not 0<=row['label']<len(row['candidates']): invalid.append(dict(uid=row['uid'],reason='label outside candidate range'))
                    if name in {'textvqa','gqa'} and not row['reference_answers']: invalid.append(dict(uid=row['uid'],reason='hidden/missing reference answers'))
                    row['source_split']=spec['split']; f.write(json.dumps(row,ensure_ascii=False)+'\n')
                if (i+1)%1000==0: print(f'{name}: adapted source {i+1}/{len(ds)}',flush=True)
        temp.replace(target)
        if total==0: invalid.append(dict(reason='empty declared task/split'))
        manifest.update(N=total,excluded=excluded,intrinsic_no_image_count=intrinsic,missing_image_count=len(missing),invalid_count=len(invalid),candidate_manifest=str(target.resolve()),candidate_manifest_sha256=sha256(target),complete_split=(total>0 and not missing and not invalid),status='ready' if total>0 and not missing and not invalid else 'blocked',missing_image_examples=missing[:20],invalid_examples=invalid[:20])
        if name=='iconqa': manifest['task']='select_txt'
        if name=='gqa' and not vocab: manifest.update(status='blocked',reason='train-only answer vocabulary required; run build_vocab command')
        elif invalid: manifest['reason']='Invalid/missing candidates or labels; whole dataset blocked, no partial accuracy'
        elif missing: manifest['reason']='Missing image resources; whole declared split blocked, no reduced subset'
        write_json(out/'resource_gaps.json',dict(missing_image_uids=missing,invalid_rows=invalid))
    except Exception as e:
        manifest.update(status='blocked',reason=f'{type(e).__name__}: {str(e)[:1200]}')
    manifest['followup_command']=f'.venv/bin/python scripts/submission_benchmarks.py prepare --datasets {name} --refresh --fetch-images'
    if name=='snli_ve': manifest['followup_command']+=' --flickr30k-root data/flickr30k-images'
    write_json(out/'manifest.json',manifest)
    print(f'{name}: {manifest["status"]} N={manifest["N"]} {manifest.get("reason","")}',flush=True)
    return manifest

def build_vocab(args):
    name=args.dataset
    spec=REGISTRY[name]; rev=HfApi().dataset_info(spec['repo']).sha
    spec={**spec,'split':'train','prefix':'train_balanced_instructions/' if name=='gqa' else 'data/train-'}
    out=args.root/name/'train_vocab_source'; out.mkdir(parents=True,exist_ok=True)
    ds=pinned_load(name,out,spec,rev)
    from collections import Counter
    counts=Counter()
    # All train rows, no evaluation labels; image decode is unnecessary.
    for r in ds.remove_columns([k for k in ds.column_names if 'image' in k.lower()]):
        counts.update([r['answer']] if name=='gqa' else parse_list(r['answers']))
    candidates=[str(k) for k,_ in counts.most_common()]
    if name=='textvqa': candidates=candidates[:5000] # fixed task vocabulary size, never a sample cap
    write_json(args.vocab_root/f'{name}.json',dict(candidates=candidates,provenance=dict(repo=spec['repo'],revision=rev,split='train',N=len(ds),fingerprint=ds._fingerprint,policy='all distinct train answers' if name=='gqa' else '5000 most frequent train answers')))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','list','build-vocab'])
    p.add_argument('--root',type=Path,default=ROOT/'data/submission_benchmarks')
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--datasets',nargs='+',choices=list(REGISTRY),default=list(REGISTRY))
    p.add_argument('--dataset',choices=['gqa','textvqa'],default='gqa')
    p.add_argument('--vocab-root',type=Path,default=ROOT/'data/submission_benchmarks/vocab')
    p.add_argument('--flickr30k-root',type=Path)
    p.add_argument('--seed',type=int,default=20260928)
    p.add_argument('--refresh',action='store_true'); p.add_argument('--fetch-images',action='store_true')
    p.add_argument('--nlvr2-authorized',action='store_true',help='User attests that official NLVR2 image terms/registration are satisfied')
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if args.command=='list': print(json.dumps(REGISTRY,indent=2)); return
    if args.command=='build-vocab': build_vocab(args); return
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(prepare_one,n,args) for n in args.datasets]
        for f in as_completed(futures): results.append(f.result())
    inventory=[json.loads((args.root/name/'manifest.json').read_text()) for name in REGISTRY if (args.root/name/'manifest.json').exists()]
    write_json(args.root/'download_summary.json',dict(protocol='full_benchmark',last_requested_datasets=args.datasets,results=inventory))

if __name__=='__main__': main()
