"""Materialize entire original A-OKVQA/ScienceQA/IconQA TRAIN sources, offline.

No old custom train files, evaluation labels, first-N caps or missing-image
filtering. Inputs are hash-pinned original downloaded parquet/author ZIP.
"""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.evaluation.jev_pair_protocol import sha256, write_json

COUNTS = {'aokvqa': 17056, 'scienceqa': 12726, 'choose_txt': 18946, 'choose_img': 34603, 'fill_in_blank': 10913}


def icon_membership(problems, splits, expected=COUNTS):
    ids = []; heldout = set()
    for task in ('choose_txt', 'choose_img', 'fill_in_blank'):
        values = splits[task+'_train']
        if len(values) != expected[task] or len(set(values)) != len(values):
            raise ValueError('Incomplete author train membership: '+task)
        for split in ('val', 'test'):
            heldout.update(splits[task+'_'+split])
        for pid in values:
            row = problems[pid]
            if row['split'] != 'train' or row['ques_type'] != task:
                raise ValueError('Original held-out item in training')
            ids.append((task,pid))
    if len({pid for _,pid in ids}) != len(ids) or {pid for _,pid in ids}&heldout:
        raise ValueError('Author train/held-out membership overlap')
    return ids


def inspect_sources(inventory, root):
    import pyarrow.parquet as pq
    records = {}
    for name, entry in inventory['sources'].items():
        path = root/entry['path']
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError('Source missing/changed: '+str(path))
        records[name] = dict(path=str(path), sha256=entry['sha256'], bytes=path.stat().st_size)
    for dataset in ('scienceqa','aokvqa'):
        names = [n for n in records if n.startswith(dataset+'_train')]
        n = sum(pq.read_metadata(records[k]['path']).num_rows for k in names)
        if n != COUNTS[dataset]:
            raise ValueError(f'Incomplete {dataset} original train: {n}')
        records[dataset] = dict(N_train=n, source_files=names)
    path = records['iconqa_zip']['path']
    with zipfile.ZipFile(path) as archive:
        problems = json.loads(archive.read('iconqa_data/problems.json'))
        splits = json.loads(archive.read('iconqa_data/pid_splits.json'))
        ids = icon_membership(problems,splits)
        for task,pid in ids:
            row = problems[pid]; prefix=f'iconqa_data/iconqa/train/{task}/{pid}/'
            for member in ['image.png', *(row['choices'] if task=='choose_img' else [])]:
                info = archive.getinfo(prefix+member)
                if info.is_dir() or info.file_size < 1:
                    raise ValueError('Missing original training image')
    records['iconqa'] = dict(N_train=len(ids), tasks={t:COUNTS[t] for t in ('choose_txt','choose_img','fill_in_blank')},
                             all_required_zip_members_present=True, train_heldout_pid_overlap=0)
    return dict(status='source_membership_verified_not_materialized', N_train=sum(COUNTS.values()), records=records)


def prepare(inventory, root, output):
    import pyarrow.parquet as pq
    from PIL import Image
    source = inspect_sources(inventory,root)
    identity = dict(source_hashes={k:v['sha256'] for k,v in inventory['sources'].items()}, code_sha256=sha256(__file__))
    output.mkdir(parents=True,exist_ok=True)
    ip=output/'source_identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Changed source/preparation code; use a new train directory')
    write_json(ip,identity)
    mp=output/'manifest.json'; cp=output/'candidates.jsonl'
    if mp.exists():
        previous=json.loads(mp.read_text())
        if previous.get('status')=='ready' and sha256(cp)==previous['candidate_manifest_sha256']:
            return previous
    def image(raw):
        if raw is None: return None
        raw=raw['bytes'] if isinstance(raw,dict) else raw
        key=hashlib.sha256(raw).hexdigest(); path=output/'images'/f'{key}.png'
        # Decode every original image even when another question reuses it.
        with Image.open(io.BytesIO(raw)) as im:
            im.load(); rgb=im.convert('RGB')
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True); rgb.save(path)
        return str(path.resolve())
    total=0; seen=set(); actual={}; tmp=output/'candidates.jsonl.tmp'
    def emit(handle,row):
        nonlocal total
        if row['uid'] in seen or row['source_split']!='train' or len(row['candidates'])<2 or not 0<=row['label']<len(row['candidates']):
            raise ValueError('Invalid/duplicate training row; never filter it out')
        seen.add(row['uid']); total+=1; actual[row['dataset']]=actual.get(row['dataset'],0)+1
        handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        if total%1000==0: print(f'TRAIN source preparation {total:,}/{source["N_train"]:,}',flush=True)
    with tmp.open('w',encoding='utf-8') as handle:
        for dataset in ('aokvqa','scienceqa'):
            index=0
            for name in source['records'][dataset]['source_files']:
                for batch in pq.ParquetFile(source['records'][name]['path']).iter_batches(batch_size=32):
                    for row in batch.to_pylist():
                        candidates=[str(c) for c in row['choices']]
                        label=int(row['correct_choice_idx'] if dataset=='aokvqa' else row['answer'])
                        path=image(row.get('image')); question=row['question']
                        if dataset=='scienceqa' and row.get('hint'): question+='\nContext: '+row['hint']
                        if dataset!='scienceqa' and path is None: raise ValueError('Required train image missing')
                        uid=f'{dataset}:train:{index}:{row.get("question_id",index)}'
                        emit(handle,dict(uid=uid,dataset=dataset,images=[path] if path else [],question=question,
                             candidates=candidates,label=label,source_split='train',reference_answers=[],
                             metadata=dict(intrinsic_no_image=path is None, source_ordinal=index, source_id=str(row.get('question_id',index)), no_lecture_solution=True)))
                        index+=1
        with zipfile.ZipFile(source['records']['iconqa_zip']['path']) as archive:
            problems=json.loads(archive.read('iconqa_data/problems.json')); splits=json.loads(archive.read('iconqa_data/pid_splits.json'))
            vocab=sorted({str(problems[pid]['answer']) for pid in splits['fill_in_blank_train']})
            from scripts.data.prepare_iconqa_remaining import compose
            for task,pid in icon_membership(problems,splits):
                r=problems[pid]; prefix=f'iconqa_data/iconqa/train/{task}/{pid}/'
                query=image(archive.read(prefix+'image.png')); question=r['question']
                if task=='choose_txt': candidates=[str(c) for c in r['choices']]; label=int(r['answer']); name='iconqa'
                elif task=='choose_img':
                    options=[image(archive.read(prefix+c)) for c in r['choices']]
                    composed=output/'composed'/f'{pid}.png'; compose(Path(query),[Path(p) for p in options],composed)
                    query=str(composed.resolve()); candidates=[str(i) for i in range(len(options))]; label=int(r['answer']); name='iconqa_choose_img'
                    question+='\nThe question diagram is above. All image options are below, numbered left-to-right from 0. Answer with the zero-based option index.'
                else: candidates=vocab; label=vocab.index(str(r['answer'])); name='iconqa_fill_in_blank'
                emit(handle,dict(uid=f'{name}:train:{pid}',dataset=name,images=[query],question=question,
                     candidates=candidates,label=label,source_split='train',reference_answers=[],
                     metadata=dict(official_pid=pid,task=task,candidate_policy='original options; fill uses whole train-only answer vocabulary')))
    if total!=source['N_train']: raise ValueError('Entire declared source was not converted')
    tmp.replace(cp)
    result=dict(schema='visual-jev-benchmark-manifest-v1',dataset_name='aokvqa_scienceqa_iconqa_whole_train',
                split='train',N=total,N_source=total,source_split='train',public_labels=True,complete_split=True,
                status='ready',candidate_manifest=str(cp.resolve()),candidate_manifest_sha256=sha256(cp),
                revision=identity['source_hashes'],fingerprint=sha256(cp),repo='pinned original released training sources',
                actual_task_counts=actual,source_membership_audit=source,
                training_scope='Whole A-OKVQA train + whole ScienceQA train + all three whole IconQA train tasks; other datasets are full transfer evaluations',
                sampling='none',source_identity=identity)
    write_json(mp,result); print(f'COMPLETE TRAIN MEMBERSHIP: {total:,}',flush=True); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True); p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=ROOT/'data/adapter_study_training')
    p.add_argument('--check-sources',action='store_true'); a=p.parse_args()
    inv=json.loads(a.inventory.read_text())
    print(json.dumps(inspect_sources(inv,a.root) if a.check_sources else prepare(inv,a.root,a.output),ensure_ascii=False,indent=2))
