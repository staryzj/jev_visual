"""Prepare EVERY official IconQA image-choice/fill-blank test item separately.

Image options are composed beneath the query, without dropping any option.
Fill-blank uses all distinct official TRAIN answers; test OOVs stay incorrect
in the full denominator. No gold/test answers construct the vocabulary.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from PIL import Image,ImageDraw
from scripts.data.submission_benchmarks import ROOT,DEFAULT_CHECKPOINT,sha256,write_json
from scripts.data.audit_full_images import audit


def compose(query,choices,target):
    images=[]
    for path in [query,*choices]:
        with Image.open(path) as im:
            im.load();images.append(im.convert('RGB'))
    first,*options=images
    cell_width=max(im.width for im in options)+20
    width=max(first.width,cell_width*len(options))
    row_height=max(im.height for im in options)+36
    canvas=Image.new('RGB',(width,first.height+20+row_height),'white')
    canvas.paste(first,((width-first.width)//2,0))
    draw=ImageDraw.Draw(canvas)
    y=first.height+20
    for k,image in enumerate(options):
        x=k*cell_width
        draw.text((x+10,y),f'Option {k}',fill='black')
        canvas.paste(image,(x+10,y+24))
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        with Image.open(target) as previous:
            previous.load()
            if previous.size!=canvas.size or previous.convert('RGB').tobytes()!=canvas.tobytes():
                raise FileExistsError('Existing option composition differs; use a new data root')
    else:
        canvas.save(target,format='PNG')


def prepare(source,out,checkpoint):
    problems_file=source/'problems.json';splits_file=source/'pid_splits.json'
    problems=json.loads(problems_file.read_text());splits=json.loads(splits_file.read_text())
    train_ids=splits['fill_in_blank_train']
    if len(train_ids)!=10913:raise ValueError('Incomplete official fill-blank training annotations')
    if set(train_ids)&set(splits['fill_in_blank_test']):raise ValueError('Train/test overlap')
    vocab=sorted({str(problems[pid]['answer']) for pid in train_ids})
    provenance=dict(source='https://github.com/lupantech/IconQA',split='train',N=len(train_ids),
                    pid_splits_sha256=sha256(splits_file),problems_sha256=sha256(problems_file),
                    policy='all distinct official train answers; no validation/test label use')
    write_json(out/'vocab/iconqa_fill_in_blank.json',dict(candidates=vocab,provenance=provenance))
    download=json.loads((source.parent/'download.json').read_text())
    results=[]
    for task,expected in [('choose_img',11535),('fill_in_blank',3638)]:
        name='iconqa_'+task
        target=out/name;target.mkdir(parents=True,exist_ok=True)
        ids=splits[task+'_test']
        if len(ids)!=expected or len(set(ids))!=expected:raise ValueError('Incomplete official task')
        mp=target/'manifest.json'
        if mp.exists() and json.loads(mp.read_text()).get('status')=='ready':
            print(f'{name}: existing full manifest retained',flush=True);results.append(json.loads(mp.read_text()));continue
        records=[];asset_hashes={}
        for i,pid in enumerate(ids):
            r=problems[pid]
            if r['ques_type']!=task or r['split']!='test':raise ValueError('Official task/split mismatch')
            directory=source/'iconqa/test'/task/pid
            query=directory/'image.png'
            if not query.is_file():raise FileNotFoundError(query)
            with Image.open(query) as im:im.load()
            question=r['question']
            if task=='choose_img':
                choices=[directory/c for c in r['choices']]
                for p in [query,*choices]:
                    with Image.open(p) as im:im.load()
                    asset_hashes[str(p.relative_to(source))]=sha256(p)
                image=target/'images'/f'{pid}.png'
                compose(query,choices,image)
                candidates=[str(j) for j in range(len(choices))];label=int(r['answer'])
                if not 0<=label<len(candidates):raise ValueError('Invalid source answer')
                question+='\nThe question diagram is above. All image options are below, numbered left-to-right from 0. Answer with the zero-based option index.'
                extra=dict(task=task,input_adaptation='query above all option images; labelled left-to-right, no option removed',
                           original_query=str(query),original_choice_images=[str(p) for p in choices])
            else:
                image=query;candidates=vocab;answer=str(r['answer'])
                label=vocab.index(answer) if answer in vocab else None
                asset_hashes[str(query.relative_to(source))]=sha256(query)
                extra=dict(task=task,candidate_policy='all distinct official TRAIN answers; OOV incorrect in full test denominator')
            records.append(dict(uid=f'{name}:{pid}',dataset=name,images=[str(image.resolve())],question=question,
                                candidates=candidates,label=label,reference_answers=[str(r['answer'])],
                                source_split='test',metadata=extra))
            if (i+1)%1000==0:print(f'{name}: prepared {i+1}/{len(ids)}',flush=True)
        cp=target/'candidates.jsonl'
        with cp.open('x',encoding='utf-8') as f:
            for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        manifest=dict(schema='visual-jev-benchmark-manifest-v1',dataset_name=name,task=task,repo='lupantech/IconQA',
                      source='https://github.com/lupantech/IconQA',split='test',N=len(records),N_source=len(records),
                      N_all_iconqa_test=21489,is_true_test=True,public_labels=True,seed=20260928,
                      revision='official ZIP SHA256:'+download['sha256'],fingerprint=sha256(problems_file),
                      checkpoint_sha256=sha256(checkpoint),candidate_manifest=str(cp.resolve()),candidate_manifest_sha256=sha256(cp),
                      complete_split=True,status='ready',protocol='full_benchmark',excluded={},sampling='none; entire official task',
                      task_adapter_sha256=sha256(__file__),download=download,
                      license=(source/'LICENSE.md').read_text(),
                      candidate_vocab=provenance if task=='fill_in_blank' else None)
        write_json(mp,manifest);write_json(target/'source_image_hashes.json',asset_hashes)
        result=audit(target)
        if result['status']!='complete':raise ValueError('Complete image audit failed')
        results.append(manifest)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/iconqa_official/iconqa_data')
    p.add_argument('--root',type=Path,default=ROOT/'data/submission_benchmarks')
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    a=p.parse_args();prepare(a.source,a.root,a.checkpoint)
