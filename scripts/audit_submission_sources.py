#!/usr/bin/env python3
"""Verify the grouped TallyQA mirror against the author's pinned test annotations."""
from collections import Counter
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import requests
from datasets import load_from_disk
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.submission_benchmarks import ROOT,write_json

def main():
    repository='manoja328/TallyQA_dataset'
    response=requests.get(f'https://api.github.com/repos/{repository}/commits/master',timeout=30)
    response.raise_for_status(); revision=response.json()['sha']
    url=f'https://raw.githubusercontent.com/{repository}/{revision}/tallyqa.zip'
    response=requests.get(url,timeout=60); response.raise_for_status()
    archive=zipfile.ZipFile(BytesIO(response.content))
    names=[name for name in archive.namelist() if Path(name).name=='test.json']
    if len(names)!=1: raise ValueError(f'Cannot uniquely find official test.json: {archive.namelist()}')
    source=json.loads(archive.read(names[0]))
    mirror=load_from_disk(str(ROOT/'data/submission_benchmarks/tallyqa/dataset')).remove_columns('image')
    original=Counter((r['question'],str(r['answer']),bool(r['issimple']),r['data_source']) for r in source)
    mirrored=Counter((qa['question'],str(qa['answer']),bool(qa['is_simple']),qa['data_source']) for r in mirror for qa in r['qa'])
    equal=original==mirrored
    report=dict(source=url,repository_revision=revision,zip_sha256=hashlib.sha256(response.content).hexdigest(),
                official_test_count=len(source),mirror_image_groups=len(mirror),mirror_question_count=sum(mirrored.values()),
                simple_count=sum(r['issimple'] for r in source),complex_count=sum(not r['issimple'] for r in source),
                complete_annotation_multiset_matches=equal,missing_question_count=sum((original-mirrored).values()),
                additional_question_count=sum((mirrored-original).values()))
    write_json(ROOT/'data/submission_benchmarks/tallyqa/official_annotation_audit.json',report)
    print(json.dumps(report,indent=2))
    if not equal: raise ValueError('Mirror does not exactly match all official TallyQA test annotations')
    repository='Sri-Harsha/scpp'
    response=requests.get(f'https://api.github.com/repos/{repository}/commits/main',timeout=30)
    response.raise_for_status(); revision=response.json()['sha']
    files=[]; original=Counter()
    for category in ['replace_att','replace_obj','replace_rel','swap_att','swap_obj']:
        url=f'https://raw.githubusercontent.com/{repository}/{revision}/data/{category}.json'
        response=requests.get(url,timeout=30); response.raise_for_status()
        payload=response.json(); source=list(payload.values()) if isinstance(payload,dict) else payload
        original.update((category,r['filename'],r['caption'],r['caption2'],r['negative_caption']) for r in source)
        files.append(dict(source=url,N=len(source),sha256=hashlib.sha256(response.content).hexdigest()))
    mirror=load_from_disk(str(ROOT/'data/submission_benchmarks/sugarcrepe_pp/dataset'))
    mirrored=Counter((r['category'],r['filename'],r['caption'],r['caption2'],r['negative_caption']) for r in mirror)
    equal=original==mirrored
    report=dict(repository_revision=revision,files=files,official_suite_count=sum(original.values()),
                mirror_count=len(mirror),complete_annotation_multiset_matches=equal,
                missing_triplet_count=sum((original-mirrored).values()),additional_triplet_count=sum((mirrored-original).values()))
    write_json(ROOT/'data/submission_benchmarks/sugarcrepe_pp/official_annotation_audit.json',report)
    print(json.dumps(report,indent=2))
    if not equal: raise ValueError('Mirror does not exactly match the official SugarCrepe++ suite')

if __name__=='__main__': main()
