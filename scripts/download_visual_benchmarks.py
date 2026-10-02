#!/usr/bin/env python3
"""Compatibility entry point for the pinned full-split submission downloader.

Older script versions downloaded hidden-label A-OKVQA test and incomplete
repo snapshots. The default now uses the official scored split registry.
"""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='data/submission_benchmarks')
    p.add_argument('--suite',choices=['all','core'],default='all')
    p.add_argument('--datasets',nargs='+')
    p.add_argument('--list',action='store_true')
    p.add_argument('--fetch-images',action='store_true')
    p.add_argument('--refresh',action='store_true')
    args=p.parse_args()
    command=[sys.executable,str(Path(__file__).with_name('submission_benchmarks.py')),'list' if args.list else 'prepare','--root',args.root]
    names=args.datasets or (['scienceqa','aokvqa','ai2d','tallyqa'] if args.suite=='core' else None)
    if names: command+=['--datasets',*names]
    if args.fetch_images: command+=['--fetch-images']
    if args.refresh: command+=['--refresh']
    subprocess.run(command,check=True)

if __name__=='__main__': main()
