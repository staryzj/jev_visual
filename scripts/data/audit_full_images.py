"""Decode every required image; restore damaged derived PNGs from pinned data.

No LOAD_TRUNCATED_IMAGES, filtering, or synthetic replacements are allowed.
Old damaged assets are retained next to their replacements for audit.
"""
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image
from datasets import load_from_disk
from scripts.data.submission_benchmarks import ROOT, read_jsonl, write_json, sha256


def decode(path):
    try:
        with Image.open(path) as image:
            image.load()
            image.convert('RGB').load()
        return None
    except Exception as error:
        return f'{type(error).__name__}: {error}'


def audit(directory, repair=False):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if not manifest.get('candidate_manifest'):
        return dict(dataset=directory.name, status='blocked', reason=manifest.get('reason'))
    paths = set()
    n = 0
    for row in read_jsonl(manifest['candidate_manifest']):
        n += 1
        paths.update(row['images'])
        paths.update(row.get('candidate_images', []))
    if n != manifest['N']:
        raise ValueError('candidate manifest count differs')
    paths = sorted(paths)
    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = list(pool.map(decode, paths))
    failures = {p: e for p, e in zip(paths, errors) if e}
    records = []
    source = None
    for path, error in list(failures.items()):
        record = dict(path=path, original_error=error, repaired=False)
        if repair:
            try:
                if source is None:
                    source = load_from_disk(str(directory / 'dataset'))
                target = Path(path)
                # Only this pipeline's <source row index>.png derivatives are restored.
                if target.parent.resolve() != (directory / 'images').resolve() or not target.stem.isdigit():
                    raise ValueError('Not a row-indexed derived asset; original source required')
                r = source[int(target.stem)]
                original = r.get('query_image', r.get('image'))
                original.load()
                temp = target.with_suffix('.repair.png')
                original.convert('RGB').save(temp, format='PNG')
                if decode(temp):
                    raise ValueError('Restored image still cannot fully decode')
                backup = target.with_suffix('.damaged-original.png')
                if backup.exists():
                    raise FileExistsError('Previous backup exists; do not overwrite')
                record.update(old_sha256=sha256(target), new_sha256=sha256(temp), backup=str(backup))
                target.replace(backup)
                temp.replace(target)
                record['repaired'] = True
                del failures[path]
            except Exception as exc:
                record['repair_error'] = f'{type(exc).__name__}: {exc}'
        records.append(record)
    result = dict(schema='full-image-decode-audit-v1', dataset=directory.name,
                  N=n, N_unique_images=len(paths), status='complete' if not failures else 'blocked',
                  failed_images=failures, repairs=records,
                  source_revision=manifest.get('revision'),
                  candidate_manifest_sha256=sha256(manifest['candidate_manifest']),
                  policy='full pixel decode; no tolerated truncation; no row exclusions')
    write_json(directory / 'image_decode_audit.json', result)
    print(f'{directory.name}: images={len(paths)}, failures={len(failures)}, repaired={sum(r["repaired"] for r in records)}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/submission_benchmarks')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    results = [audit(args.root / name, args.repair) for name in args.datasets]
    if any(r['status'] != 'complete' for r in results):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
