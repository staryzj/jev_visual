"""Build an allowlisted publication snapshot; never stage the research worktree.

Exact full-split scores/predictions are exported after hash and UID checks.
Historical logs are explicitly redacted aggregate exports, NOT byte-identical
original records. Source and export hashes are recorded separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ('scienceqa', 'aokvqa', 'iconqa', 'ai2d', 'vsr', 'sugarcrepe_pp')
ROOT_CODE = (
    'calibrate_visual_jev_v3.py demo_visual_jev_v3.py diagnostic_visual_dependency.py '
    'eval_visual_jev_v3.py predict_visual_jev_v3.py prepare_benchmark_v1.py '
    'run_benchmark_v1_experiment.py run_calibration_comparison.py run_fix_negative_transfer.py '
    'run_independent_datasets.py run_recent_methods_fix.py run_vision_backend_compatibility.py '
    'run_vision_backend_pair_proof.py run_vision_backend_pair_robustness.py '
    'train_negative_transfer_full.py train_visual_jev_v2.py train_visual_jev_v3.py '
    'finalize_negative_transfer_results.py finalize_recent_methods_fix.py '
    'visual_algorithm.py visual_jep.py visual_jev.py visual_jev_diagnostic.py '
    'visual_jev_smoke.py visual_jev_v2.py visual_jev_v2_diagnostic.py '
    'visual_jev_v3.py visual_jev_v3_pipeline.py'
).split()
SCRIPT_PATTERN = re.compile(r'visual|vl|benchmark_v1|backbone|submission|independent|controlled|negative_transfer|coco|calibration|vqa|full_local_pipeline|export_github_release')
DROP = {
    'records', 'failure_cases', 'predictions', 'examples', 'cases', 'pairs',
    'question', 'questions', 'candidate', 'candidates', 'options', 'caption',
    'captions', 'answer', 'answers', 'image', 'images', 'image_path', 'context',
    'state', 'text', 'prompt', 'positive_caption', 'negative_caption',
    'positive_captions', 'negative_captions', 'passage', 'ocr_tokens',
}
PRED_KEYS = {
    'uid', 'scores', 'prediction', 'label', 'blank_scores', 'wrong_scores',
    'wrong_uid', 'full_decision_ms', 'scores_2x2', 'category',
}
SECRET = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bhf_[A-Za-z0-9]{20,}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b|\bsk-[A-Za-z0-9_-]{30,}\b')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    dest = args.destination.resolve()
    if dest == ROOT or ROOT in dest.parents:
        raise ValueError('Use a separate publication checkout outside the research tree')
    if not (dest / '.git').exists():
        raise ValueError('Destination must be an existing cloned publication repository')
    files = []

    def record(source, target, mode, omitted=()):
        files.append(dict(path=target.relative_to(dest).as_posix(),
                          source_path=source.relative_to(ROOT).as_posix(),
                          source_sha256=sha(source), exported_sha256=sha(target),
                          bytes=target.stat().st_size, mode=mode,
                          omitted_fields=list(omitted)))

    def copy(relative, target_relative=None):
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = dest / (target_relative or relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        record(source, target, 'byte-identical')

    def aggregate(relative):
        source = ROOT / relative
        value = load(source)
        omitted = []
        def clean(item, pointer=''):
            if isinstance(item, dict):
                out = {}
                for key, child in item.items():
                    here = pointer + '/' + key
                    if key.lower() in DROP:
                        omitted.append(here)
                    else:
                        out[key] = clean(child, here)
                return out
            if isinstance(item, list):
                return [clean(child, pointer + '/' + str(i)) for i, child in enumerate(item)]
            return item
        exported = clean(value)
        exported['public_export_metadata'] = dict(
            kind='aggregate export', source_path=relative,
            source_sha256=sha(source), omitted_fields=omitted,
            note='Retained fields are unchanged; source questions/images and example logs omitted. This file is not the byte-identical original.')
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        record(source, target, 'declared-aggregate-export', omitted)

    copy('README_VISUAL_JEV.md', 'README.md')
    for relative in ['README_VISUAL_JEV.md', 'LICENSE', 'pyproject.toml', 'requirements.txt', 'BENCHMARK_PROTOCOL.md'] + ROOT_CODE:
        copy(relative)
    for path in sorted((ROOT / 'jev').glob('*.py')):
        copy(path.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / 'scripts').iterdir()):
        if path.is_file() and path.suffix in ('.py', '.sh') and (SCRIPT_PATTERN.search(path.name) or path.name in ('diagnose_jev_text.py', 'make_server_bundle.py', 'verify_publication.py')):
            copy(path.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / 'tests').glob('test_*.py')):
        if re.search(r'visual|vl_|benchmark_feature|calibration|conversation|hf_vision|independent|multidomain|vision_backend|submission|serving', path.name):
            copy(path.relative_to(ROOT).as_posix())
    for name in ('visual_jev_v3_paper.json', 'vl-research-matrix.json', 'vl-test-request.json'):
        copy('configs/' + name)
    for name in ('benchmark_v1.md', 'open-jev-vl-prototype.md', 'visual_dialog_history.md', 'visual_jev_v3_experiment.md'):
        copy('docs/' + name)

    for name in FULL:
        base = ROOT / 'experiments/results/submission_full' / name
        result = load(base / 'result.json')
        if result['status'] != 'complete' or result['protocol'] != 'full_benchmark':
            raise ValueError('Incomplete/wrong-protocol result: ' + name)
        if sha(base / 'predictions.jsonl') != result['predictions_sha256']:
            raise ValueError('Prediction hash mismatch: ' + name)
        seen = set()
        for line in (base / 'predictions.jsonl').read_text().splitlines():
            row = json.loads(line)
            if set(row) - PRED_KEYS or row['uid'] in seen:
                raise ValueError('Unexpected source content or duplicate prediction: ' + name)
            seen.add(row['uid'])
        if len(seen) != result['N'] or len(seen) != result['metrics']['count']:
            raise ValueError('Incomplete full prediction count: ' + name)
        for file in ('result.json', 'predictions.jsonl', 'resume_identity.json'):
            copy('experiments/results/submission_full/' + name + '/' + file)

    for name in ('siglip2', 'internvl', 'llava'):
        aggregate('experiments/results/backbone_generality/' + name + '/results.json')
    for name in ('coco', 'scienceqa', 'aokvqa', 'iconqa'):
        aggregate('experiments/results/independent_datasets_full/' + name + '/result.json')
    for path in sorted((ROOT / 'experiments/results/official_visual_jev_matched').glob('seed-*/*.json')):
        aggregate(path.relative_to(ROOT).as_posix())
    for name in ('metrics.json', 'visual_jev_v3_provenance.json'):
        aggregate('experiments/results/controlled_multiseed/seed-20260928/results/' + name)
    aggregate('experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/best_model_results.json')
    for path in sorted((ROOT / 'experiments/tables/submission').iterdir()):
        if path.is_file() and path.suffix in ('.json', '.csv', '.md', '.tex'):
            copy(path.relative_to(ROOT).as_posix())
    for name in ('audit.json', 'backbone_tables.csv', 'backbone_tables.md', 'PROTOCOL_AND_NEXT_STEPS.md'):
        copy('reports/backbone-adaptation-v6/' + name)
    for name in ('REFERENCE_ALIGNMENT.md', 'additional_references.bib'):
        copy('reports/reference-alignment-v6/' + name)
    for path in sorted((ROOT / 'data/submission_benchmarks').glob('*/manifest.json')):
        relative = path.relative_to(ROOT).as_posix()
        copy(relative, 'release_metadata/benchmarks/' + path.parent.name + '/manifest.json')
    for path in sorted((ROOT / 'data/submission_benchmarks').glob('*/official_annotation_audit.json')):
        copy(path.relative_to(ROOT).as_posix(), 'release_metadata/benchmarks/' + path.parent.name + '/official_annotation_audit.json')

    for item in files:
        path = dest / item['path']
        if path.stat().st_size > 20 * 1024 * 1024 or SECRET.search(path.read_text(encoding='utf-8')):
            raise ValueError('Potential secret/oversized public file: ' + item['path'])
    manifest = dict(schema='visual-jev-publication-manifest-v1',
                    target='https://github.com/staryzj/jev_visual',
                    full_prediction_instances=sum(load(ROOT / 'experiments/results/submission_full' / n / 'result.json')['N'] for n in FULL),
                    excluded=['raw source images/annotations', 'feature/candidate caches', 'checkpoints/model weights', 'partial predictions', 'secrets', 'paper photo/AI raster assets'],
                    files=files)
    (dest / 'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(files=len(files), bytes=sum(item['bytes'] for item in files), full_prediction_instances=manifest['full_prediction_instances'], secret_scan='passed', target=str(dest))))

if __name__ == '__main__':
    main()
