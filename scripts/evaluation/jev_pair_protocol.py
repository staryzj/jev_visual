"""CPU-only evidence checks for JEV -> JEV comparisons (never native -> JEV)."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def load_complete_manifest(path, *, training=False):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'ready' or not manifest.get('complete_split'):
        raise ValueError('Whole split is not ready; never replace it with a sample')
    if not manifest.get('public_labels'):
        raise ValueError('Public supervised labels are required')
    split = manifest.get('split', '')
    if training and split != 'train':
        raise ValueError('Only a genuine training split may enter training')
    if not training and (split == 'train' or split.startswith('custom')):
        raise ValueError('Training/custom rows cannot be called full benchmark evaluation')
    # SugarCrepe++ deliberately has split="evaluation (HF storage split=train)".
    candidates = Path(manifest['candidate_manifest'])
    if not candidates.is_file():
        candidates = path.parent / 'candidates.jsonl'
    if not candidates.is_file() or sha256(candidates) != manifest['candidate_manifest_sha256']:
        raise ValueError('Pinned candidate file missing or changed')
    rows = list(read_jsonl(candidates))
    if not rows or len(rows) != manifest['N']:
        raise ValueError('Full declared N differs from candidate membership')
    uids = [row['uid'] for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError('Duplicate UIDs')
    for row in rows:
        if training:
            sources = [row.get('source_split'), row.get('metadata', {}).get('source_split')]
            declared = [s for s in sources if s is not None]
            if not declared or any(s != 'train' for s in declared):
                raise ValueError('Every train row must have original source_split=train; never promote official test/validation into train')
        if len(row['candidates']) < 2:
            raise ValueError('At least two complete candidates required')
        label = row.get('label')
        if label is not None and (type(label) is not int or not 0 <= label < len(row['candidates'])):
            raise ValueError('Invalid supervised candidate label')
    return manifest, rows, candidates


def item_key(row):
    """Secondary input check when different importers rename UIDs.

    Filename/question/candidate matching is a guard, NOT a proof of content-level image
    independence. That limitation must be kept in the audit output.
    Native question templates alone do not identify a multiple-choice example.
    Sorting candidates also detects a reused example after option permutation.
    """
    images = row.get('images', [row['image']] if row.get('image') else [])
    names = tuple(Path(str(p).replace('\\', '/')).name for p in images)
    return (names, str(row.get('question', '')).strip(), tuple(sorted(str(c) for c in row.get('candidates', []))))


def source_item_key(row):
    """Known legacy/current author-ID bridges; require matching split + question.

    Never conclude independence from nonmatching renamed IDs. This only catches
    exact source-item overlap for these known importers, not all near duplicates.
    """
    family = {'scienceqa_image_only': 'scienceqa', 'iconqa_choice': 'iconqa'}.get(row.get('dataset'), row.get('dataset'))
    if family not in {'scienceqa', 'iconqa', 'aokvqa'}:
        return None
    metadata = row.get('metadata', {})
    split = row.get('source_split') or metadata.get('source_split')
    if not split:
        return None
    source_id = str(metadata.get('source_id', row['uid'].split(':', 1)[-1]))
    parts = source_id.split(':')
    if row.get('dataset') in {'scienceqa_image_only', 'iconqa_choice'} and len(parts) >= 3:
        source_id = parts[1]
    question = str(row.get('question', '')).split('\nContext:', 1)[0].strip()
    return family, split, source_id, question


def audit_label_use(provenance, evaluation_rows):
    """Reject any declared train/selection/calibration label reuse in evaluation."""
    sources = provenance.get('used_label_manifests', [])
    if not sources or not any(s.get('purpose') == 'train' for s in sources):
        raise ValueError('Declare all training/selection/calibration label manifests')
    eval_uids = {r['uid'] for r in evaluation_rows}
    eval_keys = {item_key(r) for r in evaluation_rows}
    eval_source_keys = {key for r in evaluation_rows if (key := source_item_key(r)) is not None}
    records = []
    for source in sources:
        path = Path(source['path'])
        if not path.is_file() or sha256(path) != source['sha256']:
            raise ValueError('Training provenance manifest missing or changed')
        count = 0
        for row in read_jsonl(path):
            count += 1
            if row['uid'] in eval_uids or item_key(row) in eval_keys or (source_item_key(row) is not None and source_item_key(row) in eval_source_keys):
                raise ValueError('Evaluation label reuse in ' + source['purpose'] + ': ' + row['uid'])
        if count != source['N']:
            raise ValueError('Training provenance count mismatch')
        records.append(dict(purpose=source['purpose'], N=count, sha256=source['sha256']))
    return dict(status='declared_label_memberships_disjoint', sources=records,
                boundary='Exact UID, image-filename/question/candidate-multiset and known legacy source-ID bridges only; not a perceptual-duplicate audit or proof of undeclared upstream pretraining independence.')


def validated_prediction(row, record, score_key='scores', prediction_key='prediction'):
    if record.get('uid') != row['uid']:
        raise ValueError('Ordered UID coverage mismatch')
    scores = record[score_key]
    if len(scores) != len(row['candidates']) or not all(math.isfinite(float(v)) for v in scores):
        raise ValueError('Complete finite candidate scores required')
    predicted = max(range(len(scores)), key=scores.__getitem__)
    if record[prediction_key] != predicted:
        raise ValueError('Prediction differs from argmax')
    return scores, predicted


def credit(row, record):
    scores, predicted = validated_prediction(row, record)
    name = row['dataset']
    if name == 'textvqa':
        from scripts.evaluation.vqa_official_scoring import normalize
        answers = [normalize(str(a)) for a in row['reference_answers']]
        if not answers:
            raise ValueError('Hidden labels cannot be scored')
        answer = normalize(str(row['candidates'][predicted]))
        return statistics.fmean(min(sum(answer == a for j, a in enumerate(answers) if j != i) / 3, 1)
                                for i in range(len(answers)))
    if name == 'sugarcrepe_pp':
        if len(scores) != 3:
            raise ValueError('Strict ITT requires both positive captions and one negative')
        return float(scores[0] > scores[2] and scores[1] > scores[2])
    if name == 'winoground':
        matrix = record['scores_2x2']
        if len(matrix) != 2 or any(len(v) != 2 for v in matrix) or not all(math.isfinite(float(v)) for pair in matrix for v in pair):
            raise ValueError('All four Winoground scores required')
        a, b = matrix
        return float(a[0] > a[1] and b[1] > b[0] and a[0] > b[0] and b[1] > a[1])
    return float(row.get('label') is not None and predicted == row['label'])


def primary_metric(name):
    return {'textvqa': 'vqa_consensus', 'sugarcrepe_pp': 'strict_itt',
            'winoground': 'group_score'}.get(name, 'accuracy')


def summarize_pair(rows, records):
    totals = dict(before=0.0, after=0.0)
    counts = dict(improved=0, worsened=0, unchanged=0)
    n = 0
    for row, record in itertools.zip_longest(rows, records):
        if row is None or record is None:
            raise ValueError('Partial/extra prediction membership')
        values = {}
        for side in ('before', 'after'):
            prediction = dict(uid=record['uid'], scores=record[side + '_scores'],
                              prediction=record[side + '_prediction'])
            if row['dataset'] == 'winoground':
                prediction['scores_2x2'] = record[side + '_scores_2x2']
            values[side] = credit(row, prediction)
            totals[side] += values[side]
        difference = values['after'] - values['before']
        counts['improved' if difference > 0 else 'worsened' if difference < 0 else 'unchanged'] += 1
        n += 1
    if not n:
        raise ValueError('Empty comparison')
    return dict(N=n, metric=primary_metric(rows[0]['dataset']),
                before=totals['before'] / n, after=totals['after'] / n,
                delta_pp=100 * (totals['after'] - totals['before']) / n, **counts,
                inference='Descriptive paired change; no invented confidence interval or causal superiority claim')


def import_old_jev(directory, manifest, rows, checkpoint_hash, model_files):
    """Read ONLY the old JEV column, never relabel native Qwen as old JEV."""
    directory = Path(directory)
    result_path = directory / 'result.json'
    if not result_path.is_file():
        return {}, None
    result = json.loads(result_path.read_text(encoding='utf-8'))
    identity = result.get('identity', {})
    if result.get('protocol') != 'full_benchmark_before_after' or result.get('status') != 'complete':
        return {}, None
    if (identity.get('checkpoint_sha256') != checkpoint_hash
        or identity.get('candidate_manifest_sha256') != manifest['candidate_manifest_sha256']
        or identity.get('native_model_files') != model_files):
        return {}, None
    path = directory / 'predictions.jsonl'
    if sha256(path) != result['predictions_sha256']:
        raise ValueError('Old JEV prediction hash changed')
    saved = list(read_jsonl(path))
    total = 0.0
    normalized = {}
    for row, prediction in itertools.zip_longest(rows, saved):
        if row is None or prediction is None:
            raise ValueError('Old JEV full membership mismatch')
        scores, predicted = validated_prediction(row, prediction, 'jev_scores', 'jev_prediction')
        converted = dict(uid=row['uid'], scores=scores, prediction=predicted)
        if row['dataset'] == 'winoground':
            converted['scores_2x2'] = prediction['jev_scores_2x2']
        total += credit(row, converted)
        normalized[row['uid']] = converted
    metrics = result['jev_metrics']
    expected = metrics['group_score' if rows[0]['dataset'] == 'winoground' else 'accuracy']
    if metrics['count'] != len(rows) or abs(total / len(rows) - expected) > 1e-9:
        raise ValueError('Old JEV metric differs from raw predictions')
    return normalized, dict(path=str(result_path), result_sha256=sha256(result_path),
                            predictions_sha256=sha256(path), imported_column='jev ONLY; not native')
