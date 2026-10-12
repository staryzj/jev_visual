"""Full paired OLD JEV / NEW JEV evaluation. No native-Qwen score column.

Uses the same frozen scorer, frozen image/text encoders and candidate inputs.
Training provenance is mandatory for the proposed checkpoint. Existing full
old JEV scores can be imported after independent score/UID/hash verification.
Do not launch alongside the current single-GPU evaluation worker.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.jev_pair_protocol import (
    audit_label_use, import_old_jev, load_complete_manifest, read_jsonl,
    sha256, summarize_pair, validated_prediction, write_json,
)


def state_hash(module):
    import hashlib
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def training_identity(path, checkpoint):
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('checkpoint_sha256') != sha256(checkpoint):
        raise ValueError('Training provenance does not belong to this checkpoint')
    if not value.get('full_declared_train') or value.get('status') != 'complete':
        raise ValueError('Full training is not complete; do not claim a fully trained checkpoint')
    return value


def require_matched_training(before, after):
    # Same visits/update budget, not necessarily equal wall time or parameter count.
    keys = ('train_candidate_sha256', 'N_train', 'epochs', 'seed',
            'initial_checkpoint_sha256', 'backbone', 'visual_model_identity',
            'text_model_identity', 'learning_rate', 'gradient_accumulation')
    for key in keys:
        if key not in before or key not in after or before[key] != after[key]:
            raise ValueError('Matched training differs or is undeclared: ' + key)


def write_tables(output, results):
    fields = ['dataset', 'split', 'N', 'metric', 'old_jev_pct', 'new_jev_pct', 'delta_pp']
    rows = [dict(dataset=r['dataset_name'], split=r['split'], N=r['N'],
                 metric=r['metrics']['metric'], old_jev_pct=100*r['metrics']['before'],
                 new_jev_pct=100*r['metrics']['after'], delta_pp=r['metrics']['delta_pp']) for r in results]
    with (output/'jev_to_jev.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    def cells(row):
        return [f'{row[k]:.2f}' if isinstance(row[k], float) else str(row[k]) for k in fields]
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---']*len(fields)) + ' |']
    lines += ['| ' + ' | '.join(cells(row)) + ' |' for row in rows]
    (output/'jev_to_jev.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    def tex(value):
        return value.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
    lines = ['% Audited complete JEV -> JEV results only; no unfinished scores.',
             '\\begin{tabular}{lllrrrr}', '\\hline',
             'Dataset & Split & Metric & $N$ & Old JEV (\\%) & New JEV (\\%) & $\\Delta$ (pp) \\\\', '\\hline']
    for row in rows:
        values = [row['dataset'], row['split'], row['metric'], str(row['N']),
                  f"{row['old_jev_pct']:.2f}", f"{row['new_jev_pct']:.2f}", f"{row['delta_pp']:+.2f}"]
        lines.append(' & '.join(tex(v) for v in values) + ' \\\\')
    lines.extend(['\\hline', '\\end{tabular}'])
    (output/'jev_to_jev.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--after-provenance', type=Path, required=True)
    p.add_argument('--before-provenance', type=Path)
    p.add_argument('--comparison-kind', choices=['historical_checkpoint', 'matched_training'], default='historical_checkpoint')
    p.add_argument('--root', type=Path, default=ROOT/'data/submission_benchmarks')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--text-model', type=Path, default=ROOT/'models/Qwen3-VL-4B-Instruct')
    p.add_argument('--backbone', choices=['qwen', 'siglip2', 'internvl', 'llava'], default='qwen')
    p.add_argument('--visual-model', type=Path)
    p.add_argument('--token-count', type=int, default=196)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--candidate-batch', type=int, default=32)
    p.add_argument('--old-result-root', type=Path, default=ROOT/'experiments/results/complete_eval_20261007')
    p.add_argument('--reuse-old-results', action='store_true', help='Optional verified old JEV reuse; default rescoring both checkpoints with the same features')
    p.add_argument('--controls', action='store_true', help='Full blank/wrong-image diagnostics, not semantic counterfactual pairs')
    a = p.parse_args()
    if a.candidate_batch < 1 or a.token_count < 1:
        p.error('Positive compute batch/token count required')
    if a.before.resolve() == a.after.resolve() or sha256(a.before) == sha256(a.after):
        p.error('Two distinct JEV checkpoints are required')
    after_training = training_identity(a.after_provenance, a.after)
    before_training = training_identity(a.before_provenance, a.before) if a.before_provenance else None
    if a.comparison_kind == 'matched_training':
        if before_training is None:
            p.error('Both training provenances required for matched training')
        require_matched_training(before_training, after_training)
    inputs = {}
    for name in a.datasets:
        manifest, rows, _ = load_complete_manifest(a.root/name/'manifest.json')
        leak_audit = {'after': audit_label_use(after_training, rows)}
        leak_audit['before'] = audit_label_use(before_training, rows) if before_training else {'status': 'undeclared_historical_training; not an independent algorithm-effect claim'}
        inputs[name] = (manifest, rows, leak_audit)
    # Deferred heavyweight imports: help and CPU evidence tests need no GPU library.
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from types import SimpleNamespace
    from PIL import Image
    from scripts.evaluation.run_full_before_after import model_identity
    from scripts.evaluation.cached_candidate_features import cached_features
    from scripts.evaluation.evaluate_submission_benchmarks import load_image, image_signatures, different_image_index
    from scripts.workflows.progress_eta import ProgressETA
    from scripts.training.train_visual_jev_v2 import format_candidate
    from visualjev.models.visual_jev_v2 import Qwen3VLFeatureExtractor
    from visualjev.models.visual_jev_v3_pipeline import ThreeStageVisualJEV

    old, _ = ThreeStageVisualJEV.from_checkpoint(a.before, map_location='cpu')
    new, _ = ThreeStageVisualJEV.from_checkpoint(a.after, map_location='cpu')
    scorer_hash = state_hash(old.decision)
    if scorer_hash != state_hash(new.decision):
        raise ValueError('Scorer tensors differ: cannot claim the same frozen JEV scorer')
    for module in (old, new):
        module.to(a.device).eval().requires_grad_(False)
    weights = model_identity(a.text_model)
    visual_weights = model_identity(a.visual_model) if a.visual_model else weights
    if after_training['backbone'] != a.backbone or after_training['text_model_identity'] != weights or after_training['visual_model_identity'] != visual_weights:
        raise ValueError('Evaluation encoders differ from training provenance')
    a.output.mkdir(parents=True, exist_ok=True)
    identity = dict(before_sha256=sha256(a.before), after_sha256=sha256(a.after), scorer_state_sha256=scorer_hash,
                    text_model_identity=weights, visual_model_identity=visual_weights, backbone=a.backbone,
                    token_count=a.token_count, controls=a.controls, comparison_kind=a.comparison_kind,
                    reuse_old_results=a.reuse_old_results,
                    code_sha256=sha256(__file__), evidence_checks_sha256=sha256(Path(__file__).with_name('jev_pair_protocol.py')),
                    after_provenance_sha256=sha256(a.after_provenance),
                    before_provenance_sha256=sha256(a.before_provenance) if a.before_provenance else None,
                    inputs={n: m['candidate_manifest_sha256'] for n, (m, _, _) in inputs.items()})
    ip = a.output/'identity.json'
    if ip.exists() and json.loads(ip.read_text(encoding='utf-8')) != identity:
        raise ValueError('Changed experiment identity; choose a new output directory')
    lock = a.output/'worker.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    write_json(ip, identity)
    visual_extractor = None
    try:
        processor = AutoProcessor.from_pretrained(a.text_model, local_files_only=True, min_pixels=256*28*28, max_pixels=512*28*28)
        processor.tokenizer.padding_side = 'right'
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        full = AutoModelForImageTextToText.from_pretrained(a.text_model, local_files_only=True,
                   dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': a.device}).eval().requires_grad_(False)
        extractor = Qwen3VLFeatureExtractor(SimpleNamespace(processor=processor, backbone=full.model))
        if a.backbone != 'qwen':
            if a.visual_model is None:
                raise ValueError('Target visual model required')
            from jev.hf_vision_backends import HFVisionTokenExtractor
            visual_extractor = HFVisionTokenExtractor(a.backbone, a.visual_model, device=torch.device(a.device))
        def encode_image(image):
            return extractor.encode_image_pre_merger(image) if visual_extractor is None else visual_extractor.encode(image, a.token_count)
        def score(module, visual, text):
            aligned = module.alignment(visual)
            return torch.cat([module.score_aligned(aligned, chunk).scores for chunk in text.split(a.candidate_batch)]).float().cpu().tolist()
        results = []
        totals = {n: len(rows) for n, (_, rows, _) in inputs.items()}
        means = {n: sum(len(r['candidates']) for r in rows)/len(rows) for n, (_, rows, _) in inputs.items()}
        progress = ProgressETA(totals, means)
        with torch.inference_mode():
            for name, (manifest, rows, leak_audit) in inputs.items():
                out = a.output/name; out.mkdir(parents=True, exist_ok=True)
                pp = out/'predictions.jsonl'
                saved = list(read_jsonl(pp)) if pp.exists() else []
                if len(saved) > len(rows) or [r['uid'] for r in saved] != [r['uid'] for r in rows[:len(saved)]]:
                    raise ValueError('Invalid full-split resume prefix')
                if saved:
                    summarize_pair(rows[:len(saved)], saved)
                reused, reused_source = import_old_jev(a.old_result_root/name, manifest, rows, identity['before_sha256'], weights) if a.backbone == 'qwen' and a.reuse_old_results else ({}, None)
                signatures = image_signatures(rows) if a.controls and name != 'winoground' else None
                progress.update(name, dict(N_complete=len(saved)), force=True)
                with pp.open('a', encoding='utf-8') as handle:
                    for i in range(len(saved), len(rows)):
                        row = rows[i]; started = time.perf_counter()
                        image = load_image(row)
                        visual = encode_image(image)
                        texts = [format_candidate(row['question'], c) for c in row['candidates']]
                        text = cached_features(extractor, texts, a.candidate_batch) if len(texts) > 64 else torch.cat([extractor.encode_candidates(texts[j:j+a.candidate_batch]) for j in range(0, len(texts), a.candidate_batch)])
                        record = dict(uid=row['uid'])
                        for side, module in [('before', old), ('after', new)]:
                            scores = reused[row['uid']]['scores'] if side == 'before' and row['uid'] in reused else score(module, visual, text)
                            predicted = max(range(len(scores)), key=scores.__getitem__)
                            record.update({side+'_scores': scores, side+'_prediction': predicted})
                            validated_prediction(row, record, side+'_scores', side+'_prediction')
                            if name == 'winoground':
                                matrix = []
                                for path in row['images']:
                                    with Image.open(path) as im:
                                        matrix.append(score(module, encode_image(im.convert('RGB')), text))
                                record[side+'_scores_2x2'] = matrix
                        if signatures is not None:
                            j = different_image_index(i, signatures)
                            if j is None:
                                raise ValueError('No distinct wrong image; do not invent a visual control')
                            blank = encode_image(Image.new('RGB', image.size, 'white'))
                            wrong = encode_image(load_image(rows[j]))
                            record['wrong_uid'] = rows[j]['uid']
                            for side, module in [('before', old), ('after', new)]:
                                record[side+'_blank_scores'] = score(module, blank, text)
                                record[side+'_wrong_scores'] = score(module, wrong, text)
                        record['elapsed_seconds'] = time.perf_counter()-started
                        handle.write(json.dumps(record, ensure_ascii=False)+'\n'); handle.flush(); saved.append(record)
                        if (i+1) % 10 == 0 or i+1 == len(rows):
                            status = dict(status='running', N_complete=i+1, N_total=len(rows), pid=os.getpid())
                            write_json(out/'status.json', status); progress.update(name, status, force=i+1 == len(rows))
                result = dict(schema='jev-to-jev-full-v1', status='complete', protocol='full_jev_checkpoint_comparison',
                              dataset_name=name, split=manifest['split'], N=len(rows), is_true_test=manifest['is_true_test'],
                              identity=identity, candidate_manifest_sha256=manifest['candidate_manifest_sha256'],
                              predictions_sha256=sha256(pp), metrics=summarize_pair(rows, saved), no_sampling=True,
                              training_label_use_audit=leak_audit, imported_old_jev=reused_source,
                              comparison_scope='Old JEV vs new JEV; same frozen scorer and encoders. Historical checkpoint changes do NOT isolate algorithm effects; matched-training CE vs CE+invalid is only a regularizer ablation.')
                if signatures is not None:
                    result['visual_dependency'] = {}
                    for condition in ('blank', 'wrong'):
                        changed = [{**r, **{side+'_scores': r[side+'_'+condition+'_scores'] for side in ('before', 'after')},
                                    **{side+'_prediction': max(range(len(r[side+'_'+condition+'_scores'])), key=r[side+'_'+condition+'_scores'].__getitem__) for side in ('before', 'after')}} for r in saved]
                        result['visual_dependency'][condition] = summarize_pair(rows, changed)
                    result['semantic_counterfactual'] = None
                    result['semantic_counterfactual_reason'] = 'Verified semantic pairs not supplied; wrong images are not semantic counterfactuals.'
                write_json(out/'result.json', result)
                write_json(out/'status.json', dict(status='complete', N_complete=len(rows), N_total=len(rows)))
                results.append(result); write_tables(a.output, results)
        write_json(a.output/'completion.json', dict(status='complete', datasets=a.datasets, N=sum(totals.values()), source_results=[str(a.output/n/'result.json') for n in a.datasets]))
    except Exception as error:
        write_json(a.output/'completion.json', dict(status='failed', reason=f'{type(error).__name__}: {error}',
                                                   all_requested_tasks_complete=False))
        raise
    finally:
        if visual_extractor is not None:
            visual_extractor.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
