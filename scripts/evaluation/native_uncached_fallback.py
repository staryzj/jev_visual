"""Retain verified legacy results; cold-score an ENTIRE case on cache mismatch.

The 0.02 diagnostic threshold is unchanged. Cold results have a separate root
and identity, never a mixture of a legacy cached prefix and a new cold suffix.
"""
import gc
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import transformers
from jev.native_visual_baselines import (
    NativeVisualBaseline, PROMPT_SUFFIX, move_inputs, answer_tokens, cold_conditional_scores,
)
from scripts.evaluation import native_jev_comparison as legacy
from scripts.evaluation.jev_pair_protocol import sha256, write_json, load_complete_manifest, credit, primary_metric
from scripts.workflows.adapter_study_cache import resolve_rows
from scripts.workflows.progress_eta import ProgressETA


class ColdNativeBaseline(NativeVisualBaseline):
    @property
    def protocol(self):
        return dict(name='native_conditional_mean_answer_log_likelihood', prompt_suffix=PROMPT_SUFFIX,
                    answer_tokens='Tokenizer encode(add_special_tokens=False); empty answer=EOS; no answer truncation',
                    inference_path='Full uncached forward for EVERY candidate on EVERY example in this case',
                    cached_prefix_check='Cached numerical guard failed; cached scores rejected, not tolerance-relaxed',
                    note='Released native LM head; same mean-token likelihood, images, questions and full candidate sets')

    @torch.inference_mode()
    def scores(self, image, question, candidates):
        if self.backbone not in {'internvl', 'llava'}:
            raise ValueError('Cold conditional fallback is only for native conditional language models')
        content = [dict(type='image', image=image), dict(type='text', text=question+PROMPT_SUFFIX)]
        encoded = self.processor.apply_chat_template([dict(role='user', content=content)], tokenize=True,
                     add_generation_prompt=True, return_dict=True, return_tensors='pt',
                     processor_kwargs={'truncation': False})
        encoded = move_inputs(encoded, self.device, self.dtype)
        tokens = answer_tokens(self.processor.tokenizer, candidates)
        limit = getattr(self.model.config.text_config, 'max_position_embeddings', None)
        if limit and encoded['input_ids'].shape[1]+max(map(len, tokens))-1 > limit:
            raise ValueError('Native model context capacity exceeded; no question/candidate truncation')
        scores = cold_conditional_scores(self.model, encoded, tokens)
        if len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores):
            raise ValueError('Incomplete/nonfinite uncached native scores')
        self.last_check = None
        return scores


def evaluate_uncached(runtime, config, backbone, dataset, model_files, output,
                      device='cuda:0', batch_size=32, baseline=None):
    from scripts.evaluation.evaluate_submission_benchmarks import load_image
    runtime, output = Path(runtime), Path(output)
    manifest, rows, _ = load_complete_manifest(runtime/config['evaluation_root']/dataset/'manifest.json')
    rows = list(resolve_rows(rows, runtime))
    directory = output/backbone/dataset
    directory.mkdir(parents=True, exist_ok=True)
    trigger_path = directory/'fallback_trigger.json'
    if not trigger_path.exists():
        raise ValueError('Cold fallback requires a recorded actual numerical guard failure')
    trigger = json.loads(trigger_path.read_text())
    if (trigger.get('kind') != 'cached_uncached_numerical_mismatch' or
            trigger.get('tolerance') != .02 or not math.isfinite(trigger.get('difference', float('nan'))) or
            trigger['difference'] <= .02):
        raise ValueError('Invalid numerical fallback trigger')
    identity = dict(protocol='full_native_uncached_fallback_v1', backbone=backbone, dataset=dataset,
                    N=manifest['N'], split=manifest['split'],
                    candidate_manifest_sha256=manifest['candidate_manifest_sha256'], model_files=model_files,
                    evaluator_sha256=sha256(__file__), baseline_sha256=sha256(Path(__file__).resolve().parents[2]/'jev/native_visual_baselines.py'),
                    metric_code_sha256=sha256(Path(__file__).with_name('jev_pair_protocol.py')),
                    legacy_evaluator_sha256=sha256(legacy.__file__), fallback_trigger_sha256=sha256(trigger_path),
                    torch_version=torch.__version__, transformers_version=transformers.__version__,
                    native_batch_size=batch_size, labels_used_for_inference=False, use_cache=False)
    ip, rp, pp = directory/'identity.json', directory/'result.json', directory/'predictions.jsonl'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Cold resume identity changed; existing evidence preserved')
    write_json(ip, identity)
    if rp.exists():
        old = json.loads(rp.read_text())
        if old.get('status') != 'complete' or old['identity'] != identity or old['N'] != manifest['N'] or sha256(pp) != old['predictions_sha256']:
            raise ValueError('Existing cold native result changed')
        return old
    saved = legacy.recover_prefix(pp, rows)
    total = sum(credit(row, record['native']) for row, record in zip(rows, saved))
    n = len(saved)
    own = baseline is None
    if own:
        baseline = ColdNativeBaseline(backbone, runtime/config['visual_models'][backbone], device, batch_size)
    protocol = baseline.protocol
    write_json(directory/'run_manifest.json', dict(identity=identity, repo=manifest.get('repo'),
               revision=manifest.get('revision'), fingerprint=manifest.get('fingerprint'), N=manifest['N'],
               split=manifest['split'], is_true_test=manifest['is_true_test'], public_labels=True,
               checkpoint_files_sha256=model_files, sampling='none', native_protocol=protocol,
               numerical_fallback=trigger))
    progress = ProgressETA({backbone+'/'+dataset: len(rows)},
                           {backbone+'/'+dataset: sum(len(r['candidates']) for r in rows)/len(rows)},
                           initial={backbone+'/'+dataset: n})
    measured = {}

    def publish_status(value):
        write_json(directory/'status.json', value)
        # The unchanged scheduler reads native/.../status.json. Forward ONLY
        # progress, clearly labelled with the separate evidence directory.
        legacy_directory = output.parent/'native'/backbone/dataset
        if not (legacy_directory/'result.json').exists():
            write_json(legacy_directory/'status.json', {**value, 'evidence_directory': str(directory),
                       'progress_forwarded_from': 'native_uncached'})

    try:
        with pp.open('a', encoding='utf-8') as handle:
            for index in range(n, len(rows)):
                row = rows[index]
                image = load_image(row)
                start = time.perf_counter()
                try:
                    scores = baseline.scores(image, row['question'], row['candidates'])
                finally:
                    image.close()
                prediction = dict(uid=row['uid'], scores=scores, prediction=max(range(len(scores)), key=scores.__getitem__))
                if dataset == 'winoground':
                    from PIL import Image
                    matrix = []
                    for path in row['images']:
                        with Image.open(path) as im:
                            matrix.append(baseline.scores(im.convert('RGB'), row['question'], row['candidates']))
                    prediction['scores_2x2'] = matrix
                total += credit(row, prediction)
                n = index+1
                handle.write(json.dumps(dict(uid=row['uid'], native=prediction,
                             scoring_path='full_uncached', windowed_candidate_texts=0,
                             elapsed_seconds=time.perf_counter()-start), ensure_ascii=False)+'\n')
                handle.flush()
                if n % 10 == 0 or n == len(rows):
                    update = progress.update(backbone+'/'+dataset, dict(N_complete=n), force=n == len(rows))
                    if update:
                        measured = update
                    publish_status(dict(status='running', N_complete=n, N_total=len(rows), pid=os.getpid(),
                                   scoring_path='full_uncached',
                                   **{k: v for k, v in measured.items() if k not in {'N_complete', 'N_total'}}))
        if n != manifest['N']:
            raise ValueError('Cold native full denominator incomplete')
        result = dict(schema='native-visual-full-result-v1', status='complete', backbone=backbone,
                      dataset_name=dataset, N=n, split=manifest['split'], is_true_test=manifest['is_true_test'],
                      public_labels=True, identity=identity, revision=manifest.get('revision'),
                      fingerprint=manifest.get('fingerprint'), no_sampling=True, native_protocol=protocol,
                      native_training='Released pretrained model only; no adapter/JEV training',
                      native_metric=dict(name=primary_metric(dataset), score=total/n),
                      windowed_candidate_texts=0, predictions_sha256=sha256(pp), numerical_fallback=trigger)
        write_json(rp, result)
        publish_status(dict(status='complete', N_complete=n, N_total=n, scoring_path='full_uncached'))
        return result
    finally:
        if own:
            baseline.close()


def evaluate_with_fallback(runtime, config, backbone, dataset, model_files, output,
                           device='cuda:0', batch_size=32, legacy_baseline=None, cold_baseline=None):
    output = Path(output)
    cold_root = output.parent/'native_uncached'
    directory = cold_root/backbone/dataset
    trigger = directory/'fallback_trigger.json'
    if trigger.exists():
        return evaluate_uncached(runtime, config, backbone, dataset, model_files, cold_root,
                                 device, batch_size, cold_baseline)
    failure = None
    try:
        return legacy.evaluate_native(runtime, config, backbone, dataset, model_files,
                                      output, device, batch_size, baseline=legacy_baseline)
    except ValueError as error:
        matched = re.fullmatch(r'Native cached/uncached score difference ([0-9.eE+-]+) exceeds tolerance', str(error))
        if matched is None:
            raise  # No OOM/context/missing-image/identity error is disguised as numerical fallback.
        difference = float(matched[1])
        if not math.isfinite(difference) or difference <= .02:
            raise
        failure = dict(kind='cached_uncached_numerical_mismatch', difference=difference, tolerance=.02,
                       reason=str(error), policy='Re-evaluate entire full case with uncached forward; no mixed-score prefix',
                       rejected_cached_directory=str(output/backbone/dataset),
                       legacy_identity_sha256=sha256(output/backbone/dataset/'identity.json'),
                       fallback_code_sha256=sha256(__file__))
    # Leave the exception scope so GPU tensors retained by its traceback can be released.
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    write_json(trigger, failure)
    print(f'Numerical guard retained: {backbone}/{dataset} difference={failure["difference"]:.8f} > 0.02. '
          'Scoring EVERY example/candidate with uncached forward in separate native_uncached directory.', flush=True)
    return evaluate_uncached(runtime, config, backbone, dataset, model_files, cold_root,
                             device, batch_size, cold_baseline)


def selected_native_root(native_root, backbone, dataset):
    native_root = Path(native_root)
    cold_root = native_root.parent/'native_uncached'
    directory = cold_root/backbone/dataset
    if (directory/'fallback_trigger.json').exists():
        if not (directory/'result.json').exists():
            raise ValueError('Full uncached fallback remains incomplete; do not use the rejected cached result')
        return cold_root
    return native_root
