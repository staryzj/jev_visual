"""CUDA-synchronized B1 measurement from the full efficiency protocol."""
import time

import torch
from PIL import Image

from .prompts import parse_answer, parse_rationale, question as short_question, rationale_question
from .scorer import external_only, prepare_external


def visual_count(model, processor, ids):
    token_ids = set()
    for owner in [model.config, processor, processor.tokenizer]:
        for name in ['image_token_id', 'image_token_index', 'img_context_token_id']:
            value = getattr(owner, name, None)
            if isinstance(value, int):
                token_ids.add(value)
    vocab = processor.tokenizer.get_vocab()
    for text in ['<|image_pad|>', '<IMG_CONTEXT>', '<image>']:
        if text in vocab:
            token_ids.add(vocab[text])
    return sum(int(ids.eq(value).sum()) for value in token_ids)


def context_limit(model):
    config = getattr(model.config, 'text_config', model.config)
    return int(getattr(config, 'max_position_embeddings', 8192))


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000

def measure(item, *, kind, model, processor=None, encoder=None, bridge=None,
            tap=None, short_cap=64, rationale_cap=1024):
    """Measure one item on one visible CUDA device with preloaded components.

    Call inside inference mode and the math SDPA context. For external scoring,
    pass the frozen receiver, external encoder and locked bridge. Native arms
    take a model and processor. Returns the original per-item measurement fields.
    """
    row = item['row']
    if kind == 'matching' and not row['images']:
        return dict(uid=row['uid'], dataset=row['dataset'], label=row['label'], candidate_N=len(row['candidates']), has_image=False, cluster=item["cluster"], supported=False, correct=False, prediction=None, e2e_ms=None, generated_tokens=0, input_tokens_processed=0, reason='Native SigLIP2 does not support image-free QA')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    image = None
    if row['images']:
        with Image.open(row['images'][0]) as source_image:
            image = source_image.convert('RGB')
    image_ms = (time.perf_counter() - start) * 1000
    output = ''
    cap = None
    exhausted = False
    truncated = 0
    if kind == 'external':
        if image is None:
            scores, decision_ms = timed(lambda: model.no_image(row))
            lengths = [len(ids) for ids in model.reference.prompts(row)]
            stages = dict(image_decode_ms=image_ms, preprocessing_ms=0, vision_ms=0, bridge_ms=0, decision_ms=decision_ms)
            unique_visual = visual_processed = 0
        else:
            (batches, grid), prep_ms = timed(lambda: prepare_external(model, row, image.size))
            source, vision_ms = timed(lambda: encoder.encode(image, tap))
            features, bridge_ms = timed(lambda: bridge([source], [grid])[0])
            with external_only(model):
                scores, decision_ms = timed(lambda: model.logits(batches, features))
            lengths = [int(batch['input_ids'].numel()) for batch in batches]
            unique_visual = len(features)
            visual_processed = unique_visual * len(batches)
            stages = dict(image_decode_ms=image_ms, preprocessing_ms=prep_ms, vision_ms=vision_ms, bridge_ms=bridge_ms, decision_ms=decision_ms)
        prediction = int(scores.argmax())
        generated_ids = []
        input_count = sum(lengths)
        nonpad = input_count
        visible = 0
        extra = dict(logits=scores.cpu().tolist(), unique_visual_tokens=unique_visual, visual_positions_processed=visual_processed, candidate_sequence_lengths=lengths, forward_passes=len(lengths))
    else:
        def processing():
            if kind == 'matching':
                texts = [row['question'] + ' Answer: ' + candidate for candidate in row['candidates']]
                return processor(text=texts, images=[image], padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda:0')
            question = rationale_question(row) if kind == 'rationale' else short_question(row)
            content = ([dict(type='image')] if image is not None else []) + [dict(type='text', text=question)]
            prompt = processor.apply_chat_template([dict(role='user', content=content)], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return processor(text=[prompt], **(dict(images=[image]) if image is not None else {}), return_tensors='pt').to('cuda:0')
        inputs, prep_ms = timed(processing)
        inputs = {key: value.to(torch.bfloat16) if value.is_floating_point() else value for key, value in inputs.items()}
        input_count = int(inputs['input_ids'].numel())
        nonpad = int(inputs['attention_mask'].sum()) if 'attention_mask' in inputs else input_count
        image_positions = visual_count(model, processor, inputs['input_ids'])
        if kind == 'matching':
            lengths = [len(processor.tokenizer(row['question'] + ' Answer: ' + candidate, add_special_tokens=True)['input_ids']) for candidate in row['candidates']]
            truncated = sum(length > 64 for length in lengths)
            scores, infer_ms = timed(lambda: model(**inputs).logits_per_image[0].float())
            prediction = int(scores.argmax())
            generated_ids = []
            visible = 0
            extra = dict(candidate_scores=scores.cpu().tolist(), truncated_candidates=truncated, image_patch_tokens=196)
        else:
            cap = rationale_cap if kind == 'rationale' else short_cap
            assert inputs['input_ids'].shape[1] + cap <= context_limit(model), 'Full input plus generation budget exceeds published context; no silent truncation'
            generated, infer_ms = timed(lambda: model.generate(**inputs, max_new_tokens=cap, do_sample=False, use_cache=True))
            generated_ids = generated[0, inputs['input_ids'].shape[1]:].cpu().tolist()
            output = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            visible = sum(token not in processor.tokenizer.all_special_ids for token in generated_ids)
            exhausted = len(generated_ids) >= cap and generated_ids[-1] not in processor.tokenizer.all_special_ids
            if kind == 'rationale':
                prediction, marker = parse_rationale(output, row)
            else:
                prediction = parse_answer(output, len(row['candidates']), row['candidates'])
                marker = None
            extra = dict(output=output, generated_token_ids=generated_ids, explicit_rationale_present=kind == 'rationale', final_answer_marker_found=marker is not None, hidden_reasoning_tokens=None)
            if marker is not None:
                extra['rationale_tokens_retokenized_estimate'] = len(processor.tokenizer.encode(output[:marker], add_special_tokens=False))
                extra['answer_tokens_retokenized_estimate'] = len(processor.tokenizer.encode(output[marker:], add_special_tokens=False))
        stages = dict(image_decode_ms=image_ms, preprocessing_ms=prep_ms, full_native_inference_ms=infer_ms)
        extra.update(visual_input_placeholders=image_positions, padded_input_positions=input_count, nonpad_input_positions=nonpad, processor_shapes={key: list(value.shape) for key, value in inputs.items() if 'pixel' in key or 'grid' in key})
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak = torch.cuda.max_memory_allocated() / 2**30
    return dict(uid=row['uid'], dataset=row['dataset'], cluster=item['cluster'], label=row['label'], candidate_N=len(row['candidates']), has_image=image is not None, supported=True, prediction=prediction, correct=prediction == row['label'], e2e_ms=elapsed_ms, generated_tokens=len(generated_ids), visible_generated_tokens=visible, input_tokens_processed=input_count, input_tokens_nonpad=nonpad, budget_exhausted=exhausted, generation_cap=cap, peak_memory_GiB=peak, stages_ms=stages, **extra)

