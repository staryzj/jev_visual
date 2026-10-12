"""Opt-in batched frozen-feature caching, without editing the pinned encoder.

This is an execution backend, not a different training protocol. Existing cache
bytes are never replaced. BF16 batched kernels need not be bitwise identical;
checked serial-vs-batch diagnostics and creation provenance are retained.
"""
import json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.workflows.adapter_study_cache import FeaturePool, key, load_feature
from scripts.evaluation.jev_pair_protocol import sha256, write_json


@contextmanager
def file_lock(path, nonblocking=False):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


class LockedFeaturePool(FeaturePool):
    """Per-key locks protect text/blank features shared by independent workers."""
    def __init__(self, root, identity):
        with file_lock(Path(root) / '.locks/identity.lock'):
            super().__init__(root, identity)
        self.canvas_sizes = {}
        self.creation_backend = None

    def stored(self, category, identifier, compute):
        lock = self.root / '.locks' / (key([category, identifier]) + '.lock')
        with file_lock(lock):
            info = self.root / category / (identifier + '.json')
            existed = info.exists() and info.with_suffix('.pt').exists()
            result = super().stored(category, identifier, compute)
            if not existed and self.creation_backend:
                metadata = json.loads(info.read_text())
                write_json(info, {**metadata, 'creation_backend': self.creation_backend})
            return result

    def get(self, row, encoder, backbone, controls=True):
        # Header dimensions exactly reproduce load_image's multi-image canvas;
        # cache hits no longer decode an entire image merely to size its blank.
        from PIL import Image
        from scripts.evaluation.evaluate_submission_benchmarks import load_image
        text, tp = self.stored('text', key([row['question'], row['candidates']]),
                               lambda: encoder.encode_text(row))
        hashes = []
        for path in row['images']:
            if path not in self.image_hashes:
                self.image_hashes[path] = sha256(path)
            hashes.append(self.image_hashes[path])
        tokens, vp = self.stored(backbone + '/visual', key(hashes or ['intrinsic_white_224']),
                                 lambda: encoder.encode_image(load_image(row)))
        blank = bp = None
        if controls:
            paths = tuple(row['images'])
            if paths not in self.canvas_sizes:
                sizes = []
                for path in paths:
                    with Image.open(path) as image:
                        sizes.append(image.size)
                self.canvas_sizes[paths] = ((sum(w for w, _ in sizes) + 16 * (len(sizes)-1),
                                            max(h for _, h in sizes)) if sizes else (224, 224))
            size = self.canvas_sizes[paths]
            blank, bp = self.stored(backbone + '/blank', key(['white', size]),
                                    lambda: encoder.encode_image(Image.new('RGB', size, 'white')))
        return dict(visual=tokens, text=text, blank=blank), dict(uid=row['uid'], visual=vp, text=tp, blank=bp)


def compare_features(serial, batched):
    """Reject shape, finite-value, or materially different frozen features."""
    import torch
    if serial.shape != batched.shape or not torch.isfinite(batched).all():
        raise ValueError('Batched frozen feature shape/finite check failed')
    a, b = serial.float(), batched.float()
    relative = float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a).clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min())
    if cosine < .999 or relative > .02:
        raise ValueError(f'Batched frozen feature check failed: cosine={cosine}, relative_L2={relative}')
    return dict(bitwise_equal=torch.equal(serial, batched), cosine_min=cosine, relative_l2=relative)


def vision_batch(encoder, images):
    """Batch native vision towers and split output by image, NOT by question."""
    import torch
    from jev.hf_vision_backends import normalize_visual_tokens, resample_visual_tokens, _move_inputs
    visual = encoder.visual
    if visual is None:
        return [encoder.encode_image(image) for image in images]
    processor = visual.processor if visual.name == 'siglip2' else getattr(visual.processor, 'image_processor', visual.processor)
    inputs = [processor(images=image, return_tensors='pt') for image in images]
    counts, pixels, sizes = [], [], []
    for image, inp in zip(images, inputs):
        value = inp['pixel_values']
        if visual.name == 'llava':
            from transformers.models.llava_onevision.modeling_llava_onevision import image_size_to_num_patches
            count = image_size_to_num_patches((image.height, image.width), visual.model.config.image_grid_pinpoints,
                                              visual.model.config.vision_config.image_size)
            value = value[0, :count] if value.ndim == 5 else value
            if value.shape[0] != count:
                raise ValueError('LLaVA batch patch count differs from single-image processor')
            sizes.append([image.height, image.width])
        counts.append(value.shape[0])
        pixels.append(value)
    kwargs = _move_inputs({'pixel_values': torch.cat(pixels)}, visual.device, visual.dtype)
    with torch.inference_mode():
        if visual.name == 'siglip2':
            output = visual.model.vision_model(**kwargs, return_dict=True)
        else:
            if visual.name == 'llava':
                kwargs['image_sizes'] = torch.tensor(sizes, device=visual.device, dtype=torch.long)
            output = visual.model.get_image_features(**kwargs)
        # Match the original extractor's last_hidden_state selection exactly,
        # including its native (not multimodal-projector) InternVL/LLaVA features.
        states = output.last_hidden_state
        if states.ndim != 3 or states.shape[0] != sum(counts):
            raise ValueError('Unsupported native batched output; do not mix images')
        return [resample_visual_tokens(normalize_visual_tokens(part), encoder.config['token_count'])
                .detach().cpu().to(torch.bfloat16) for part in states.split(counts)]


def accelerated_training_cache(runtime, config, rows, backbone, pool, index_path, device,
                               vision_batch_size=32, decode_workers=4):
    from scripts.workflows.adapter_study_cache import FrozenEncoders
    from scripts.workflows.progress_eta import ProgressETA
    from scripts.evaluation.evaluate_submission_benchmarks import load_image
    index_path = Path(index_path)
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        if payload['identity'] != pool.identity or [r['uid'] for r in payload['rows']] != [r['uid'] for r in rows]:
            raise ValueError('Whole training cache membership differs')
        return payload['rows']
    encoder = None
    checked = False
    disabled = False
    refs = []
    audit_path = pool.root / ('batch_audit_' + backbone + '.jsonl')
    progress = ProgressETA({'cache_' + backbone: len(rows)}, {'cache_' + backbone: sum(len(r['candidates']) for r in rows)/len(rows)})
    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = FrozenEncoders(runtime, config, backbone, device)
        return encoder
    class LazyEncoder:
        def encode_text(self, row): return get_encoder().encode_text(row)
        def encode_image(self, image): return get_encoder().encode_image(image)
    try:
        with ThreadPoolExecutor(max_workers=decode_workers) as executor:
            for start in range(0, len(rows), vision_batch_size):
                chunk = rows[start:start+vision_batch_size]
                pending = {}
                for row in chunk:
                    hashes = []
                    for path in row['images']:
                        if path not in pool.image_hashes: pool.image_hashes[path] = sha256(path)
                        hashes.append(pool.image_hashes[path])
                    identifier = key(hashes or ['intrinsic_white_224'])
                    path = pool.root / backbone / 'visual' / (identifier + '.pt')
                    if not (path.exists() and path.with_suffix('.json').exists()):
                        pending.setdefault(identifier, row)
                if pending and vision_batch_size > 1 and not disabled:
                    identifiers = list(pending)
                    images = list(executor.map(load_image, pending.values()))
                    try:
                        computed = vision_batch(get_encoder(), images)
                        if not checked:
                            checks = [compare_features(get_encoder().encode_image(image), feature)
                                      for image, feature in zip(images, computed)]
                            record = dict(status='checked',backbone=backbone,N=len(checks), checks=checks,
                                          feature_keys=identifiers,batch_size=vision_batch_size,
                                          code_sha256=sha256(__file__), policy='BF16 numerical equivalence, not bitwise guarantee')
                            with audit_path.open('a') as handle: handle.write(json.dumps(record)+'\n')
                            print(f'{backbone}: batch={vision_batch_size} frozen-feature check passed ({len(checks)} images)', flush=True)
                            checked = True
                        backend = dict(mode='batched_native_vision',batch_size=vision_batch_size,
                                       code_sha256=sha256(__file__),audit=str(audit_path.resolve()))
                        pool.creation_backend = backend
                        try:
                            for identifier, feature in zip(identifiers, computed):
                                pool.stored(backbone + '/visual', identifier, lambda f=feature: f)
                        finally: pool.creation_backend = None
                    except (RuntimeError, ValueError, AttributeError) as error:
                        # Preserve every row; fallback changes execution, NOT dataset.
                        import torch
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        disabled = True
                        with audit_path.open('a') as handle:
                            handle.write(json.dumps(dict(status='serial_fallback',reason=str(error),code_sha256=sha256(__file__)))+'\n')
                        print(f'{backbone}: using original serial encoder after batch check/error: {error}', flush=True)
                    finally:
                        for image in images: image.close()
                for row in chunk:
                    _, reference = pool.get(row, LazyEncoder(), backbone)
                    refs.append(reference)
                measured = progress.update('cache_' + backbone, dict(N_complete=len(refs)), force=len(refs)==len(rows))
                if measured:
                    write_json(pool.root / ('parallel_cache_' + backbone + '.json'),
                               dict(status='complete' if len(refs)==len(rows) else 'running', **measured))
        write_json(index_path, dict(status='complete',identity=pool.identity,rows=refs,N=len(rows),
                                   policy='Every training UID; frozen encoder only; preserved existing feature bytes',
                                   acceleration_audit=str(audit_path.resolve()) if audit_path.exists() else None))
        return refs
    finally:
        if encoder is not None: encoder.close()


class FeaturePrefetch:
    """Read-only lookahead; original optimizer order and arithmetic are unchanged."""
    def __init__(self, pool, references, order, depth=8):
        self.pool, self.references, self.order, self.depth = pool, references, order, depth
        self.position = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = {}
        self._fill()

    def _fill(self):
        for pos in range(self.position, min(len(self.order), self.position+self.depth)):
            if pos not in self.futures:
                self.futures[pos] = self.executor.submit(load_feature, self.pool, self.references[self.order[pos]])

    def __call__(self, pool, record):
        if pool is not self.pool or self.position >= len(self.order) or record['uid'] != self.references[self.order[self.position]]['uid']:
            raise ValueError('Feature prefetch order differs from original trainer; stop rather than alter training')
        result = self.futures.pop(self.position).result()
        self.position += 1
        self._fill()
        return result

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
