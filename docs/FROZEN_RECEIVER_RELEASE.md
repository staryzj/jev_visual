# Visual-JEV frozen-receiver runtime

Visual-JEV replaces the receiver's image representations with features from
one external encoder. The visual interface normalizes, spatially resamples
and projects those features into the receiver's embedding space. The
released language backbone, LoRA and scalar decision head remain frozen.

## Install

Use a repository checkout and Python 3.12 with a CUDA build of PyTorch.
The tested environment uses Transformers 5.10.2, PEFT 0.19.1 and
Accelerate 1.13.0. Install PyTorch for your CUDA version, then:

```bash
python -m pip install -e '.[transfer]'
```

An editable checkout is required because the runtime reads the pinned
source and manifests from the repository. The BF16 27B receiver needs a
large CUDA device; the validation used an 80 GB H100. Quantization and
device offloading are outside this recorded inference configuration.

## Download and verify

```bash
# All base models, source models, released receiver packages and four interfaces
python -m visualjev.assets --destination /path/to/assets --workers 3

# A smaller Qwen-only inference setup
python -m visualjev.assets --destination /path/to/assets \
  --assets base27 openjev27 qwen4 interfaces

# Check sizes and content hashes without downloading
python -m visualjev.assets --destination /path/to/assets --verify-only
```

The downloader uses immutable upstream revisions, resumes partial files,
and verifies each file. A complete download is about 89.9 GB. Base and
source models stay outside Git. The original interface checkpoints and
copies of the released receiver packages are also available as
[release assets](https://github.com/staryzj/jev_visual/releases/tag/visual-jev-v0.2.0).
Receiver ZIPs unpack directly into `assets/openjev27` and `assets/openjev9`;
put the four `.pt` files in `assets/interfaces`.

| Directory | Contents |
|---|---|
| `base27`, `base9` | Pinned Qwen receiver base weights |
| `openjev27`, `openjev9` | Released LoRA, scalar head, configuration and license |
| `qwen4`, `llava`, `internvl`, `siglip2` | Pinned source models and processors |
| `interfaces` | Four original trained Visual-JEV checkpoints |

The four selected interfaces target **27B**. The downloaded 9B package is
a dependency archive; these interfaces cannot be attached to its different
embedding width.

## Score questions

Each JSONL input row contains `question`, `candidates`, and optionally
`image`, `uid`, `state` and a zero-based `label`. Relative image paths are
resolved against the input file's directory. Missing images use the shared
text-only receiver.

```json
{"uid":"example","image":"scene.png","question":"Where is the red circle?","candidates":["left","right"],"label":0}
```

```bash
python -m visualjev.transfer.infer \
  --assets /path/to/assets --encoder llava \
  --input /path/to/questions.jsonl --output /path/to/predictions.jsonl
```

Output contains candidate scores, T=1 probabilities, separately labelled
temperature-calibrated probabilities, the chosen answer and checkpoint
hash. The runtime generates no answer tokens. It preserves the recorded
2,048-token maximum and rejects oversized prompts rather than truncating.

## Spatial controls and interface fitting

`--mapping selected` uses the saved interface's mapping. With LLaVA,
`--mapping flat`, `--mapping crop` and `--mapping coordinates` compare Flat
1D, cropped bilinear and processor-coordinate sampling using the same
projection and receiver weights. These are fixed-weight interventions;
they do not reproduce separately trained mapping arms.

`visualjev/transfer/training.py` provides the recorded feature-alignment,
trusted-teacher decision loss and serial input-gradient updates. It updates
the interface while keeping the receiver frozen. Callers supply disjoint
training records, native training targets, an optimizer, the fitting budget
and a checkpoint-selection rule. The selected historical interfaces used
their own recorded recipes; the included primitives are not a claim that
one new training command reproduces every selected checkpoint.

## Validation

```bash
VISUAL_JEV_ASSETS=/path/to/assets python -m unittest discover \
  -s tests -p test_frozen_transfer.py -v
```

Seven CPU tests cover checkpoint loading, frozen parameters, file hashes,
Qwen identity mapping, padding exclusion and constant-feature preservation.
The GPU smoke test compares the portable and original experiment code on
one rendered scene: all four encoder tensors and bridge outputs are exactly
equal. Candidate scoring, the three LLaVA controls and the text-only path
also run successfully. See `release_metadata/runtime_validation.json`.
This checks runtime migration; it is not a new benchmark evaluation.
