# Open-Jev-VL prototype

This prototype keeps the complete Qwen3-VL multimodal base (`visual` and
`language_model`) and feeds the final hidden state into a new scalar decision
head.  It does not load or convert the released Open-Jev-9B LoRA or decision
head: those weights belong to a different backbone and are structurally and
semantically incompatible.

When loading the bare base model, the new head is random and uncalibrated.
Successful bare-model inference proves only that image pixels and candidate
text travel through the full local model.  Use `jev.train --vision` to produce
a VL-specific trained head, optional VL adapters, and calibration temperature.

## Local setup

```bash
cd ~/projects/Open-Jev
source .venv/bin/activate
uv pip install --python .venv/bin/python modelscope qwen-vl-utils pillow torchvision
modelscope download --model Qwen/Qwen3-VL-4B-Instruct \
  --local_dir ./models/Qwen3-VL-4B-Instruct
```

Run the direct image+text smoke test:

```bash
python -m jev.vl_smoke \
  --model ./models/Qwen3-VL-4B-Instruct \
  --image ./test.jpg \
  --device cuda:0
```

Run the HTTP service.  `--image-root` is a security boundary: request paths
must resolve below it.

```bash
python -m jev.server \
  --model ./models/Qwen3-VL-4B-Instruct \
  --vision \
  --image-root . \
  --device cuda:0 \
  --batch-size 1 \
  --max-length 2048
```

In a second shell:

```bash
curl -sS http://127.0.0.1:8791/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @configs/vl-test-request.json
```

The response metadata deliberately reports
`qwen3_vl_random_decision_head_requires_training` and
`untrained_random_initialization`.

## Train a new VL checkpoint

The original Open-Jev-9B adapter/head is never reused.  Three independent
training modes are supported:

```bash
# Frozen visual+language base; train only a fresh decision head.
python -m jev.train --model ./models/Qwen3-VL-4B-Instruct \
  --vision --image-root . --data ./data/vl-paper --output ./runs/vl-head \
  --vl-tuning head --lora-rank 0 --max-length 2048

# Adapt language attention plus the fresh head.
python -m jev.train --model ./models/Qwen3-VL-4B-Instruct \
  --vision --image-root . --data ./data/vl-paper --output ./runs/vl-language \
  --vl-tuning language-lora --lora-rank 8 --max-length 2048

# Adapt visual qkv, language attention and the fresh head.
python -m jev.train --model ./models/Qwen3-VL-4B-Instruct \
  --vision --image-root . --data ./data/vl-paper --output ./runs/vl-full \
  --vl-tuning vision-language-lora --lora-rank 8 --max-length 2048
```

Serve a completed checkpoint with `--vision` and the same image security root:

```bash
python -m jev.server --checkpoint ./runs/vl-full/checkpoint \
  --vision --image-root . --device cuda:0
```

See `docs/research/open-jev-vl/research-plan.md` for the controlled ablation,
data-leakage rules, metrics and evidence boundary.  The bundled `data/vl-smoke`
fixture and `runs/vl-smoke-*` outputs are wiring checks, never paper results.
