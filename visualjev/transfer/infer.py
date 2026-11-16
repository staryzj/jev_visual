"""Run the selected Visual-JEV 27B routes on local JSONL questions."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Root containing pinned models and interfaces")
    parser.add_argument("--encoder", choices=["qwen", "llava", "internvl", "siglip2"], default="qwen")
    parser.add_argument("--input", type=Path, required=True, help="JSONL: question, candidates, optional image, uid, state")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mapping", choices=["selected", "flat", "crop", "coordinates"], default="selected")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; choose a new output file")
    if args.mapping in ("crop", "coordinates") and args.encoder != "llava":
        parser.error("crop/coordinates controls are defined for the recorded LLaVA layout")
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from PIL import Image
    from .connector import from_checkpoint
    from .encoders import ExternalEncoder
    from .mapping_controls import map_with_fixed_weights
    from .receiver import FrozenReceiver
    from ..assets import ROOT, content_hash
    manifest = json.loads((ROOT / "release_metadata/visual_jev_interfaces.json").read_text())
    entry = manifest["interfaces"][args.encoder]
    checkpoint = args.assets / entry["path"]
    if content_hash(checkpoint) != entry["sha256"]:
        parser.error("Selected interface checkpoint differs from the recorded hash")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    receiver = FrozenReceiver(args.assets / "base27", args.assets / "openjev27/package/checkpoint", args.device)
    bridge = from_checkpoint(checkpoint, args.device)
    source_key = "qwen4" if args.encoder == "qwen" else args.encoder
    encoder = None  # Image-free rows use the shared receiver without visual extraction.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as output:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question, candidates, state = row["question"], row["candidates"], row.get("state", "")
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                if row.get("image"):
                    image_path = Path(row["image"])
                    if not image_path.is_absolute():
                        image_path = args.input.parent / image_path
                    with Image.open(image_path) as original:
                        image = original.convert("RGB")
                    if encoder is None:
                        encoder = ExternalEncoder(args.encoder, args.assets / source_key, args.device)
                    record = encoder.encode(image, entry["tap"])
                    batches, grid = receiver.prepare_external(question, candidates, image.size, state)
                    features = map_with_fixed_weights(bridge, record, grid, args.mapping)
                    with receiver.external_only():
                        scores = receiver.logits(batches, features)
                else:
                    scores = receiver.no_image(question, candidates, state)
                scores = scores.detach().float().cpu()
                prediction = int(scores.argmax())
                result = dict(uid=row.get("uid", str(line_no)), encoder=args.encoder, mapping=args.mapping,
                              candidates=candidates, prediction=prediction, answer=candidates[prediction],
                              scores=scores.tolist(), probabilities=scores.softmax(-1).tolist(),
                              probabilities_calibrated=(scores / receiver.temperature).softmax(-1).tolist(),
                              generated_answer_tokens=0, interface_sha256=entry["sha256"])
                if "label" in row:
                    result["correct"] = prediction == row["label"]
                receiver.unchanged()
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(f"Scored {result['uid']}: {result['answer']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
