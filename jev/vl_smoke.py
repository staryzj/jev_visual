"""Run one local image+text Open-Jev-VL wiring smoke test."""
import argparse
import json
from pathlib import Path

import torch

from .serving import load_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", default="test.jpg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    predictor = load_predictor(
        model_id=args.model, device=args.device, max_length=args.max_length,
        batch_size=1, vision=True, image_root=image.parent,
    )
    request = {
        "state": "Inspect the attached image.",
        "images": [image.name],
        "questions": {
            "content": {
                "type": "choice",
                "instructions": "Which description best matches the image?",
                "criteria": {
                    "interface": "A software interface or presentation frame is visible.",
                    "animal": "An animal is the main subject.",
                },
            }
        },
    }
    result = predictor.predict(request)
    model = predictor.scorer.model
    result["prototype"] = {
        "visual_backbone": type(model.backbone.visual).__name__,
        "language_backbone": type(model.backbone.language_model).__name__,
        "head_status": model.head_status,
        "base_trainable_parameters": sum(
            value.numel() for value in model.backbone.parameters() if value.requires_grad),
        "head_trainable_parameters": sum(
            value.numel() for value in model.head.parameters() if value.requires_grad),
    }
    result["cuda"] = {
        "available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024 ** 3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024 ** 3, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
