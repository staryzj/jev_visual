"""Run the isolated Visual-JEV V1 baseline on one image and candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from jev.api import candidate_prompts, compile_request
from jev.serving import load_predictor
from visual_jev import VisualJEVBaseline


DEFAULT_CANDIDATES = (
    "interface: A software interface or presentation frame is visible.",
    "animal: An animal is the main subject.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", default="test.jpg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--question",
        default="Which description best matches the image?",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        help="Candidate label/description. Repeat for multiple candidates.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    candidates = tuple(args.candidates or DEFAULT_CANDIDATES)
    if len(candidates) < 2:
        raise ValueError("the smoke test needs at least two candidates for softmax")

    predictor = load_predictor(
        model_id=args.model,
        device=args.device,
        max_length=args.max_length,
        batch_size=len(candidates),
        vision=True,
        image_root=image_path.parent,
    )
    decision_model = predictor.scorer.model
    baseline = VisualJEVBaseline(decision_model, debug=True)

    # Reuse Open-Jev's candidate prompt renderer.  The labels below are only
    # display keys; every actual text feature sees question + one candidate.
    criteria = {f"candidate_{index}": text for index, text in enumerate(candidates)}
    record = compile_request(
        "Judge each proposed answer using the supplied image.",
        {
            "visual_choice": {
                "type": "choice",
                "instructions": args.question,
                "criteria": criteria,
            }
        },
    )[0]
    prompts = candidate_prompts(record)

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.load()
    with torch.inference_mode():
        output = baseline.score_prompts(
            image,
            prompts,
            temperature=args.temperature,
        )

    print("\nCandidate results (one shared JEV scalar head, then one softmax):")
    for candidate, score, probability in zip(
        candidates,
        output.scores.float().cpu().tolist(),
        output.probabilities.float().cpu().tolist(),
    ):
        print(f"  {candidate}")
        print(f"    score={score:.6f} probability={probability:.6f}")
    print(f"  probability_sum={output.probabilities.float().sum().item():.6f}")

    print("\nWeight status:")
    for name, status in baseline.weight_status.items():
        print(f"  {name}: {status}")
    print(
        "  warning: scores/probabilities only verify V1 wiring; the random "
        "VisualAlgorithm projector and decision head are not calibrated."
    )


if __name__ == "__main__":
    main()
