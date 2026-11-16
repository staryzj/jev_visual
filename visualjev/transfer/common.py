"""Portable checkpoint and geometry resources."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)
