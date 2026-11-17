"""Browser UI for the paper's selected frozen-receiver Visual-JEV checkpoints."""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
UI = Path(__file__).resolve().parent
MAX_BODY = 24 * 1024 * 1024


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected an image, question and candidate list.")
    question = payload.get("question", "")
    options = payload.get("candidates", [])
    image = payload.get("image", "")
    if not isinstance(question, str) or not question.strip() or len(question) > 12000:
        raise ValueError("Enter a question of 1–12,000 characters.")
    if not isinstance(options, list) or not 2 <= len(options) <= 12:
        raise ValueError("Supply between 2 and 12 options.")
    if any(not isinstance(c, str) or not c.strip() or len(c) > 4000 for c in options):
        raise ValueError("Every option must contain text, up to 4,000 characters.")
    options = [c.strip() for c in options]
    if len(set(options)) != len(options):
        raise ValueError("Options must be distinct.")
    if not isinstance(image, str) or not image.startswith("data:image/") or ";base64," not in image:
        raise ValueError("Upload a PNG, JPEG or WebP image.")
    try:
        raw = base64.b64decode(image.split(",", 1)[1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid image encoding.") from exc
    if not raw or len(raw) > 16 * 1024 * 1024:
        raise ValueError("The image must be smaller than 16 MB.")
    from PIL import Image, ImageOps
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError("Use PNG, JPEG or WebP.")
            if im.width * im.height > 24_000_000:
                raise ValueError("Use an image with at most 24 megapixels.")
            im.load()
            decoded = ImageOps.exif_transpose(im).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Cannot decode this image.") from exc
    return decoded, question.strip(), options


class Runtime:
    def __init__(self, assets, encoder, device):
        self.assets, self.name, self.device = Path(assets).resolve(), encoder, device
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.error = None
        self.loading = True
        self.receiver = self.bridge = self.encoder = None
        self.load_seconds = None

    def load(self):
        try:
            import torch
            from visualjev.assets import content_hash
            from visualjev.transfer.connector import from_checkpoint
            from visualjev.transfer.encoders import ExternalEncoder
            from visualjev.transfer.receiver import FrozenReceiver
            self.entry = json.loads((ROOT / "release_metadata/visual_jev_interfaces.json").read_text())["interfaces"][self.name]
            checkpoint = self.assets / self.entry["path"]
            if content_hash(checkpoint) != self.entry["sha256"]:
                raise ValueError("Visual interface hash does not match the paper checkpoint.")
            for key in ("base27", "openjev27", "qwen4" if self.name == "qwen" else self.name):
                if not (self.assets / key).exists():
                    raise FileNotFoundError(f"Missing model resource: {self.assets / key}")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            start = time.perf_counter()
            self.receiver = FrozenReceiver(self.assets / "base27", self.assets / "openjev27/package/checkpoint", self.device)
            self.bridge = from_checkpoint(checkpoint, self.device)
            self.encoder = ExternalEncoder(self.name, self.assets / ("qwen4" if self.name == "qwen" else self.name), self.device)
            torch.cuda.synchronize(self.device)
            self.load_seconds = time.perf_counter() - start
            self.ready.set()
            logging.info("Paper runtime ready on %s, %.1f seconds", self.device, self.load_seconds)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logging.exception("Model loading failed")
        finally:
            self.loading = False

    def status(self):
        return {"ready": self.ready.is_set(), "loading": self.loading, "busy": self.lock.locked(),
                "error": self.error, "encoder": self.name, "receiver": "Open-Jev 27B v1.1",
                "load_seconds": self.load_seconds}

    def predict(self, payload):
        image, question, candidates = validate(payload)
        if not self.ready.is_set():
            raise RuntimeError(self.error or "Models are still loading. Please wait.")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("The model is processing another request. Please retry shortly.")
        try:
            import torch
            from torch.nn.attention import sdpa_kernel, SDPBackend
            from visualjev.transfer.mapping_controls import map_with_fixed_weights
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                record = self.encoder.encode(image, self.entry["tap"])
                batches, grid = self.receiver.prepare_external(question, candidates, image.size)
                features = map_with_fixed_weights(self.bridge, record, grid, "selected")
                torch.cuda.synchronize(self.device)
                visual_done = time.perf_counter()
                with self.receiver.external_only():
                    scores = self.receiver.logits(batches, features).detach().float().cpu()
                torch.cuda.synchronize(self.device)
                finished = time.perf_counter()
                self.receiver.unchanged()
            choice = int(scores.argmax())
            return {"request_id": str(uuid.uuid4()), "question": question, "candidates": candidates,
                    "prediction": choice, "answer": candidates[choice], "scores": scores.tolist(),
                    "probabilities": scores.softmax(-1).tolist(),
                    "probabilities_calibrated": (scores / self.receiver.temperature).softmax(-1).tolist(),
                    "temperature": self.receiver.temperature, "encoder": self.name,
                    "receiver": "Open-Jev 27B v1.1", "mapping": "selected",
                    "interface_sha256": self.entry["sha256"], "generated_answer_tokens": 0,
                    "image_size": list(image.size), "timing_seconds": {
                        "visual_and_preparation": visual_done - started, "scoring": finished - visual_done,
                        "model_total": finished - started}, "fresh_inference": True}
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    runtime = None

    def reply(self, status, body, kind="application/json; charset=utf-8"):
        data = json.dumps(body, ensure_ascii=False).encode() if isinstance(body, dict) else body
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, (UI / "web/index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self.reply(200, self.runtime.status())
        elif self.path == "/example.png":
            self.reply(200, (UI / "examples/scene.png").read_bytes(), "image/png")
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            return self.reply(404, {"error": "Not found"})
        origin = self.headers.get("Origin")
        if origin and origin not in ("http://" + self.headers.get("Host", ""), "https://" + self.headers.get("Host", "")):
            return self.reply(403, {"error": "Cross-origin requests are not permitted."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("Upload too large or empty request.")
            payload = json.loads(self.rfile.read(length))
            result = self.runtime.predict(payload)
            self.reply(200, result)
        except (ValueError, TypeError, KeyError) as exc:
            self.reply(400, {"error": str(exc)})
        except RuntimeError as exc:
            logging.exception("Inference request failed")
            self.reply(503, {"error": str(exc)})
        except Exception:
            logging.exception("Unexpected inference failure")
            self.reply(500, {"error": "Inference failed. Check the server log."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", default=os.environ.get("VISUAL_JEV_ASSETS", str(ROOT / "assets")))
    p.add_argument("--encoder", choices=["qwen", "llava", "internvl", "siglip2"], default="qwen")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = Runtime(args.assets, args.encoder, args.device)
    Handler.runtime = runtime
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=runtime.load, daemon=True).start()
    logging.info("Open http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
