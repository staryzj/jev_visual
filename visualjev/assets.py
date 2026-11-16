"""Download immutable Visual-JEV dependencies and verify their recorded bytes.

This module uses only Python's standard library. Model files stay outside Git;
the lock file records exact upstream revisions, sizes and content hashes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "release_metadata" / "visual_jev_assets.lock.json"


def content_hash(path: Path, git_blob: bool = False) -> str:
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path: Path, record: dict) -> bool:
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    expected = record.get("sha256") or record.get("git_blob")
    return expected == content_hash(path, git_blob=not bool(record.get("sha256")))


def download_file(asset: dict, record: dict, destination: Path, progress) -> dict:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe asset path: {relative}")
    target = destination / asset["key"] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not matches(target, record):
            raise ValueError(f"Existing asset differs from lock: {target}")
        progress(asset["key"], record["path"], record["size"], "verified")
        return {"path": str(target), "status": "verified"}
    partial = target.with_name(target.name + ".partial")
    url = asset["url_prefix"] + quote(record["path"], safe="/") if "url_prefix" in asset else (
        f"https://huggingface.co/{asset['repo_id']}/resolve/{asset['revision']}/"
        f"{quote(record['path'], safe='/')}?download=true"
    )
    last_error = None
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > record["size"]:
            raise ValueError(f"Oversized partial asset: {partial}")
        if offset == record["size"]:
            if matches(partial, record):
                partial.replace(target)
                progress(asset["key"], record["path"], offset, "verified")
                return {"path": str(target), "status": "downloaded"}
            raise ValueError(f"Asset hash mismatch: {partial}")
        headers = {"User-Agent": "Visual-JEV/reproducible-assets", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            # Refresh signed redirects after an interrupted or expired transfer.
            separator = "&" if "?" in url else "?"
            request_url = url + (f"{separator}retry={time.time_ns()}" if attempt else "")
            with urlopen(Request(request_url, headers=headers), timeout=60) as response:
                if offset and response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected range response")
                    mode = "ab"
                elif response.status == 200:
                    mode, offset = "wb", 0
                else:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                last_notice = time.monotonic()
                with partial.open(mode) as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                        offset += len(chunk)
                        if offset > record["size"]:
                            raise ValueError("Response exceeds locked asset size")
                        if time.monotonic() - last_notice >= 10:
                            progress(asset["key"], record["path"], offset, "downloading")
                            last_notice = time.monotonic()
            if not matches(partial, record):
                raise ValueError(f"Downloaded asset size or hash mismatch: {partial}")
            partial.replace(target)
            progress(asset["key"], record["path"], offset, "verified")
            return {"path": str(target), "status": "downloaded"}
        except HTTPError as exc:
            if exc.code in (401, 403) and attempt >= 1:
                raise RuntimeError(f"Access denied for {asset['key']}/{record['path']}") from exc
            last_error = exc
        except (OSError, TimeoutError) as exc:
            last_error = exc
        if attempt < 5:
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Download failed for {asset['key']}/{record['path']}: {last_error}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--destination", type=Path, default=ROOT / "models")
    parser.add_argument("--assets", nargs="+", help="Asset keys; omitted means all assets")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="Skip model tensors")
    parser.add_argument("--status", type=Path)
    args = parser.parse_args(argv)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    assets = lock["assets"]
    if args.assets:
        unknown = set(args.assets) - {a["key"] for a in assets}
        if unknown:
            parser.error(f"Unknown assets: {sorted(unknown)}")
        assets = [a for a in assets if a["key"] in args.assets]
    jobs = [(a, f) for a in assets for f in a["files"]
            if not args.metadata_only or not f["path"].endswith((".safetensors", ".pt"))]
    total = sum(f["size"] for _, f in jobs)
    mutex = threading.Lock()
    state = {"status": "running", "bytes_total": total, "files_total": len(jobs), "files": {}, "errors": []}

    def progress(asset, filename, count, phase):
        with mutex:
            state["files"][f"{asset}/{filename}"] = {"bytes": count, "status": phase}
            state["bytes_received"] = sum(f["bytes"] for f in state["files"].values())
            state["files_verified"] = sum(f["status"] == "verified" for f in state["files"].values())
            if args.status:
                args.status.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.status.with_name(args.status.name + ".tmp")
                temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
                temporary.replace(args.status)
            if phase == "verified":
                print(f"Verified {asset}/{filename} ({count:,} bytes)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.verify_only:
            futures = {pool.submit(matches, args.destination / a["key"] / f["path"], f): (a, f) for a, f in jobs}
        else:
            futures = {pool.submit(download_file, a, f, args.destination, progress): (a, f) for a, f in jobs}
        for future in as_completed(futures):
            asset, record = futures[future]
            try:
                result = future.result()
                if args.verify_only:
                    if not result:
                        raise ValueError(f"Missing or changed asset: {asset['key']}/{record['path']}")
                    progress(asset["key"], record["path"], record["size"], "verified")
            except Exception as exc:
                with mutex:
                    state["errors"].append(str(exc))
                print(str(exc), flush=True)
    state["status"] = "failed" if state["errors"] else "complete"
    if args.status:
        args.status.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in state.items() if k != "files"}), flush=True)
    return int(bool(state["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
