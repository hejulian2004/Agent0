"""Download the three missing paper-listed SFT source datasets.

Completed files and Hugging Face's local download metadata are reused on
restart. Transient download failures are retried. Video rows in MM-RLHF are
excluded because this reproduction trains on image problems.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import os
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from huggingface_hub import snapshot_download


SOURCES = {
    "llava_ov_image": {
        "repo": "lmms-lab/LLaVA-OneVision-Data",
        "revision": "7ca5e5bf8b2006d5dfa0549756198474f0897f63",
        "path": "llava-ov-image/train",
        "patterns": ["*/train-*.parquet", "README.md"],
    },
    "mm_rlhf": {
        "repo": "yifanzhang114/MM-RLHF",
        "revision": "29cfeea5929979d12b42a8f30201e03c188408a6",
        "path": "mm-rlhf",
        "patterns": ["data.jsonl", "data/train-*.parquet", "long.zip",
                     "mcq.zip", "short.zip", "safety.zip", "README.md"],
    },
    "smr": {
        "repo": "yifanzhang114/SMR",
        "revision": "88c3fd38b124324568886f656211d787089cf739",
        "path": "smr",
        "patterns": ["SMR.json", "README.md"],
    },
}


def _extract_zip(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive member {member.filename!r}")
        archive.extractall(destination)


def _proxy_summary() -> str:
    value = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not value:
        value = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
    if not value:
        return "none (HTTPS_PROXY/ALL_PROXY are unset)"
    parsed = urlsplit(value)
    if not parsed.hostname:
        return "set, but URL could not be parsed"
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}" if parsed.port else f"{parsed.scheme}://{parsed.hostname}"


def _permanent_error(exc: Exception) -> bool:
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EACCES, errno.EPERM):
        return True
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status in (400, 401, 403, 404)


def download(name: str, root: Path, max_retries: int) -> None:
    spec = SOURCES[name]
    target = root / "data/raw/.staging" / spec["path"]
    target.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"Downloading {name} from {spec['repo']} at {spec['revision']}", flush=True)
    print(f"Hub endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}; "
          f"HTTPS proxy: {_proxy_summary()}; "
          f"Xet disabled: {os.environ.get('HF_HUB_DISABLE_XET', '0')}", flush=True)
    failures = 0
    while True:
        try:
            snapshot_download(
                repo_id=spec["repo"], repo_type="dataset", revision=spec["revision"],
                allow_patterns=spec["patterns"], local_dir=target, max_workers=8,
            )
            break
        except Exception as exc:
            if _permanent_error(exc):
                raise
            failures += 1
            if max_retries and failures > max_retries:
                raise RuntimeError(f"{name}: exceeded {max_retries} download retries") from exc
            delay = min(60, 5 * 2 ** min(failures - 1, 4))
            detail = re.sub(r"https?://\S+", "<url>", str(exc).splitlines()[0])[:200]
            print(f"{name}: download interrupted ({type(exc).__name__}: {detail}); "
                  f"retrying in {delay}s; existing files are kept", flush=True)
            time.sleep(delay)
    if name == "mm_rlhf":
        train = target / "train"
        train.mkdir(exist_ok=True)
        link = train / "data.jsonl"
        if not link.exists():
            link.symlink_to("../data.jsonl")
        for archive in ("long.zip", "mcq.zip", "short.zip", "safety.zip"):
            _extract_zip(target / archive, target)
    elif name == "smr":
        train = target / "train"
        train.mkdir(exist_ok=True)
        link = train / "SMR.json"
        if not link.exists():
            link.symlink_to("../SMR.json")
    manifest = {
        "source": name, "repo": spec["repo"], "commit": spec["revision"],
        "files_pattern": spec["patterns"], "started_utc": started,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "downloaded", "local_path": str(target),
        "note": "SMR images are referenced from its component datasets" if name == "smr" else "",
    }
    (target / "AGENT0_DOWNLOAD.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", choices=sorted(SOURCES))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-retries", type=int, default=0,
                        help="Retries per source after transient failures; 0 means keep retrying")
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    root = args.root.resolve()
    for name in args.sources or list(SOURCES):
        download(name, root, args.max_retries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
