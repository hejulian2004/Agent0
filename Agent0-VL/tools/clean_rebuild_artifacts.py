"""Inventory and remove old generated SFT/RL datasets and standalone logs.

The default is a dry run. Raw downloads, validation sets, processed provenance
indexes, and all checkpoints are deliberately outside the deletion targets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path


ACTIVE_MARKERS = ("verl.trainer.main_ppo", "swift sft", "tools.sft_builder.build_stream",
                  "tools.sft_builder.rebuild_paper_1000", "tools.build_multisource_rl",
                  "rl-agent0-2x4090-optimized.sh", "sft-agent0-2x4090-full.sh")


def targets(root: Path) -> list[Path]:
    selected: list[Path] = []
    for directory in (root / "data/sft", root / "data/rl"):
        if not directory.exists():
            continue
        selected.extend(path for path in directory.iterdir()
                        if path.name != "validation"
                        and not path.name.startswith("validation_"))
    for directory in (root / "logs", root / "outputs"):
        if not directory.exists():
            continue
        selected.extend(path for path in directory.iterdir() if path.name != "README.md")
    return sorted(selected)


def active_jobs() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                            text=True, check=True)
    return [line for line in result.stdout.splitlines()
            if any(marker in line for marker in ACTIVE_MARKERS)
            and "clean_rebuild_artifacts.py" not in line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true", help="Delete listed targets")
    args = parser.parse_args()
    root = args.root.resolve()
    paths = targets(root)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run",
                      "targets": [str(path.relative_to(root)) for path in paths]},
                     ensure_ascii=False, indent=2))
    if not args.apply:
        return 0
    jobs = active_jobs()
    if jobs:
        raise SystemExit("Training or data construction is active; refusing cleanup:\n"
                         + "\n".join(jobs))
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    timestamp = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    for directory in (root / "logs", root / "outputs"):
        if directory.exists():
            with (directory / "README.md").open("a", encoding="utf-8") as handle:
                handle.write(f"\n- {timestamp}: removed {len(paths)} old generated data/log targets "
                             "for paper-source rebuild; raw, validation, and checkpoints retained.\n")
    print(f"Removed {len(paths)} targets. Historical checkpoints were retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
