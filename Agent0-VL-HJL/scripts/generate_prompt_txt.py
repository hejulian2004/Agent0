"""Generate the checked-in system prompt from the canonical protocol module."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def _load_canonical_templates():
    """Load the prompt module without importing VERL's heavyweight root package."""

    module_path = REPO_ROOT / "verl" / "prompts" / "agent0_templates.py"
    spec = importlib.util.spec_from_file_location(
        "agent0vl_canonical_templates",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load canonical prompt module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_prompt(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    text = _load_canonical_templates().render_system_prompt().rstrip() + "\n"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "scripts" / "prompt.txt",
        help="Prompt output path (default: scripts/prompt.txt)",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    generate_prompt(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
