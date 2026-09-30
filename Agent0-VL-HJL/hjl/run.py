"""Command-line interface to execute HJL visual reasoning on an image."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .config import HJLConfig
from .engine import HJLEngine
from .trajectory import to_canonical_trajectory


def _create_mock_image() -> Path:
    """Create a temporary dummy image for smoke testing."""
    tmp = tempfile.NamedTemporaryFile(prefix="hjl_smoke_", suffix=".png", delete=False)
    img = Image.new("RGB", (100, 100), color=(180, 180, 180))
    # Draw a small simulated target feature
    for x in range(40, 55):
        for y in range(40, 45):
            img.putpixel((x, y), (20, 20, 20))
    img.save(tmp.name, format="PNG")
    return Path(tmp.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HJL hierarchical visual reasoning agent.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "react", "react_verifier", "hjl"],
        default="hjl",
        help="Execution mode.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Perform visual inspection and reasoning.",
        help="Instruction prompt.",
    )
    parser.add_argument("--category", type=str, default="visual_object", help="Object or scene category.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum allowable inspection steps.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save trajectories.",
    )
    parser.add_argument("--mock", action="store_true", help="Use mock model caller and generate test image if needed.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    image_path = args.image
    if not image_path:
        if not args.mock:
            parser.error("--image is required for live execution (when not running with --mock).")
        mock_file = _create_mock_image()
        image_path = str(mock_file)
        logging.info(f"Generated test image at: {image_path}")

    # Load configuration
    config = HJLConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    out_dir = Path(args.output_dir or config.trajectory_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_file = out_dir / "hjl_trajectory.jsonl"

    engine = HJLEngine(config=config, mock=args.mock)
    logging.info(f"Running mode: {args.mode} on {image_path}")

    if args.mode == "direct":
        result = engine.run_direct(image_path, args.instruction, args.category)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == "react":
        result = engine.run_react(image_path, args.instruction, args.category, args.max_steps)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == "react_verifier":
        result = engine.run_react_verifier(image_path, args.instruction, args.category, args.max_steps)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        state = engine.run_hjl(
            image_path=image_path,
            instruction=args.instruction,
            category=args.category,
            max_steps=args.max_steps,
            trajectory_output_path=traj_file,
        )

        canonical = to_canonical_trajectory(state)
        canonical_file = out_dir / f"{state.sample_id}_canonical.json"
        with canonical_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(canonical.to_dict(), indent=2, ensure_ascii=False))

        logging.info(f"HJL completed with stop reason: {state.stop_reason}")
        logging.info(f"Trajectory saved to {traj_file}")
        logging.info(f"Canonical trajectory saved to {canonical_file}")

        print("\n=== FINAL PREDICTION ===")
        print(json.dumps(state.final_prediction, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
