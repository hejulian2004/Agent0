"""Record the exact owned server profile, then run vLLM's public entrypoint."""
import os
from pathlib import Path
import sys

from tools.local_profile import load_config
from tools.data_builder.sft_stream import _write_state


def main():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root, "local_4090")
    (root / "logs/teacher").mkdir(parents=True, exist_ok=True)
    _write_state(root / "logs/teacher/current_profile.json", {
        "pid": os.getpid(), "argv": sys.argv[1:], "model": config["assets"]["teacher_model"],
        "status": "starting", "profile": {key: value for key, value in config["sft_data_generation"]["local_teacher"].items() if key != "api_key"},
    })
    from vllm.entrypoints.launchers.api_server.entry import main as serve
    serve()


if __name__ == "__main__":
    main()
