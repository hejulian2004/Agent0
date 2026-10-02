"""Verify the real local server process arguments without changing it."""
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse


def verify(config):
    teacher = config["sft_data_generation"]["local_teacher"]
    parsed = urlparse(teacher["base_url"])
    if parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}:
        raise ValueError("local_4090 requires a locally inspectable teacher")
    with urlopen(teacher["base_url"] + "/models", timeout=10) as response:
        models = json.load(response)
    if teacher["model"] not in {row["id"] for row in models["data"]}:
        raise ValueError("Teacher served model differs from profile")
    serve = teacher["serve"]
    expected = {"--model": config["assets"]["teacher_model"],
        "--tensor-parallel-size": str(serve["tensor_parallel_size"]),
        "--pipeline-parallel-size": str(serve["pipeline_parallel_size"]),
        "--max-model-len": str(serve["max_model_len"]),
        "--max-num-seqs": str(serve["max_num_seqs"]),
        "--max-num-batched-tokens": str(serve["max_num_batched_tokens"]),
        "--gpu-memory-utilization": str(serve["gpu_memory_utilization"]),
        "--dtype": str(serve["dtype"]), "--kv-cache-dtype": str(serve["kv_cache_dtype"])}
    candidates = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if not any("vllm" in arg or "tools.local_teacher" in arg for arg in args):
            continue
        flags = {}
        for index, arg in enumerate(args[:-1]):
            if arg.startswith("--"):
                flags[arg] = args[index + 1]
        for alias, name in {"--tensor-model-parallel-size": "--tensor-parallel-size",
                            "--pipeline-model-parallel-size": "--pipeline-parallel-size"}.items():
            if alias in flags:
                flags[name] = flags[alias]
        if flags.get("--port", "8000") != str(parsed.port or 8000):
            continue
        if flags.get("--model") != expected["--model"]:
            continue
        candidates.append(flags)
    for flags in candidates:
        if any(flags.get(key) != value for key, value in expected.items()):
            continue
        spec = json.loads(flags.get("--speculative-config", "{}"))
        count = teacher["speculative_decoding"]["num_speculative_tokens"] if teacher["speculative_decoding"]["enabled"] else 0
        if count and (spec.get("method") != "mtp" or spec.get("num_speculative_tokens") != count):
            continue
        if not count and spec.get("num_speculative_tokens", 0):
            continue
        return {"matched": True, "model": teacher["model"]}
    raise ValueError("Existing teacher profile cannot be matched exactly; no restart performed")
