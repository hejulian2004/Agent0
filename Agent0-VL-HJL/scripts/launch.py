"""Launch Agent0 workflows from the project-root config.yaml."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(profile=None) -> dict[str, Any]:
    from tools.local_profile import load_config as load
    return load(ROOT, profile)


def flatten(prefix: str, value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(f"{prefix}.{key}" if prefix else key, child)
    else:
        yield prefix, value


def hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def response_environment(config: dict[str, Any]) -> dict[str, str]:
    response = config["responses"]
    env = os.environ.copy()
    key_name = str(response.get("api_key_env", "AGENT0_RESPONSES_API_KEY"))
    api_key = env.get(key_name)
    if not api_key and not DRY_RUN:
        raise SystemExit(f"Missing API key environment variable: {key_name} (define in .env or environment)")
    values = {
        "AGENT0_RESPONSES_BASE_URL": env.get("AGENT0_RESPONSES_BASE_URL") or response.get("base_url"),
        "AGENT0_RESPONSES_API_KEY": api_key,
        "AGENT0_RESPONSES_MODEL": env.get("AGENT0_RESPONSES_MODEL") or response.get("model"),
        "AGENT0_RESPONSES_TIMEOUT_SECONDS": env.get("AGENT0_RESPONSES_TIMEOUT_SECONDS") or response.get("timeout_seconds"),
        "AGENT0_RESPONSES_MAX_RETRIES": env.get("AGENT0_RESPONSES_MAX_RETRIES") or response.get("max_retries"),
        "AGENT0_RESPONSES_MAX_TOOL_ROUNDS": env.get("AGENT0_RESPONSES_MAX_TOOL_ROUNDS") or response.get("max_tool_rounds"),
        "AGENT0_RESPONSES_MAX_OUTPUT_TOKENS": env.get("AGENT0_RESPONSES_MAX_OUTPUT_TOKENS") or response.get("max_output_tokens"),
    }
    if not values["AGENT0_RESPONSES_MODEL"] and not DRY_RUN:
        raise SystemExit("Set AGENT0_RESPONSES_MODEL in .env or responses.model in config.yaml to the target endpoint model")
    env.update({key: str(value) for key, value in values.items() if value is not None})
    return env


def sandbox_environment(config: dict[str, Any]) -> dict[str, str]:
    settings = config["sandbox"]
    backend = str(settings["backend"])
    if backend not in {"local_subprocess", "remote"}:
        raise SystemExit("sandbox.backend must be local_subprocess or remote")
    env = os.environ.copy()
    endpoint_env = str(settings.get("endpoint_env", "AGENT0_SANDBOX_ENDPOINT"))
    endpoint = env.get(endpoint_env)
    env["AGENT0_SANDBOX_BACKEND"] = backend
    env["SANDBOX_RUN_TIMEOUT"] = str(settings["run_timeout_seconds"])
    env["SANDBOX_CPU_TIMEOUT"] = str(settings["cpu_timeout_seconds"])
    env["SANDBOX_MEM_LIMIT_MB"] = str(settings["memory_limit_mb"])
    env["SANDBOX_MAX_OUTPUT_BYTES"] = str(settings["max_output_bytes"])
    env["SANDBOX_MAX_CONCURRENT_PROCESSES"] = str(settings["max_concurrent_processes"])
    env["SANDBOX_PRELOAD_PACKAGES"] = json.dumps(settings.get("preload_packages", []))
    # Keep Ultralytics-generated settings/cache inside this project's venv.
    yolo_config_dir = ROOT / ".venv/share/ultralytics-config"
    if not DRY_RUN:
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
    env["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
    tool_runtime = config.get("tool_runtime", {})
    detector = tool_runtime.get("detector", {})
    retrieval = tool_runtime.get("retrieval", {})
    env["AGENT0_DETECTOR_MODEL"] = str(detector.get("model_path", ""))
    env["AGENT0_DETECTOR_CONFIDENCE"] = str(detector.get("confidence", 0.25))
    env["AGENT0_DETECTOR_MAX_DETECTIONS"] = str(detector.get("max_detections", 100))
    env["AGENT0_DETECTOR_DEVICE"] = str(detector.get("device", "cpu"))
    env["AGENT0_RETRIEVAL_CORPUS_DIR"] = str(retrieval.get("corpus_dir", "data/knowledge"))
    env["AGENT0_RETRIEVAL_MAX_RESULTS"] = str(retrieval.get("max_results", 5))
    env["AGENT0_RETRIEVAL_SNIPPET_CHARS"] = str(retrieval.get("snippet_chars", 800))
    if backend == "remote":
        if not endpoint and not DRY_RUN:
            raise SystemExit(f"Remote sandbox selected; set {endpoint_env} in the environment")
        if endpoint:
            env["SANDBOX_ENDPOINT"] = endpoint
    else:
        env.pop("SANDBOX_ENDPOINT", None)
    return env


def python_executable(config: dict[str, Any]) -> str:
    value = Path(str(config["runtime"]["python"]))
    return str(value if value.is_absolute() else ROOT / value)


DRY_RUN = False


def run(command: list[str], *, env: dict[str, str] | None = None) -> int:
    if DRY_RUN:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def launch_rl(config: dict[str, Any], extra: list[str]) -> int:
    hydra = config["rl"]["hydra"]
    hydra["actor_rollout_ref"]["rollout"]["sandbox_timeout"] = config["sandbox"]["run_timeout_seconds"]
    overrides = [f"{key}={hydra_value(value)}" for key, value in flatten("", hydra)]
    command = [python_executable(config), "-m", "verl.trainer.main_ppo", "--config-name=agent0_trainer", *overrides, *extra]
    return run(command, env=sandbox_environment(config))


def launch_sft(config: dict[str, Any], stage: int, extra: list[str]) -> int:
    sft = dict(config["sft"])
    stage_values = config["sft"][f"stage{stage}"]
    sft.update(stage_values)
    command = [
        python_executable(config), "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={config['runtime']['nproc_per_node']}",
        "-m", "verl.trainer.agent0_sft_trainer",
        "--model_path", str(sft["model_path"]),
        "--train_data", str(sft["train_data"]),
        "--output_dir", str(sft["output_dir"]),
        "--experiment_name", str(sft["experiment_name"]),
        "--learning_rate", str(sft["learning_rate"]),
        "--num_train_epochs", str(sft["num_train_epochs"]),
        "--per_device_train_batch_size", str(sft["per_device_train_batch_size"]),
        "--gradient_accumulation_steps", str(sft["gradient_accumulation_steps"]),
        "--max_length", str(sft["max_length"]),
        "--warmup_ratio", str(sft["warmup_ratio"]),
        "--weight_decay", str(sft["weight_decay"]),
        "--max_grad_norm", str(sft["max_grad_norm"]),
        "--gradient_checkpointing", str(sft["gradient_checkpointing"]).lower(),
        "--dataloader_num_workers", str(sft["dataloader_num_workers"]),
        "--ulysses_sequence_parallel_size", str(sft["ulysses_sequence_parallel_size"]),
        "--use_remove_padding", str(sft["use_remove_padding"]).lower(),
        "--report_to_wandb", str(sft["report_to_wandb"]).lower(),
        "--seed", str(sft["seed"]),
    ]
    if sft.get("val_data"):
        command.extend(["--val_data", str(sft["val_data"])])
    return run(command + extra)


def launch_qlora_smoke(config: dict[str, Any], extra: list[str]) -> int:
    values = config["qlora_smoke"]
    required = ("model", "sft_data", "rl_data", "output_dir", "base_model_backup")
    missing = [key for key in required if not values.get(key)]
    if missing and not DRY_RUN:
        raise SystemExit("Set qlora_smoke." + ", qlora_smoke.".join(missing) + " in config.yaml")
    names = {
        "model": "--model", "sft_data": "--sft-data", "rl_data": "--rl-data",
        "output_dir": "--output-dir", "base_model_backup": "--base-model-backup",
        "epochs": "--epochs", "max_length": "--max-length", "min_pixels": "--min-pixels",
        "max_pixels": "--max-pixels", "gradient_accumulation_steps": "--gradient-accumulation-steps",
        "learning_rate": "--learning-rate", "weight_decay": "--weight-decay",
        "lora_rank": "--lora-rank", "lora_alpha": "--lora-alpha", "lora_dropout": "--lora-dropout",
        "device": "--device", "seed": "--seed", "save_every_steps": "--save-every-steps",
        "max_train_records": "--max-train-records", "resume_from": "--resume-from",
    }
    command = [python_executable(config), "tools/train_qlora_smoke.py"]
    for key, option in names.items():
        value = values.get(key)
        if value is not None:
            command.extend([option, str(value)])
    return run(command + extra)


def launch_serve_teacher(config: dict[str, Any], extra: list[str]) -> int:
    sft_gen = config.get("sft_data_generation", {})
    local_cfg = sft_gen.get("local_teacher", {})
    serve_cfg = local_cfg.get("serve", {})
    spec_cfg = local_cfg.get("speculative_decoding", {})

    model = str(local_cfg.get("model") or "qwen3.8-27b")
    host = str(serve_cfg.get("host") or "0.0.0.0")
    port = str(serve_cfg.get("port") or "8000")
    tp_size = str(serve_cfg.get("tensor_parallel_size") or config.get("runtime", {}).get("nproc_per_node", 4))
    gpu_mem = str(serve_cfg.get("gpu_memory_utilization") or 0.85)
    max_len = str(serve_cfg.get("max_model_len") or 32768)

    command = [
        python_executable(config), "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", port,
        "--tensor-model-parallel-size", tp_size,
        "--gpu-memory-utilization", gpu_mem,
        "--max-model-len", max_len,
        "--trust-remote-code",
    ]

    # Speculative Decoding parameters
    if spec_cfg.get("enabled", True):
        spec_model = spec_cfg.get("speculative_model")
        if spec_model:
            command.extend(["--speculative-model", str(spec_model)])
        num_spec = spec_cfg.get("num_speculative_tokens")
        if num_spec:
            command.extend(["--num-speculative-tokens", str(num_spec)])
        draft_tp = spec_cfg.get("speculative_draft_tensor_parallel_size")
        if draft_tp:
            command.extend(["--speculative-draft-tensor-parallel-size", str(draft_tp)])
        spec_len = spec_cfg.get("speculative_max_model_len")
        if spec_len:
            command.extend(["--speculative-max-model-len", str(spec_len)])
        disable_bs = spec_cfg.get("speculative_disable_by_batch_size")
        if disable_bs:
            command.extend(["--speculative-disable-by-batch-size", str(disable_bs)])

    return run(command + extra)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("rl", "sft-stage1", "sft-stage2", "qlora-smoke", "evaluate", "probe", "build-data", "build-rl", "build-sft", "serve-teacher", "preflight", "prepare-balanced", "reset-generation", "build-balanced-sft", "sft-local", "export-sft"))
    parser.add_argument("--dry-run", action="store_true", help="print the resolved command without starting it")
    parser.add_argument("--profile", choices=("local_4090", "local_4090_checkpointed"))
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    config = load_config(args.profile)
    global DRY_RUN
    DRY_RUN = args.dry_run

    if args.profile:
        from tools.local_workflows import launch_local
        return launch_local(config, args.action, extra, root=ROOT, dry_run=DRY_RUN)

    if args.action == "serve-teacher":
        return launch_serve_teacher(config, extra)
    if args.action == "build-sft":
        env = sandbox_environment(config)
        sft_gen = config.get("sft_data_generation", {})
        backend = str(sft_gen.get("backend", "local")).lower()

        # Check if overridden by CLI extra arguments
        if "--teacher-backend" in extra:
            idx = extra.index("--teacher-backend")
            if idx + 1 < len(extra):
                backend = extra[idx + 1].lower()

        if backend == "remote":
            remote_cfg = sft_gen.get("remote_teacher", {})
            key_name = str(remote_cfg.get("api_key_env", "AGENT0_RESPONSES_API_KEY"))
            api_key = env.get(key_name) or os.environ.get("AGENT0_RESPONSES_API_KEY", "")
            base_url = str(remote_cfg.get("base_url") or "https://api.openai.com/v1")
            model = str(remote_cfg.get("model") or "gpt-4o")
            timeout = str(remote_cfg.get("timeout_seconds") or 180)
            concurrency = str(remote_cfg.get("concurrency") or 8)
        else:
            # Default: Local Qwen 27B teacher model
            local_cfg = sft_gen.get("local_teacher", {})
            api_key = str(local_cfg.get("api_key") or "EMPTY")
            base_url = str(local_cfg.get("base_url") or "http://127.0.0.1:8000/v1")
            model = str(local_cfg.get("model") or "qwen3.8-27b")
            timeout = str(local_cfg.get("timeout_seconds") or 300)
            concurrency = str(local_cfg.get("concurrency") or 8)

            # Speculative decoding environment variables
            spec_cfg = local_cfg.get("speculative_decoding", {})
            if spec_cfg.get("enabled"):
                env["AGENT0_SPECULATIVE_ENABLED"] = "true"
                if spec_cfg.get("speculative_model"):
                    env["AGENT0_SPECULATIVE_MODEL"] = str(spec_cfg["speculative_model"])
                if spec_cfg.get("num_speculative_tokens"):
                    env["AGENT0_NUM_SPECULATIVE_TOKENS"] = str(spec_cfg["num_speculative_tokens"])
                if spec_cfg.get("method"):
                    env["AGENT0_SPECULATIVE_METHOD"] = str(spec_cfg["method"])

        env["AGENT0_RESPONSES_BASE_URL"] = env.get("AGENT0_RESPONSES_BASE_URL") or base_url
        env["AGENT0_RESPONSES_API_KEY"] = env.get("AGENT0_RESPONSES_API_KEY") or api_key
        env["AGENT0_RESPONSES_MODEL"] = env.get("AGENT0_RESPONSES_MODEL") or model
        env["AGENT0_RESPONSES_TIMEOUT_SECONDS"] = env.get("AGENT0_RESPONSES_TIMEOUT_SECONDS") or timeout
        env["AGENT0_CONCURRENCY"] = env.get("AGENT0_CONCURRENCY") or concurrency

        command = [python_executable(config), "-m", "scripts.build_sft_dataset", *extra]
        return run(command, env=env)
    if args.action == "rl":
        return launch_rl(config, extra)
    if args.action in {"sft-stage1", "sft-stage2"}:
        return launch_sft(config, 1 if args.action == "sft-stage1" else 2, extra)
    if args.action == "qlora-smoke":
        return launch_qlora_smoke(config, extra)
    if args.action in {"evaluate", "probe", "build-data"}:
        env = sandbox_environment(config)
        env.update(response_environment(config))
        if args.action == "evaluate":
            evaluation = config["evaluation"]
            details = config["evaluation_details"]
            command = [python_executable(config), "-m", "scripts.evaluate", "--benchmarks", str(evaluation["benchmarks"]),
                       "--output_dir", str(evaluation["output_dir"]), "--data_dir", str(evaluation["data_dir"]),
                       "--batch_size", str(evaluation["batch_size"]),
                       "--save_per_sample_results", str(details["save_per_sample_results"]).lower(),
                       "--save_intermediate", str(details["save_intermediate"]).lower(),
                       "--intermediate_save_interval", str(details["intermediate_save_interval"])]
        elif args.action == "probe":
            command = [python_executable(config), "-m", "scripts.probe_responses"]
        else:
            data = config["data_generation"]
            command = [python_executable(config), "-m", "tools.data_builder.smoke_build_local",
                       "--repo-root", str(data["repo_root"]), "--output", str(data["output"]),
                       "--samples-per-dataset", str(data["samples_per_dataset"]),
                       "--max-attempts", str(data["max_attempts"]), "--seed", str(data["seed"])]
        return run(command + extra, env=env)

    data = config["data_export"]
    command = [python_executable(config), "-m", "tools.data_builder.smoke_build_rl", "--input", str(data["input"]),
               "--output", str(data["output"]), "--samples-per-dataset", str(data["samples_per_dataset"])]
    if data.get("allow_eval_only"):
        command.append("--allow-eval-only")
    return run(command + extra)


if __name__ == "__main__":
    raise SystemExit(main())
