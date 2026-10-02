"""Resolved local profiles shared by command printing and execution."""
from __future__ import annotations

import copy
from pathlib import Path
import yaml


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(root, profile=None):
    with (Path(root) / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    profiles = config.pop("profiles", {})
    if profile:
        if profile not in profiles:
            raise ValueError(f"Unknown profile: {profile}")
        def resolve(name, seen=()):
            if name in seen or name not in profiles:
                raise ValueError(f"Invalid profile inheritance: {name}")
            values = copy.deepcopy(profiles[name])
            parent = values.pop('inherits', None)
            return merge(resolve(parent, (*seen, name)), values) if parent else values
        config = merge(config, resolve(profile))
        config["selected_profile"] = profile
    return config


def validate_local(config):
    if 'checkpointed' in config:
        from agent0_protocol.checkpointed import validate_config
        validate_config(config['checkpointed'], training=True)
    sft = config["sft_local"]
    rl = config["rl"]["hydra"]
    actor = rl["actor_rollout_ref"]["actor"]
    rollout = rl["actor_rollout_ref"]["rollout"]
    gpu_count = len(config["runtime"]["gpus"])
    if sft["tp"] * sft["pp"] != gpu_count:
        raise ValueError("local SFT GPU count must equal TP*PP")
    if sft["global_batch_size"] % sft["micro_batch_size"]:
        raise ValueError("SFT global batch must divide micro batch")
    if gpu_count % rollout["tensor_model_parallel_size"] or gpu_count % actor["ulysses_sequence_parallel_size"]:
        raise ValueError("GPU count must be divisible by rollout TP and Actor SP")
    if rollout["prompt_length"] + rollout["max_total_response_length"] > rollout["max_model_len"]:
        raise ValueError("Prompt + response buffer exceeds model window")
    batch = rl["data"]["train_batch_size"]
    dp = gpu_count // actor["ulysses_sequence_parallel_size"]
    if batch % gpu_count or batch < actor["ppo_mini_batch_size"]:
        raise ValueError("Invalid prompt/PPO batch for selected GPUs")
    if actor["ppo_mini_batch_size"] * rollout["n"] % (dp * actor["ppo_micro_batch_size_per_gpu"]):
        raise ValueError("PPO mini/micro batch mismatch")
    schedule = config["rl_schedule"]
    return {
        "sft_steps": None if 'checkpointed' in config else config["local_data"]["sft_rows"] * sft["epochs"] // sft["global_batch_size"],
        "warmup_steps": config["local_data"]["rl_rows_per_phase"] * schedule["warmup_epochs"] // batch,
        "formal_steps": config["local_data"]["rl_rows_per_phase"] * schedule["formal_epochs"] // batch,
        "trajectories_per_step": batch * rollout["n"],
    }
