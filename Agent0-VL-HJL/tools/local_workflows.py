"""Concrete local commands resolved from the selected profile and CLI."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
from datetime import datetime

from tools.local_profile import validate_local


def sft_command(root, config, args):
    s = config["sft_local"]
    python = str(root / s["python"])
    model = args.model or config["assets"]["base_model"]
    data = args.data or config["local_data"]["sft_output"]
    output = args.output or str(root / s["output_dir"] / datetime.now().strftime("sft_%Y%m%d_%H%M%S"))
    container = str(Path(output) / "canonical_swift.jsonl")
    validate = [python, "-m", "tools.local_sft", "--data", str(root / data), "--model", model,
                "--output", container, "--max-length", str(args.max_length or s["max_length"])]
    command = [str(root / ".venv-sft/bin/megatron"), "sft", "--model", model,
        "--dataset", container, "--template", "hjl_canonical_vl", "--external_plugins",
        str(root / "tools/swift_canonical_plugin.py"), str(root / "tools/training/megatron_save_control.py"),
        "--tuner_type", "lora", "--lora_rank", str(s["lora_rank"]), "--lora_alpha", str(s["lora_alpha"]),
        "--target_modules", "all-linear", "--torch_dtype", "bfloat16", "--freeze_vit", "true",
        "--freeze_aligner", "true", "--tensor_model_parallel_size", str(s["tp"]),
        "--pipeline_model_parallel_size", str(s["pp"]), "--sequence_parallel", "true",
        "--micro_batch_size", str(args.micro_batch_size or s["micro_batch_size"]),
        "--global_batch_size", str(args.batch_size or s["global_batch_size"]),
        "--num_train_epochs", str(args.epochs or s["epochs"]),
        "--max_length", str(args.max_length or s["max_length"]), "--truncation_strategy", "delete",
        "--lazy_tokenize", "true", "--strict", "true", "--packing", "false", "--split_dataset_ratio", "0",
        "--lr", str(s["learning_rate"]), "--lr_warmup_fraction", str(s["warmup_ratio"]),
        "--recompute_granularity", "full", "--recompute_method", "uniform", "--recompute_num_layers", "1",
        "--cross_entropy_loss_fusion", "true", "--cross_entropy_fusion_impl", "te", "--vit_gradient_checkpointing", "true",
        "--gradient_accumulation_fusion", "false", "--attention_backend", "auto",
        "--save_strategy", "steps", "--save_steps", str(s["save_steps"] if args.save_steps is None else args.save_steps),
        "--save_total_limit", "2", "--async_save", "false", "--no_save_optim", "false", "--no_save_rng", "false",
        "--dataloader_num_workers", str(s["workers"]), "--dataloader_prefetch_factor", "1",
        "--dataloader_pin_memory", "false", "--dataset_num_proc", "1", "--logging_steps", "1", "--output_dir", output]
    if 'checkpointed' in config:
        validate += ['--mode-limits', json.dumps({mode: config['checkpointed']['limits'][f'sft_{mode}_tokens']
                                               for mode in ('solve', 'repair', 'verify')})]
        position = command.index('--target_modules')
        command[position + 1:position + 2] = config['checkpointed']['adapters']['target_modules']
    return validate, command, output


def teacher_command(root, config):
    t = config["sft_data_generation"]["local_teacher"]
    serve = t["serve"]
    command = [str(root / config["runtime"]["python"]), "-m", "tools.local_teacher",
        "--model", config["assets"]["teacher_model"], "--served-model-name", t["model"],
        "--host", str(serve.get("host", "127.0.0.1")), "--port", str(serve.get("port", 8000)),
        "--tensor-parallel-size", str(serve["tensor_parallel_size"]),
        "--pipeline-parallel-size", str(serve["pipeline_parallel_size"]),
        "--max-model-len", str(serve["max_model_len"]), "--max-num-seqs", str(serve["max_num_seqs"]),
        "--max-num-batched-tokens", str(serve["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(serve["gpu_memory_utilization"]),
        "--dtype", str(serve["dtype"]), "--kv-cache-dtype", str(serve["kv_cache_dtype"]),
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--trust-remote-code"]
    spec = t["speculative_decoding"]
    if spec["enabled"] and spec["num_speculative_tokens"]:
        command += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": spec["num_speculative_tokens"]})]
    return command


def launch_local(config, action, extra, *, root, dry_run):
    if 'checkpointed' in config:
        from agent0_protocol.checkpointed import validate_config
        validate_config(config['checkpointed'], training=True)
        if action not in {'preflight', 'prepare-balanced', 'reset-generation', 'build-balanced-sft', 'build-rl', 'sft-local', 'export-sft', 'rl'}:
            raise SystemExit('Checkpointed training worker integration is incomplete. '
                             'Refusing fallback to legacy confidence rewards or merged-model training.')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument('--base-model')
    parser.add_argument('--adapter-bundle')
    parser.add_argument('--mode', choices=('solve', 'repair', 'verify'))
    parser.add_argument('--adapter', choices=('shared',), default='shared')
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--phase", choices=("full", "warmup", "serc"), default=config["rl_schedule"]["phase"])
    parser.add_argument("--resume-dir")
    parser.add_argument("--formal-from-checkpoint")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--gpus")
    args, overrides = parser.parse_known_args(extra)
    if 'checkpointed' in config and action in {'sft-local', 'export-sft'}:
        if args.mode:
            raise SystemExit('Checkpointed SFT/export trains one shared adapter; use --adapter shared, not --mode')
        config = copy.deepcopy(config)
        cp = config['checkpointed']
        config['sft_local']['max_length'] = max(cp['limits'][f'sft_{mode}_tokens']
                                                for mode in ('solve', 'repair', 'verify'))
        config['sft_local']['lora_rank'] = cp['adapters']['rank']
        config['sft_local']['lora_alpha'] = cp['adapters']['alpha']
        config['sft_local']['output_dir'] = cp['sft_output_root'] + '/shared'
        config['local_data']['sft_output'] = str(Path(config['local_data']['sft_output']).parent / 'roles' / 'shared.jsonl')
    plan = validate_local(config)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus or ",".join(map(str, config["runtime"]["gpus"])),
               PYTHONPATH=str(root) + os.pathsep + env.get("PYTHONPATH", ""),
               OMP_NUM_THREADS=str(config["runtime"]["omp_num_threads"]))
    python = str(root / config["runtime"]["python"])
    commands = []
    inventory = None
    if action in {"preflight", "prepare-balanced", "reset-generation", "build-balanced-sft", "build-rl"}:
        operation = {"preflight": "preflight", "prepare-balanced": "prepare", "reset-generation": "reset",
                     "build-balanced-sft": "sft", "build-rl": "rl"}[action]
        commands = [[python, "-m", "tools.data_builder.local_data", operation]]
        if 'checkpointed' in config:
            commands[0] += ['--profile', config['selected_profile']]
        if operation == "sft":
            from scripts.launch import sandbox_environment
            env.update(sandbox_environment(config))
            t = config["sft_data_generation"]["local_teacher"]
            env.update(AGENT0_RESPONSES_BASE_URL=env.get("AGENT0_RESPONSES_BASE_URL") or t["base_url"],
                       AGENT0_RESPONSES_MODEL=env.get("AGENT0_RESPONSES_MODEL") or t["model"],
                       AGENT0_RESPONSES_API_KEY=env.get("AGENT0_RESPONSES_API_KEY") or "EMPTY",
                       AGENT0_RESPONSES_TIMEOUT_SECONDS="300", AGENT0_RESPONSES_MAX_OUTPUT_TOKENS="0")
            if 'checkpointed' in config:
                env['AGENT0_RESPONSES_TIMEOUT_SECONDS'] = str(config['checkpointed']['limits']['request_timeout_seconds'])
    elif action == "serve-teacher":
        commands = [teacher_command(root, config)]
    elif action == "sft-local":
        validate, train, inventory = sft_command(root, config, args)
        commands = [validate] + ([] if args.preflight_only else [train + overrides])
        from tools.sft_environment import environment
        env = environment(root, env)
        env.update(NPROC_PER_NODE="4", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    elif action == "export-sft":
        from tools.sft_environment import environment
        env = environment(root, env)
        env['NPROC_PER_NODE'] = '4'
        if not args.model:
            raise SystemExit("export-sft requires --model CHECKPOINT")
        commands = [[str(root / ".venv-sft/bin/megatron"), "export", "--mcore_adapter", args.model,
                     "--to_hf", "true", "--merge_lora", "true", "--output_dir", args.output or args.model + "-merged", *overrides]]
        if 'checkpointed' in config:
            commands[0][commands[0].index('--merge_lora') + 1] = 'false'
            commands[0][commands[0].index('--output_dir') + 1] = args.output or args.model + '-adapter'
    elif action == "rl":
        if 'checkpointed' in config:
            base = args.base_model or args.model
            bundle = args.adapter_bundle or config['checkpointed'].get('adapter_bundle')
            if not base or not bundle:
                raise SystemExit('Checkpointed RL requires --base-model and --adapter-bundle')
            if args.formal_from_checkpoint:
                raise SystemExit('Checkpointed formal-from-checkpoint is not yet supported; use a compatible full-run resume')
            if args.batch_size is not None or args.epochs is not None or overrides:
                raise SystemExit('Tune checkpointed sampling and schedule in config.yaml')
            commands = [[python, '-m', 'tools.checkpointed_rl', '--base-model', base,
                '--adapter-bundle', bundle, '--profile', config['selected_profile'], '--phase', args.phase]]
            for name in ('resume_dir', 'output'):
                if getattr(args, name):
                    commands[0] += ['--' + name.replace('_', '-'), getattr(args, name)]
            if args.preflight_only:
                commands[0] += ['--preflight-only']
            if dry_run:
                commands[0] += ['--dry-run']
        else:
            if not args.model:
                raise SystemExit("RL requires --model /absolute/path/to/new_sft_merged")
            commands = [[python, "-m", "tools.local_rl", "--model", args.model, "--phase", args.phase]]
            for name in ("resume_dir", "formal_from_checkpoint", "output"):
                if getattr(args, name):
                    commands[0] += ["--" + name.replace("_", "-"), getattr(args, name)]
            for name in ('batch_size', 'epochs'):
                if getattr(args, name) is not None:
                    commands[0] += ['--' + name.replace('_', '-'), str(getattr(args, name))]
            commands[0] += overrides
            if args.preflight_only:
                commands[0] += ["--preflight-only"]
    else:
        raise SystemExit(f"{action} uses the generic entrypoint; omit --profile")
    if dry_run:
        print(json.dumps({"profile": config.get('selected_profile', 'local_4090'), "schedule": plan}, indent=2))
        for command in commands:
            print(shlex.join(command), flush=True)
        if action == 'rl':
            # Nested resolution is a CPU-only command print, never a Ray launch.
            preview_env = dict(env, CUDA_VISIBLE_DEVICES='')
            return subprocess.run(commands[0] + ['--dry-run'], cwd=root, env=preview_env).returncode
        return 0
    os.chdir(root)
    if inventory and not args.preflight_only:
        Path(inventory).mkdir(parents=True, exist_ok=False)
        (Path(inventory) / "README.md").write_text("Local LoRA SFT; status: launched (completion requires inspection).\n\n" +
            "\n".join(shlex.join(command) for command in commands) + "\n\nProfile: config.yaml/" + config.get("selected_profile", "local_4090") + ". GPU=" + env["CUDA_VISIBLE_DEVICES"] + "; BF16; LoRA16/64; TP4 PP1; context30720; global4/micro4; epochs3. Optimizer/RNG saved; HF export separate.\n")
    for command in commands:
        result = subprocess.run(command, cwd=root, env=env)
        if result.returncode:
            return result.returncode
    return 0
