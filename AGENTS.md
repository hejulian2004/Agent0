# Repository Guidelines

## Project Structure & Module Organization

This repository contains two related projects:

- `Agent0/` contains the language-agent implementation, curriculum/executor training code, and the vendored `executor_train/verl` framework.
- `Agent0-VL/` contains the vision-language pipeline, rollout and reward code under `verl/`, launch scripts under `scripts/`, the tool sandbox under `sandbox/`, and SFT-data utilities under `tools/sft_builder/`.
- `docs/` and `figs/` contain project documentation and publication/web assets. Keep datasets, checkpoints, API keys, and generated artifacts local; do not commit them.

Tests are concentrated in `Agent0/executor_train/verl/tests/` and `Agent0/executor_train/verl_tool/servers/tests/`.

## Build, Test, and Development Commands

Run commands from the component directory they target. For Agent0-VL, install the pinned dependencies in `Agent0-VL/requirements.txt` after selecting the appropriate PyTorch/CUDA build:

```bash
cd Agent0-VL
python -m venv .venv
.venv/bin/pip install -r requirements.txt
python -c "import asyncio; from sandbox.internal_sandbox import parallel_sandbox; print(asyncio.run(parallel_sandbox(['print(40+2)'])))"
bash scripts/sft_stage1.sh       # Stage-1 SFT
bash scripts/sft_stage2.sh       # Stage-2 SFT
bash scripts/rl-agent0.sh        # SERC/RL training
```

For the data pipeline, use `python -m tools.sft_builder.build_stream ...` and verify outputs with the supplied merge/audit tools. Run focused CPU tests with `cd Agent0/executor_train/verl && pytest -q tests/<path>`; full suites can require substantial GPU and distributed resources.

## Qwen3.8-27B Teacher Reference

See [`docs/qwen3.8-27b-launch.md`](docs/qwen3.8-27b-launch.md) for the verified vLLM launch profile. The local service uses `/mnt/d/qwen3.8-27B/start_qwen3_8_27b_vllm_local.sh`; its default is one sequence, while SFT generation requires an explicit 64-sequence profile. Never print or commit the API key.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and uppercase names for constants. Follow nearby code; no repository-wide formatter is configured, so keep imports, type hints, and line wrapping consistent with the surrounding file.

## Testing Guidelines

Name test files `test_*.py` and test functions `test_*`. Prefer the narrowest relevant CPU test first, then run sandbox smoke tests and GPU/distributed tests only when the change affects those paths. Record model, CUDA, and dependency versions for non-CPU results; no global coverage threshold is defined.

## Commit & Pull Request Guidelines

Existing history uses short, descriptive update subjects and merge commits. Use a concise imperative subject, keep unrelated changes separate, and explain data/model or configuration changes in the body. PRs should describe the motivation, affected component, exact tests/commands run, and hardware or environment assumptions; link an issue when applicable and include screenshots for documentation or visual changes. Never include secrets, local dataset paths that expose credentials, checkpoints, or large generated files.

## QLoRA and RL Runtime Notes

The verified RL profile uses `Agent0-VL/.venv`, bitsandbytes NF4 4-bit weights, BF16 compute/storage, double quantization, LoRA rank 8, alpha 32, and `target_modules=all-linear`. QLoRA freezes the quantized base, including the vision tower, and trains only adapters. Actor FSDP must use `use_orig_params=True`; `verl/workers/sharding_manager/fsdp_vllm.py` dequantizes BNB weights and merges LoRA weights before loading vLLM.

For the installed vLLM, pass `actor_rollout_ref.rollout.load_format=dummy_hf`: `hf` is rejected by the engine, while `dummy_hf` keeps the dummy loader and selects the full FSDP state needed for synchronization. Use `CUDA_VISIBLE_DEVICES=2,3`; GPU0/1 may belong to other jobs. Do not restart or modify the teacher service at `127.0.0.1:8000` unless explicitly requested.

RL is much slower than SFT: with `train_batch_size=8` and `rollout.n=8`, one step can generate 64 multi-turn trajectories, run tools/verifier/repair, compute reference log-probabilities, and update the actor. The current long-context profile takes roughly 11 minutes per step. A bitsandbytes warning about vision dimension 3420 not being divisible by 64 only selects a slower kernel; it is not an OOM or correctness failure. Prefer background runs with sparse, user-requested status checks.

## Current Validation Data

The current small validation set is kept under `Agent0-VL/data/`:

- `sft/validation/stage1_5_repaired_final_v2.jsonl`: 5 Stage-1 SFT rows.
- `sft/validation/stage2_5_repaired_final.jsonl`: 5 Stage-2 SFT rows.
- `rl/validation_10_rebuilt.parquet`: 10 RL rows; `validation_10_rebuilt.preview.jsonl` is its readable preview.

Both SFT files passed the repository's strict audit with no rejected or duplicate rows. Their schema is exactly `{messages, images}` with no `system` message; tool observations are user messages containing `[Code Execution Result]`. The RL Parquet schema is `prompt`, `images`, `reward_model`, `data_source`, and `extra_info`; all 10 embedded images decode successfully. Older candidates, failed outputs, and duplicate validation files were removed; the raw dataset was not modified. The retained manifests are archival metadata and their `input` fields may reference removed intermediate candidates.

Use `Agent0-VL/.venv/bin/python` for data checks and project tools; do not install or modify CUDA drivers. The teacher service is currently stopped. Start it only when explicitly needed, use GPUs 2 and 3 without interrupting other jobs, and stop it after generation.
