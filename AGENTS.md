# Repository Guidelines

## Project Structure & Module Organization

This repository contains two projects. `Agent0/` implements the language agent, curriculum generation, executor training, and sandbox integration. `Agent0-VL/` is the vision-language project: `verl/` holds training, rollout, evaluation, and reward code; `sandbox/` provides local tool execution; `scripts/` contains launchers; and `tools/local_sft_builder/` contains additive local data utilities. Tests are under `Agent0-VL/tests/`, including `tests/local_sft_builder/`. Shared docs and figures are in `docs/` and `figs/`.

Datasets, checkpoints, logs, SQLite ledgers, run outputs, and API credentials are local artifacts and must remain outside Git-tracked source files.

## Build, Test, and Development Commands

For Agent0-VL, install dependencies with:

```bash
cd Agent0-VL
pip install -r requirements.txt
```

Verify the CPU sandbox with `python -c "import asyncio; from sandbox.internal_sandbox import parallel_sandbox; print(asyncio.run(parallel_sandbox(['print(40+2)'])))"`. Run tests with `python -m pytest -q tests` or focus on the local builder with `python -m pytest -q tests/local_sft_builder`. The optional ms-swift loader smoke test requires ms-swift.

Use `MODEL=... SFT_DATA=... bash scripts/sft_stage1.sh` for tool-use SFT and `MODEL=... SFT_DATA=... bash scripts/sft_stage2.sh` for math-code annealing. RL uses `MODEL_PATH=... ITERATION=1 bash scripts/rl-agent0.sh`; evaluation uses `bash scripts/evaluate.sh --model_path ... --benchmarks mathverse,mathvista,chartqa --output_dir ...`.

## Cross-Environment Development Workflow

Develop and edit source files on Windows in the sibling `D:\论文\Agent0\Agent0-dev` worktree. Only add missing local functionality; do not modify, delete, or rename author-provided source files. Run checks, commit, and push `dev` to `user`. On vcc, connect with `ssh vcc`, use `/mnt/d/Agent0-dev`, check out the pushed commit, and repeat Linux/GPU tests. The Qwen3.8-27B environment is `/mnt/d/qwen3.8-27B`; use `/mnt/d/qwen3.8-27B/.venv/bin/python` for backend checks. Keep `/mnt/d/Agent0` on upstream `main`. Never synchronize datasets, model weights, virtual environments, API keys, logs, SQLite ledgers, or run outputs through Git; keep artifacts local.

## Coding Style & Testing Guidelines

Use four-space Python indentation, `snake_case` for functions and variables, and `PascalCase` for classes and typed records. Keep shell scripts portable Bash with quoted variables. Add regression tests as `test_*.py` files and cover behavior changes before submitting them.

## Commit & Pull Request Guidelines

Use concise imperative commit subjects with an optional scope, for example `fix(agent0vl): handle empty observations`. PRs should explain the change, list executed tests, note environment or dependency changes, and confirm that no datasets, checkpoints, logs, secrets, or generated artifacts are included.
