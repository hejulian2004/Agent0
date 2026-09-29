# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`Agent0-VL` is the **Responses-only** version of `Agent0-VL`. All remote Solver, Verifier, Repairer, data generation, and evaluation calls use the OpenAI Python SDK Responses API (`client.responses.create(...)`). Legacy Chat Completions endpoints and tools are not supported.

### Pinned Environment & Hardware Constraints
- **Python**: 3.12 (managed via `uv`) in `.venv/bin/python`.
- **Core Stack**: vLLM 0.30.0, PyTorch 2.13.0+cu132, CUDA 13.2 runtime, glibc 2.35, NVIDIA driver 595.91.07 (resolved on 4x RTX 4090 GPUs).
- **vLLM Compatibility**: Never downgrade vLLM. If the pinned wheel in `requirements.lock` fails on a host, diagnose environment issues directly.
- **Attention Backend**: FlashAttention source extensions are incompatible with CUDA toolkit headers on this host. All active FSDP paths use PyTorch SDPA (Scaled Dot-Product Attention) and local tensor padding helpers (`verl/utils/attention_padding.py`).
- **Secrets & Credentials**: Never write API keys or endpoints into `config.yaml` or source files. Pass them via environment variables (e.g., `AGENT0_RESPONSES_API_KEY`, `AGENT0_SANDBOX_ENDPOINT`).

---

## Development & Operational Commands

### Environment Setup
```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --torch-backend=auto requirements.lock
```

### Testing & Validation
```bash
# Run all unit tests
.venv/bin/python -m pytest tests

# Run a specific test file
.venv/bin/python -m pytest tests/test_agent0_protocol.py

# Run a single test method
.venv/bin/python -m pytest tests/test_agent0_schema.py::test_schema_version_matches_dataset_protocol

# Syntax and bytecode compilation check
.venv/bin/python -m compileall -q agent0_protocol tools/data_builder verl
```

### Unified Launcher (`scripts/launch.py` and Wrappers)
`scripts/launch.py` reads `config.yaml` and executes tasks with appropriate environment variables and Hydra overrides:

```bash
# Dry-run command resolution (prints command without running)
.venv/bin/python -m scripts.launch --dry-run [action]

# Probe remote Responses endpoint (checks text, image, strict tools, and multi-turn continuation)
export AGENT0_RESPONSES_API_KEY=your-key
.venv/bin/python -m scripts.launch probe

# RL / GRPO Training (passes resolved config.yaml parameters as Hydra overrides to verl.trainer.main_ppo)
scripts/rl-agent0.sh
# Or with specific Hydra overrides:
.venv/bin/python -m scripts.launch rl trainer.total_epochs=2

# Supervised Fine-Tuning (SFT)
scripts/sft_stage1.sh
scripts/sft_stage2.sh

# Evaluation on multimodal benchmarks
scripts/evaluate.sh

# HJL (Hierarchical Judgment Loop) Execution
.venv/bin/python -m hjl.run --mock --mode hjl --max-steps 8
.venv/bin/python -m hjl.run --image data/example.png --mode direct
.venv/bin/python -m hjl.run --image data/example.png --mode react
.venv/bin/python -m hjl.run --image data/example.png --mode react_verifier

# Data building (single-turn smoke builder and RL Parquet exporter)
.venv/bin/python -m scripts.launch build-data
.venv/bin/python -m scripts.launch build-rl

# Standalone QLoRA smoke fine-tuning (requires configured paths in config.yaml)
.venv/bin/python -m scripts.launch qlora-smoke
```

---

## Architecture & Code Organization

### 1. Unified Configuration (`config.yaml`)
Single source of truth for all workflows:
- `responses`: API URL, model name, timeouts, retry limits, tool round caps.
- `sandbox`: Backend selection (`local_subprocess` or `remote`), CPU/RAM/output limits, and preloaded package allowlist.
- `tool_runtime`: Local detector weights (`yolo26n.pt`) and local retrieval corpus directory.
- `sft`: Model path, hyperparameters, and stage-specific overrides (`stage1`, `stage2`).
- `rl.hydra`: Complete resolved PPO/GRPO actor, critic, reward model, rollout, and trainer parameters for VERL.

### 2. Protocol Layer (`agent0_protocol/`)
Implements the canonical format bridging external Responses API calls, internal dataset representation, and VERL RL training:
- **`schema.py`**: Defines `CanonicalTrajectory` (`schema_version: agent0.responses.v1`) and `RawRollout`. Trajectories enforce strict item sequences (`message`, `reasoning`, `function_call`, `function_call_output`). `RawRollout` stores actual sampled vLLM token IDs, masks, and logprobs without text reconstruction.
- **`tools.py`**: Global `ToolRegistry` managing 9 canonical tools: `python_exec`, `crop_image`, `zoom_image`, `rotate_image`, `ocr`, `plot_parser`, `visual_analyzer`, `object_detector`, and `retrieve`. Controls `ToolExecutionContext` for stateful multi-turn image transformations with sibling call isolation, snapshot checkpointing, and clean image rollback on verifier self-repair.
- **`adapters.py`**:
  - `ResponsesAdapter`: Serializes/deserializes between OpenAI SDK wire objects and canonical trajectory items.
  - `QwenModelAdapter`: Maps between vLLM token sequences and Qwen special tokens (`<tool_call>`, `<think>`) without lossy decoding/re-encoding.
- **`responses_runtime.py`**: Handles live Responses endpoint communication and enforces fail-fast capability probing on startup.
- **`verifier.py`**: Semantic validation of trajectories, call/output pairing, tool definitions, final answer extraction, and `retry_function`/`repair_function` supporting image rollback.

### 3. Execution Sandboxes (`sandbox/`)
- **`subprocess_sandbox.py`**: Local subprocess isolation enforcing strict wall-clock, CPU-time, memory (default 1024MB), and output byte caps. Automatically preloads trusted packages (`math`, `np`, `Image`, `cv2`, `sp`, `RapidOCR`). Runs YOLO detection outside the restricted sandbox in the parent process due to PyTorch memory constraints.
- **Remote Sandbox**: When `sandbox.backend: remote`, connects to a SandboxFusion service via `AGENT0_SANDBOX_ENDPOINT`.

### 4. Training Engine (`verl/`)
Built upon VERL (Volcano Engine Reinforcement Learning):
- **RL / GRPO (`verl/trainer/main_ppo.py`, `verl/trainer/ppo/ray_trainer.py`)**: Ray-based distributed PPO/GRPO training.
- **Agent0 SPMD Rollout (`verl/workers/rollout/vllm_rollout/vllm_agent0_rollout_spmd.py`)**: Executes tool-in-the-loop rollouts directly within vLLM worker processes, invoking tools via the local sandbox. Pre-step image context is checkpointed and rolled back when Verifier confidence triggers Self-Repair (`action: PATCH`).
- **Reward Management (`verl/workers/reward_manager/agent0.py`)**: Combines mathematical answer verification (`math-verify`), tool efficiency penalties, and step verification scores.
- **FSDP SFT (`verl/trainer/agent0_sft_trainer.py`)**: Reads canonical trajectories from Parquet datasets. Note: Current FSDP SFT trains text tokens only and deliberately rejects image records.
- **vLLM Weight Sync (`verl/workers/sharding_manager/fsdp_vllm.py`)**: Uses `LLM.collective_rpc` to trigger vLLM 0.30's `reload_weights` across workers during RL iterations.

### 5. Hierarchical Judgment Loop (`hjl/`)
Industrial visual anomaly detection orchestrator:
- **`graph.py`**: Lightweight, self-contained `StateGraph` (zero unpinned dependencies) compiling the HJL inspection loop.
- **`state.py`**: Explicit shared state `HJLState`, `HJLPhase` (`GLOBAL_DISCOVERY`, `HYPOTHESIS_INSPECTION`, `EVIDENCE_RESOLUTION`), and persistent `EvidenceState` storing atomic `EvidenceItem` records (`SUPPORT`, `CONTRADICT`, `NEUTRAL`).
- **`taxonomy.py` & `routing.py`**:
  - Three strictly decoupled checkpoints: Global (`CONFIRMED_NORMAL`, `PASS`, `FAIL`), Regional (`PASS`, `FAIL`), Evidence (`PASS` with `ANOMALY`/`NORMAL`, `FAIL` with `UNRESOLVED`).
  - 8 failure types (7 semantic + deterministic `TOOL_FAILURE`).
  - `FailureRoutingPolicy` mapping failure modes to abstract action masks (`ActionType`).
- **`tools_adapter.py`**: Adapts canonical tools (`crop_image`, `zoom_image`, `rotate_image`, `retrieve`, `visual_analyzer`) to standardized `ToolResult`.
- **`trajectory.py`**: Exports execution history to `.jsonl` and validates `CanonicalTrajectory` (`agent0.responses.v1`).
- **`engine.py`**: Unified runner supporting `direct`, `react`, `react_verifier`, and `hjl` modes.

