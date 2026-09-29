# Agent0-Vl-HJL

This is the Responses-only copy of `Agent0-VL`. All remote Solver, Verifier,
Repairer, dataset-generation, and evaluation calls use the OpenAI Python SDK
`client.responses.create(...)`. The original `Agent0-VL` directory is preserved.

## Environment

The pinned stack uses Python 3.12 and vLLM 0.30.0. The `requirements.lock` file
contains exact versions of all direct and transitive packages. Neither setup
nor runtime modifies the NVIDIA driver.

Before installing, check `nvidia-smi`, `ldd --version`, Python 3.12 availability,
and the vLLM wheel's Python/CUDA/PyTorch compatibility. This project was
resolved on four RTX 4090 GPUs, NVIDIA driver 595.91.07, glibc 2.35, Python
3.12.14, PyTorch 2.13.0+cu132, and CUDA runtime 13.2. If the pinned stable
vLLM wheel is incompatible with the target host, stop and diagnose; do not
silently downgrade it.

```bash
cd Agent0-Vl-HJL
uv python install 3.12
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --torch-backend=auto requirements.lock
```

The old `flash-attn` source extension has no matching wheel for this PyTorch
build and cannot compile against the mismatched CUDA toolkit headers on this
host. The active FSDP paths therefore use PyTorch SDPA and local tensor padding
helpers. All Python and CUDA runtime packages belong in `.venv`; `.python` and
`.venv` are ignored by Git.

## Responses endpoint

Edit the `responses` section in [config.yaml](/mnt/d/Agent0-dev-hjl/Agent0-Vl-HJL/config.yaml)
for the endpoint, model, timeout, retry count, and tool limits. Put only the key
in `AGENT0_RESPONSES_API_KEY`.

```bash
export AGENT0_RESPONSES_API_KEY=your-key
.venv/bin/python -m scripts.launch --dry-run probe
.venv/bin/python -m scripts.launch probe
```

The runtime probes text, image, strict function tools, two consecutive function
calls, `function_call_output`, and continuation before accepting the endpoint.
An unsupported endpoint fails at startup. The probe uses the configured model,
so test it after starting your vLLM Responses server and loading the actual VL
checkpoint. The server must support `/v1/responses`; this project has no other
remote inference path.

Tool execution is configured by the `sandbox` section. Local subprocess mode
applies wall-clock, CPU, memory, output-size, and concurrency limits. It does
not isolate filesystem access or networking; for untrusted generated code,
configure `backend: remote` and provide the SandboxFusion service URL using the
environment variable named by `endpoint_env`. Remote mode fails if its endpoint
or HTTP client dependency is missing and never falls back to local execution.
The current host does not allow Bubblewrap namespaces and has no running Docker
daemon, so those OS/container isolation modes are unavailable here.

The Python sandbox preloads the packages listed in `sandbox.preload_packages`.
By default, generated code can use `math`, `np` (NumPy), `Image` (Pillow),
`cv2` (OpenCV), `sp` (SymPy), and `RapidOCR` without import statements. Other
packages must be installed in `.venv` and imported explicitly; the tool does
not install packages dynamically. The YOLO detector runs through its dedicated
registered handler in the main `.venv` process because PyTorch needs more
memory than the restricted generic code sandbox allows.

## Registered tools

The canonical registry exposes `python_exec`, `crop_image`, `zoom_image`,
`rotate_image`, `ocr`, `plot_parser`, `visual_analyzer`, `object_detector`, and
`retrieve`. OCR uses RapidOCR's bundled PP-OCRv6 models. Object detection uses
the local COCO YOLO26n model at `tool_runtime.detector.model_path` and runs on
CPU by default. Image tools use the request's input image automatically and do
not take an `image_path` argument. Crop, resize, and rotate update the active
image for the next tool round, so later OCR, chart parsing, or detection sees
that intermediate image. Calls emitted together use the same image snapshot;
the last successful transform becomes active for the following round.
Retrieval searches text files placed under `tool_runtime.retrieval.corpus_dir`.
Chart parsing returns OCR labels and their image locations; it does not estimate
the plotted numeric series.

## Protocol and data

`agent0_protocol/` owns the semantic contract. A trajectory stores
`schema_version`, `trajectory_id`, the exact model-visible `tools` definitions,
ordered `items`, optional raw `rollout`, and `metadata`. Item types are `message`,
`reasoning`, `function_call`, and `function_call_output`. Arguments and outputs
remain JSON objects in Python. The Responses adapter serializes function output
at the HTTP boundary and parses function arguments on receipt. The registry in
`agent0_protocol/tools.py` provides schemas and handlers for API requests,
local Qwen generation, datasets, and verification.

The Qwen adapter in `agent0_protocol/adapters.py` owns chat-template and
model-token conversion. Runtime and verifier use semantic items. SFT and RL
read canonical trajectories. GRPO alone uses the local vLLM engine; each
training trajectory additionally stores actual sampled `response_token_ids`,
`old_logprobs`, masks, and policy/model version metadata. Its semantic items
are used for verification and logging. They do not reconstruct sampled tokens.

The current FSDP SFT trainer trains text tokens from canonical trajectories.
It does not pass image tensors through the VL model; image SFT still needs
implementation and a live GPU check before that path can be accepted.

## Configure and launch

Training, evaluation, Responses endpoint, data generation, and rollout
hyperparameters are in the project-root `config.yaml`. It contains the full
resolved RL/PPO configuration, including values that previously came from
VERL defaults, plus SFT stage settings and the optional QLoRA smoke parameters.
The shell scripts use that file through `scripts.launch`; optional trailing
arguments can override Hydra values for one run. Keep API credentials in the
environment rather than this file.

```bash
# Inspect the resolved command without starting a job
.venv/bin/python -m scripts.launch --dry-run rl

# GRPO, SFT stages, and evaluation
scripts/rl-agent0.sh
scripts/sft_stage1.sh
scripts/sft_stage2.sh
scripts/evaluate.sh
# Optional QLoRA smoke workflow after setting its required paths in config.yaml
.venv/bin/python -m scripts.launch qlora-smoke
```

The RL base structure remains in `verl/trainer/config/agent0_trainer.yaml`;
the launcher passes every resolved RL/PPO setting from `config.yaml` as an
explicit Hydra override.

## Validation

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m compileall -q agent0_protocol tools/data_builder verl
```

For live acceptance, configure the endpoint and a supported VL model, then run
`.venv/bin/python -m scripts.launch probe` to execute the startup probe.
Run dataset generation and evaluation with the same environment. A real GRPO
smoke run additionally requires enough free GPU memory for the training and
vLLM model. Mock tests prove protocol behavior but cannot establish endpoint
capability or model training compatibility.
