# Migration acceptance status

Status: **incomplete**. The copy and offline migration are implemented, but live
endpoint and GPU GRPO acceptance are pending. Do not use this as evidence that
an arbitrary vLLM/VL endpoint supports the complete Responses tool protocol.

## Passed locally

- `Agent0-VL` has no tracked Git changes; edits are confined to `Agent0-Vl-HJL`.
- Python 3.12.14 `.venv`, uv 0.12.20, and the full `requirements.lock` resolve.
- `uv pip check` reports all 254 installed packages compatible.
- RTX 4090, NVIDIA driver 595.91.07, glibc 2.35, PyTorch 2.13.0+cu132,
  vLLM 0.30.0, OpenAI SDK 3.20.0 import; `torch.cuda.is_available()` is true.
- Installed vLLM wheel tag `cp38-abi3-linux_x86_64` is supported by this
  Python/platform combination.
- Mock SDK tests cover text, image, function registration, two tool rounds,
  call ID pairing, error output, timeout/retry configuration, and probe failure.
- Canonical SFT JSONL and RL Parquet readers pass tests.
- Fake local vLLM engine test checks exact sampled IDs/logprobs/masks and the
  semantic trajectory; it does not prove GPU execution.
- Final source search found no old Chat Completions calls or legacy tool fields.

## Pending or unsupported

- No configured `AGENT0_RESPONSES_*` endpoint/model is available here. Run
  `.venv/bin/python -m scripts.probe_responses` against the target VL model;
  text, image, function calls, output handoff, and two successive rounds must
  all pass before using it.
- The four GPUs each had approximately 1.2–1.5 GiB free during final checks.
  A real VL rollout and small GRPO training smoke test were not run.
- `FSDPVLLMShardingManager` now calls the vLLM 0.30 worker `reload_weights`
  API through `LLM.collective_rpc`, but weight synchronization with the target
  checkpoint remains unverified on GPU.
- Canonical FSDP SFT currently handles text tokens. Image SFT needs a VL
  processor, image tensors, and a live training test. The dataset rejects image
  records in this text-only path instead of silently training placeholders.
- Optional Megatron code still contains FlashAttention-specific kernels. The
  active FSDP path uses PyTorch SDPA; Megatron has not been migrated or tested.

A failed live probe or GRPO smoke must keep this status incomplete until the
underlying issue is corrected and those checks pass.
