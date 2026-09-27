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
bash scripts/build-balanced-sft-with-teacher.sh  # Four-GPU teacher; SFT data then RL data
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft-agent0-2x4090-mixed.sh  # Single-stage SFT
bash scripts/rl-agent0-2x4090-optimized.sh   # Two-GPU SERC/RL training
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

The verified RL profile uses `Agent0-VL/.venv`, bitsandbytes NF4 4-bit weights, BF16 compute/storage, double quantization, LoRA rank 8, alpha 32, and the language-side q/k/v/o/gate/up/down projection targets in `agent0_trainer.yaml`. SFT uses `target_modules=all-linear`. QLoRA freezes the quantized base, including the vision tower, and trains only adapters. Actor FSDP must use `use_orig_params=True`; `verl/workers/sharding_manager/fsdp_vllm.py` dequantizes BNB weights and merges LoRA weights before loading vLLM.

For the installed vLLM, pass `actor_rollout_ref.rollout.load_format=dummy_hf`: `hf` is rejected by the engine, while `dummy_hf` keeps the dummy loader and selects the full FSDP state needed for synchronization. Use `CUDA_VISIBLE_DEVICES=2,3`; GPU0/1 may belong to other jobs. Do not restart or modify the teacher service at `127.0.0.1:8000` unless explicitly requested.

RL is much slower than SFT: rollout work scales with `train_batch_size * rollout.n`, and every trajectory can run multiple Solver/tool/Verifier/Repair turns before reference log-probabilities and the actor update. The current two-GPU wrapper uses train batch 2 and `n=8` (16 trajectories per step); older batch-8 runs generated 64. A bitsandbytes warning about vision dimension 3420 not being divisible by 64 only selects a slower kernel; it is not an OOM or correctness failure. Prefer background runs with sparse, user-requested status checks.

## Current Balanced Light Reproduction (2026-09-27)

The user requests one mixed SFT run, without Stage 1 / Stage 2 separation. Keep the released `main` data formats, prompts, answer extraction and training rewards. Preserve the local QLoRA/FSDP/vLLM and CPU-offload adaptations for RTX 4090 hardware. The source balancing, reduced data volume, single-stage curriculum and Qwen3.8-27B teacher are explicit light-reproduction choices, not the paper's complete setup.

### SFT and RL Sources

SFT uses nine principal paper sources: Geometry3K, GeoQA, Mulberry, LLaVA-OV-Image, MM-RLHF, SMR, MM-Eureka, ReTool and arXivQA. Geometry3K contributes 112 final rows; each other source contributes 111, totaling 1,000. The previously discussed 11 downloaded training sources include ChartQA and ThinkLite, which remain in the current RL pool. The paper also mentions additional math/chart SFT sources; do not describe this nine-source subset as its complete source coverage.

The balanced candidate manifest is `Agent0-VL/data/sft/balanced_1000_v2/manifest.json`; the final SFT output is `data/sft/large/mixed_balanced_1000.jsonl`. Candidates were prepared and their image references checked on 2026-09-26. Final teacher trajectories were not generated by the agent; check the user's subsequent run before claiming completion. SMR's `arxivqa/images/...` references now resolve to the downloaded arXivQA image archive. Teacher failures are replaced by later candidates from the same source; source quotas must still hold after filtering.

RL uses the available local training splits of ChartQA, arXivQA and ThinkLite. Each 200-row phase contains 28, 101 and 71 rows respectively, proportional to downloaded training counts. The paper lists six RL sources; local MathVerse, MathVista and WeMath copies are evaluation artifacts and are excluded from this training pool. A shared source such as arXivQA is allowed, but selected SFT, RL and validation questions must not overlap. Warm-up and formal RL must also be disjoint. `tools/build_multisource_rl.py` excludes the final balanced SFT file, so generate RL after SFT succeeds.

### User-Run Four-GPU Teacher Generation

Run from `Agent0-VL/`:

```bash
# Candidates are already prepared; rerun only when deliberately rebuilding them.
bash scripts/rebuild-balanced-data.sh prepare
# Start teacher, generate/audit SFT, then build RL; does not train the student.
bash scripts/build-balanced-sft-with-teacher.sh
```

The teacher wrapper defaults to GPUs `0,1,2,3`, TP=4, 64 concurrent builder requests, `max_num_seqs=64`, context 24576, 16384 batched tokens, BF16 weights, FP8 KV cache, GPU memory utilization 0.90 and request timeout 600 seconds. It serves `/mnt/d/qwen3.8-27B/model` as `qwen3.8-27b` on port 8000 with the external teacher virtualenv. It checks selected GPUs for occupancy before starting its own service and validates the served model identity. An existing healthy service is reused; its GPU/TP profile is not reconfigured or verified by the wrapper. Never expose the API key.

Progress appears in the terminal; each run has `logs/balanced_sft_build_<timestamp>/generation.log`, `teacher.log` and `README.md`. Generation resumes per-source state files; changed source inputs, system prompt or recorded generation configuration are rejected on resume. Cleanup terminates builder process groups and only the teacher the wrapper started. The user may lower `TEACHER_CONCURRENCY` explicitly. The four-GPU generation profile was syntax-checked but was not GPU-tested by the agent.

For an already running teacher use `bash scripts/rebuild-balanced-data.sh sft`. Use `bash scripts/rebuild-balanced-data.sh rl` to rebuild only RL after the final SFT file exists. Do not regenerate candidates during an active build or silently reuse old trajectories after changing prompts.

### Main Format and Parsing Requirements

SFT exports exactly `{messages, images}`. Messages have `role` and `content`; no stored system message is allowed. Image markers must match image counts. Tool observations are user messages containing `[Code Execution Result]`. Training and teacher generation use the released `scripts/prompt.txt`; Verifier/Repair injections follow the executable `main` rollout. The paper's prose mentions think tags and JSON tool plans, while released main uses fenced Python through the SFT prompt; follow executable main and document that discrepancy rather than mixing protocols.

Final-answer extraction accepts main's `\boxed{...}` and `FINAL_ANSWER:` forms. Locally added answer-tag-only and plain-answer fallbacks were removed. Code, answer, Verifier JSON and Repair JSON extraction were compared with main. Both the SERC and external warm-up reward managers use `verl/utils/reward_score/math_verify.py` for correctness, including its 300-character solution window; do not substitute local benchmark-specific scoring for training. This preserves main's limitations on free-text answers. Teacher trajectories are filtered for successful tool execution and verifier quality; the current strict balanced build also requires a reliable source reference match. Reference answers are never sent in teacher requests.

RL Parquet fields are `prompt`, `images`, `reward_model`, `data_source`, `extra_info`: user question/image markers, embedded image bytes, and `reward_model.ground_truth`. Final SFT and RL outputs must pass their schema, image, quota and duplicate checks before training. Related repair/batching CPU tests passed (11 tests), and main correctness-scoring smoke checks passed on 2026-09-26; no new student training run was performed.

### Current Single-Stage SFT Training

```bash
PREFLIGHT_ONLY=1 bash scripts/sft-agent0-2x4090-mixed.sh
# Default two-GPU run on 2,3:
bash scripts/sft-agent0-2x4090-mixed.sh
# User-requested four-GPU alternative:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft-agent0-2x4090-mixed.sh
# After SFT succeeds, two-GPU warm-up then formal RL:
bash scripts/rl-agent0-2x4090-optimized.sh
```

The mixed launcher reads `data/sft/large/mixed_balanced_1000.jsonl` directly, audits exactly 1,000 rows, and supports two or four processes. Defaults: local Qwen2.5-VL-7B-Instruct base, 3 epochs, lr `1e-5`, warm-up ratio 0.05, per-device batch 1, global batch 128 (accumulation 64 on two GPUs or 32 on four), max length 10240, QLoRA NF4/BF16 compute and storage with double quantization, rank 16/alpha 64, all-linear adapters, frozen vision tower/aligner, gradient checkpointing and ZeRO-3. Lowering `MAX_LENGTH` explicitly changes truncation behavior. The four-GPU alternative and restored context limit have not been validated by a new GPU training run.

Every run creates a fresh `checkpoints/sft_2x4090/sft_2x4090_mixed_<timestamp>/` with `mixed_adapter/`, `mixed_merged/`, `training.log` and inventory README. Progress logs every training step; interruption cleans up the training process group. Successful export updates `checkpoints/sft_2x4090/latest_mixed_merged`, which the RL wrapper loads by default. Current RL defaults to two GPUs: warm-up 3 epochs / 300 steps, then formal SERC 1 epoch / 100 additional steps, global batch 2 and rollout n=8. Both training launchers now accept `--gpus`, `--batch-size`, dataset/model/output options and `--dry-run`. SFT accepts `--grad-accum`, `--epochs`, `--max-length`, `--learning-rate` and `--workers`/`--concurrency` (CPU dataloader workers). RL accepts `--tp`, `--concurrency`/`--max-num-seqs`, `--rollout-n`, actor mini/micro batches, phase epochs/step caps and context limits. RL TP defaults to selected GPU count and must divide it. Batch/rollout/actor combinations must satisfy FSDP divisibility. Step defaults use floor(Parquet row count / global batch) times phase epochs; overlong-prompt filtering can reduce the actual dataloader horizon. Four-GPU RL is configurable but has not been GPU-validated. `--steps` is the additional formal phase cap; keep epoch count large enough to supply those steps. Do not substitute a stale Stage-2 export or automatically start training after data generation.

## Legacy Sequential SFT Launch Memory

This retained launcher belongs to the earlier sequential experiment. The current user-requested workflow uses the mixed launcher documented above. Run the sequential workflow from `Agent0-VL/` using `scripts/sft-agent0-2x4090-full.sh`:

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/sft-agent0-2x4090-full.sh
```

It runs Stage 1 and then Stage 2 serially; Stage 2 starts only after Stage 1 succeeds and consumes that run's `stage1/last` adapter. The defaults are `CUDA_VISIBLE_DEVICES=2,3`, two processes, QLoRA NF4 with BF16 compute/storage and double quantization, LoRA rank 16/alpha 64 with `target_modules=all-linear`, frozen vision tower and aligner, DeepSpeed ZeRO-3, per-device batch 1, gradient accumulation 64 (global batch 128), and `max_length=4096`. It uses `Agent0-VL/.venv/bin/python` and `.venv/bin/swift` directly and validates both datasets before loading the model. The default datasets are `data/sft/large/stage1_500_local.jsonl` and `data/sft/large/stage2_500.jsonl`.

Use a no-training preflight when changing data or environment:

```bash
PREFLIGHT_ONLY=1 bash scripts/sft-agent0-2x4090-full.sh
```

Each full run creates `checkpoints/sft_2x4090/sft_2x4090_<timestamp>/` containing `stage1/`, `stage2/`, `training.log`, and a run-local `README.md` inventory. The final adapters are exposed through `stage1/last` and `stage2/last`. The base model defaults to the verified project-local copy at `checkpoints/base/Qwen2.5-VL-7B-Instruct`; the launcher checks its config, weight index, and all five safetensors shards before starting, so it does not silently wait for a remote download. Override the GPU pair or base model with environment variables, for example `CUDA_VISIBLE_DEVICES=0,1 BASE_MODEL=/path/to/model bash scripts/sft-agent0-2x4090-full.sh`.

The full launcher uses the paper SFT learning rate `1e-5` for both stages and exports the Stage-2 adapter to a merged model at `stage2_merged/`; it also updates `checkpoints/sft_2x4090/latest_stage2_merged`, which remains available through an explicit RL `MODEL_PATH` override. The optimized RL launcher defaults to the mixed SFT export at `checkpoints/sft_2x4090/latest_mixed_merged`.

For separate stages, use `scripts/sft_stage1.sh` and `scripts/sft_stage2.sh`. Stage 2 first looks for `checkpoints/sft_stage1/last`; if absent, it falls back to the retained merged Stage-1 model at `checkpoints/paper_500/sft_stage1_qlora_merged`. Both scripts print progress every training step (`logging_steps=1`). Persistent combined logging is provided by the one-click script.

## Two-GPU RL Launch Memory

The standard two-4090 RL entry point is `Agent0-VL/scripts/rl-agent0-2x4090-optimized.sh`. By default it uses `data/rl/rl_warmup_200_multisource.parquet` for the three-epoch external-correctness warm-up, saves the warm-up FSDP checkpoint, and resumes the formal SERC phase on `data/rl/rl_200_multisource.parquet` in the same run directory. Run it from `Agent0-VL/`:

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/rl-agent0-2x4090-optimized.sh
```

Use `SKIP_WARMUP=1` to start SERC directly. The current multisource datasets are generated by `tools/build_multisource_rl.py`; each phase contains 200 rows, and the warm-up and formal sets are disjoint. `tools/prepare_rl_warmup.py` and the non-multisource filenames belong to the older workflow.

This launcher accepts command-line options; run `--help` for the complete list and `--dry-run` to inspect settings without creating a run. Options override environment variables. It defaults to physical GPUs 2 and 3 and refuses to start if either selected GPU already uses more than 512 MiB. It checks the virtualenv, datasets, trainer config, launch script, model config, and `nvidia-smi` before starting. Any error stops the run immediately, and there is no automatic retry. Interrupt handling terminates both training and the progress monitor.

The current default profile uses `data/rl/rl_warmup_200_multisource.parquet` for the 300-step external-correctness warm-up, then `data/rl/rl_200_multisource.parquet` for 100 SERC steps, with validation metadata from `data/rl/validation_10_rebuilt.parquet`. It defaults to the mixed SFT export at `checkpoints/sft_2x4090/latest_mixed_merged`; set `MODEL_PATH` to use another model. It uses global train batch 2, GRPO rollout `n=8`, TP=2, `max_num_seqs=16` (experimental concurrency cap; the prior profile used 6, and actual concurrency may be lower due to memory and token budgets), `max_model_len=max_num_batched_tokens=9216`, maximum total response length 3072, vLLM `gpu_memory_utilization=0.65`, vLLM CPU offload disabled, actor micro-batch 1, save frequency 5, and all validation/testing disabled. The trainer config supplies `load_format=dummy_hf`, actor/optimizer offload, and the required FSDP/QLoRA settings. Old/reference log-probability batches and actor updates stay on CPU until each micro-batch is computed; vLLM synchronization fails fast if any model parameters remain unloaded. Resume rejects a run that used a different SFT model.

The launcher writes live terminal output to `logs/<run_name>.log` and emits a progress heartbeat every 30 seconds with elapsed time, inferred stage, rollout worker count, GPU memory/utilization, host RAM, worker RSS, and the latest trainer/rollout line. Check progress with `tail -f logs/<run_name>.log`. Checkpoints are written under `checkpoints/paper_500/<run_name>/`. `scripts/rl-agent0.sh` is the lower-level Hydra launcher used by the optimized wrapper; prefer the optimized wrapper for the standard two-4090 run.

## Current Validation Data

The current small validation set is kept under `Agent0-VL/data/`:

- `sft/validation/stage1_5_repaired_final_v2.jsonl`: 5 Stage-1 SFT rows.
- `sft/validation/stage2_5_repaired_final.jsonl`: 5 Stage-2 SFT rows.
- `rl/validation_10_rebuilt.parquet`: 10 RL rows; `validation_10_rebuilt.preview.jsonl` is its readable preview.

Both SFT files passed the repository's strict audit with no rejected or duplicate rows. Their schema is exactly `{messages, images}` with no `system` message; tool observations are user messages containing `[Code Execution Result]`. The RL Parquet schema is `prompt`, `images`, `reward_model`, `data_source`, and `extra_info`; all 10 embedded images decode successfully. Older candidates, failed outputs, and duplicate validation files were removed; the raw dataset was not modified. The retained manifests are archival metadata and their `input` fields may reference removed intermediate candidates.

Use `Agent0-VL/.venv/bin/python` for data checks and project tools; do not install or modify CUDA drivers. The agent-started teacher and generation jobs were stopped at the user’s request on 2026-09-26. Check live processes before assuming the service is still stopped. The user will run generation and training scripts themselves; do not start them automatically. The current teacher-generation script defaults to GPUs 0,1,2,3; RL defaults to GPUs 2,3. Do not interrupt unrelated jobs.

## Model and Log Inventory Requirements

Every new model output, Hydra run directory, or training log under `Agent0-VL/checkpoints/`, `Agent0-VL/outputs/`, or `Agent0-VL/logs/` must be registered in the nearest `README.md`. Record the path, filesystem creation time, last-write time when relevant, status (`running`, `success`, or `failed`), base model, dataset, GPU/precision/quantization setup, LoRA or QLoRA settings, context and batch limits, and the command or configuration source. Update the inventory when a run finishes, fails, is resumed, or is deliberately removed. Never place API keys, tokens, or other secrets in these README files; keep large generated artifacts local and untracked.

## Latest RL Run Memory

The detached retry `serc_rl_200_4gpu_qlora_nf4_textlora_ctx16384_maxseq8_solverprompt_formal_retry3` ran as PID `2554463` from **2026-09-21 15:22:31** to an OOM at approximately **15:56:37 +08:00**. It used `data/rl/rl_200.parquet` with validation `data/rl/validation_10_rebuilt.parquet`, the retained `checkpoints/paper_500/sft_stage1_qlora_merged` base, `CUDA_VISIBLE_DEVICES=0,1,2,3`, QLoRA NF4/BF16, LoRA rank 8/alpha 32, rollout `n=8`, `max_num_seqs=8`, max prompt 8192, max response 2048, and max trajectory 16384. Tool execution, verification, and self-repair were enabled. The quantized base was frozen, including the vision tower and visual aligner; only language-side LoRA adapters trained. Hydra output was `Agent0-VL/outputs/2026-09-21/15-22-36/`, the log is `Agent0-VL/logs/serc_rl_200_4gpu_qlora_nf4_textlora_ctx16384_maxseq8_solverprompt_formal_retry3.log`, and no RL checkpoint was written. The failure occurred in `verl/workers/actor/dp_actor.py:337` during `loss.backward()`: external PID `1312109` occupied 21.07 GiB on logical GPU0, leaving 2.39 GiB for a requested 2.45 GiB allocation.

## Configurable Training CLI Examples

Run from Agent0-VL; inspect `--help` before changing experiments. No training or model loading occurs with `--dry-run`.

```bash
bash scripts/sft-agent0-2x4090-mixed.sh --gpus 0,1,2,3 --batch-size 1 --grad-accum 32 --epochs 3 --workers 1 --prefetch-factor 1 --dry-run
bash scripts/rl-agent0-2x4090-optimized.sh --gpus 0,1,2,3 --tp 2 --batch-size 4 --rollout-n 8 --concurrency 32 --mini-batch-size 1 --micro-batch-size 1 --dry-run
```

SFT `--batch-size` is per GPU; effective batch is GPU count times per-GPU batch times gradient accumulation. Without an explicit accumulation value the launcher rounds up to target at least 128. RL `--batch-size` is the global prompt batch, so trajectories per step equal batch times rollout-n. RL `--concurrency` is the scheduling cap, not a guarantee all trajectories run simultaneously. GPU IDs must be unique. Remove `--dry-run` to launch only when explicitly requested by the user. Syntax, default/two/four-GPU dry runs and invalid-argument rejection were checked; no GPU training was started for this CLI change.

## CPU Data Loading Memory Control (2026-09-27)

Current SFT and RL wrappers default to one dataloader worker and prefetch factor 1. Options: `--workers N`, `--prefetch-factor N`, `--memory-limit-percent 90` (integer 1..90), `--memory-wait-seconds 180`. SFT also accepts `--concurrency` for worker count. RL `--concurrency` still means vLLM sequence concurrency; use `--workers` for CPU loading. Each worker can prefetch at most the configured number of batches; SFT has a separate loader per distributed rank. Active batches and transient allocations are additional memory.

`Agent0-VL/tools/runtime_guard/sitecustomize.py` installs an opt-in import hook for PyTorch map/iterable DataLoader fetchers when `AGENT0_DATA_MEMORY_GUARD=1`. The wrappers add this directory to PYTHONPATH so spawned training/data workers inherit it. It avoids importing torch into unrelated processes at Python startup. Before fetching a batch, it checks host MemAvailable and accessible cgroup v2 memory limits, excluding reclaimable inactive file cache. At the configured pressure threshold (default 90%), fetching pauses; it resumes two percentage points below the threshold (88%). A sustained pause times out after 180 seconds with a clear error, rather than hanging indefinitely. Pause/resume/wait messages show in training logs.

This is data-loading backpressure, not an OS-enforced 90% total-memory cap. CPU-offloaded model/optimizer states, active tensors, concurrent fetches and other jobs can still exceed the threshold; never promise a strict total-memory ceiling. The guard does not stop or kill unrelated jobs. Queued batches are bounded by worker count and prefetch factor; workers=0 removes the multiprocessing prefetch queue.

SFT additionally uses lazy tokenization, dataset preprocessing process count 1 and disabled pinned memory. RL now reads train worker/prefetch counts from data configuration instead of hard-coding eight workers; wrappers also set validation loaders and overlong-prompt preprocessing to the conservative profile. Model output formats and rewards remain unchanged by loading control. Syntax/dry-run checks and a CPU import-hook plus simulated pressure pause/resume check were performed; no training run was started.

Use explicit loading limits when launching either current training script:

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/sft-agent0-2x4090-mixed.sh --gpus 0,1,2,3 --batch-size 1 --grad-accum 32 --workers 1 --prefetch-factor 1 --memory-limit-percent 90 --memory-wait-seconds 180
bash scripts/rl-agent0-2x4090-optimized.sh --gpus 2,3 --batch-size 2 --concurrency 16 --workers 1 --prefetch-factor 1 --memory-limit-percent 90 --memory-wait-seconds 180
```

Append `--dry-run` to inspect the resolved settings. To remove the multiprocessing waiting queue entirely, use `--workers 0`; fetch-time memory checks remain enabled. Raising workers or prefetch increases the queue's batch count and memory demand. The 90% figure measures overall host/container memory pressure, not the exact byte size of queued data; in-flight allocations can cross it after a fetch has started.

RL loader overrides use `+data.train_num_workers` and `+data.train_prefetch_factor` because those fields are added to the base configuration. Existing validation fields use `data.val_num_workers` and `data.val_prefetch_factor` without `+`. Hydra composition of these options was checked successfully. Keep this distinction when editing the launcher to avoid startup errors.

## Strict Final SFT Quota Requirement

The user's requirement is 1,000 accepted final trajectories, not 1,000 attempted prompts. Build from larger per-source candidate pools until Geometry3K has 112 accepted rows and the other eight sources have 111 each. Failed format, reference-answer, execution, verifier or duplicate checks do not count toward these quotas; replace failures within the same source. If a candidate pool is exhausted, fail with the source shortfall; do not relax validation or declare a smaller dataset complete.

The balanced builder no longer passes `--keep-unverified`: a missing reliable source reference cannot silently count as a correct answer based only on teacher confidence. Source adapters read assistant reference text from `messages` and `conversations` including `content`/`value` fields. After trajectory checks and reference matching, build_stream now applies the same strict final-row audit before writing/counting each row. Resume configuration includes a row-audit version; older outputs are rejected on resume rather than silently accepted under a stricter policy. Merge and exact-1,000 validation remain a final independent check. Mathematical or open-ended reference matching is currently exact/numeric after normalization; automated acceptance is not proof of absolute correctness. Some open-ended sources may fail to reach quota with this stricter reference policy; report that limitation instead of admitting unverified rows.

Existing running Python processes retain imported code and launch arguments; changes do not retroactively revalidate a user's active generation. Do not interrupt their run automatically or overwrite their outputs. Prepare new candidates if reference extraction changed and use a fresh generation output directory before rebuilding under the strict policy.

## Low-Confidence SFT Steps and Repair

Low confidence of an original Solver step is a Repair trigger, not an immediate dataset rejection. The SFT builder now verifies every Solver segment after its tool observation, including intermediate segments. Confidence below 0.7 triggers PATCH -> Solver regeneration -> tool execution when requested -> post-repair verification. Successful repairs retain the original low-confidence step, verifier critique, patch, corrected segment and post-verifier in the same training row. An explicit Solver continuation follows a non-final verified segment, so a Verifier response is not mistaken for subsequent Solver reasoning.

Quality filters use the current/final verification and the repair's post-verification, rather than rejecting the initial confidence. Unsuccessful/malformed repairs, failed post-verification, wrong final reference answers and unexecutable tool traces remain rejection reasons. An original failed tool execution is still rejected by the strict tool-success filter even if another call later succeeds; retaining recovered execution-error traces requires a separate repair-aware execution audit, not weakening all tool checks. These changes do not retroactively affect an already running generation process.

The focused repair/audit test file passed all 7 tests after the intermediate-step verification change. The test retains the original confidence 0.4 and repaired confidence 0.95 in a final row that passes strict audit. Resume state records `solver_verification_flow_version=2` to reject mixing older final-only verification output with the new per-step flow. No teacher generation or student training was started for this correction.

## Final Data Build Preflight Review (2026-09-27)

Do not resume the older balanced_1000 output: its first geometry run used the old verification policy and attempted 960 prompts for only one accepted row. Fresh strict candidates now use `data/sft/balanced_1000_v2/`; the final dataset name remains `data/sft/large/mixed_balanced_1000.jsonl`. Prepare re-normalizes selected raw train records using the existing deduplicated question index, reflecting the corrected reference and choice extraction. arXivQA reads `label` and preserves both choice-letter/text aliases; geometry/GeoQA/arXivQA include answer choices in the user question. LLaVA and SMR read assistant references from conversations. Missing-reference candidates are excluded. Original source assistant answers are references, not guaranteed human-verified gold labels; exact/numeric checks can reject valid open-ended paraphrases.

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/rebuild-balanced-data.sh prepare
bash scripts/rebuild-balanced-data.sh preflight
bash scripts/build-balanced-sft-with-teacher.sh
```

The teacher wrapper runs data preflight before loading the teacher. Preparation can take time because it scans raw data; it prints per-source candidate counts. Old candidate/output files are preserved. Preflight checks reference presence, image paths, candidate counts, total quotas and compatible state versions. Full input/prompt/config hashes are additionally checked by build_stream on resume.

Qwen3 teacher requests explicitly set `chat_template_kwargs.enable_thinking=false` so hidden reasoning does not consume the public-response budget. This is a teacher API/template adaptation; released Solver/Verifier/Repair prompt text is unchanged. Empty public content and token-limit-truncated responses are rejected, not accepted as trajectories. Fresh generation is still needed to assess real yield under the stricter rules. Four-GPU/64-concurrency teacher execution has not been validated by a new live generation run. All 18 local CPU tests passed, and real-source reference/choice plus empty/truncated response checks were performed. Candidate exhaustion fails; never loosen quotas or correctness gates to announce 1,000 rows.

## Tool Verification and User Runbook (2026-09-27)

This section is the current runbook and takes precedence over historical launch examples above. The repository memory file is `AGENTS.md`; do not create a separate `agent.md` with divergent instructions. The user runs teacher generation and training manually.

Actual builder-tool checks passed: plain Python computation; loading a real local image via `image_path`; pixel operations; variable continuity between code blocks in the same image-tool invocation; execution errors returned as `[Code Execution Result]`; and an end-to-end scripted-teacher trajectory using the real image sandbox, followed by reference matching, Verifier and export audit. Tests are in `Agent0-VL/tests/test_sft_builder_tools.py`. Latest full CPU result: `.venv/bin/python -m pytest -q tests`, 21 passed in 4.39 seconds. The scripted teacher checks plumbing, not live teacher quality or four-GPU throughput.

Tool observations follow main's text format. Processed/cropped images are not reattached to teacher requests; printing a generated image path does not let the teacher see that image. Image-tool temporary outputs are removed after execution. Variable context is retained within one multi-block invocation, not across separate trajectory turns. Do not claim live crop-image feedback, persistent tool sessions, a hard 90% RAM cap, or proven GPU OOM avoidance. Low confidence still triggers Repair before final acceptance gates.

### Run in order, in the foreground

All commands below run from `Agent0-VL`. Run the next stage only after the preceding command succeeds. Preparation does not delete old data or logs. It creates the new normalized candidate pool in `data/sft/balanced_1000_v2`; final generation must produce exactly 1,000 accepted rows (Geometry3K 112, eight other SFT sources 111 each), replacing rejected candidates within their source. Candidate exhaustion fails explicitly.

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL

# 1. Rebuild normalized candidates and check their references/images/state.
bash scripts/rebuild-balanced-data.sh prepare
bash scripts/rebuild-balanced-data.sh preflight

# 2. Generate and audit SFT data with the four-GPU teacher, then build RL data.
CUDA_VISIBLE_DEVICES=0,1,2,3 TEACHER_TP_SIZE=4 TEACHER_CONCURRENCY=64 \
  bash scripts/build-balanced-sft-with-teacher.sh

# 3. Audit the final mixed SFT dataset without loading/training the student.
bash scripts/sft-agent0-2x4090-mixed.sh --preflight-only

# 4. Train mixed SFT for 3 epochs on four GPUs (effective batch 128).
bash scripts/sft-agent0-2x4090-mixed.sh \
  --gpus 0,1,2,3 --batch-size 1 --grad-accum 32 --epochs 3 \
  --workers 1 --prefetch-factor 1 --memory-limit-percent 90

# 5. Train RL on two GPUs: warm-up 3 epochs, then formal SERC 1 epoch.
bash scripts/rl-agent0-2x4090-optimized.sh \
  --gpus 2,3 --tp 2 --batch-size 2 --rollout-n 8 --concurrency 16 \
  --warmup-epochs 3 --epochs 1 \
  --workers 1 --prefetch-factor 1 --memory-limit-percent 90
```

Step 2 automatically runs `rebuild-balanced-data.sh sft` followed by `rebuild-balanced-data.sh rl`. No additional RL-data rebuild is needed after it succeeds. To rebuild only RL data after a completed SFT generation, run `bash scripts/rebuild-balanced-data.sh rl`. RL warm-up/formal datasets are 200 rows each, mutually disjoint and excluded from selected SFT/validation questions. Default final files are `data/sft/large/mixed_balanced_1000.jsonl`, `data/rl/rl_warmup_200_multisource.parquet` and `data/rl/rl_200_multisource.parquet`.

The teacher wrapper reads its API key securely from the configured environment/key file. It starts a local teacher only when needed, stops a teacher it owns after generation, and leaves a reused existing service running. Existing teacher reuse does not establish that service's GPU/TP/concurrency profile. Generation log location is printed at startup. Fresh training runs use timestamped logs/checkpoint inventories. SFT exports a merged model and updates `checkpoints/sft_2x4090/latest_mixed_merged`; RL defaults to this mixed SFT export. At global RL batch 2, the 200-row sets imply 300 warm-up and 100 formal steps. Epoch values are the user's current reproduction settings, not an independently verified paper claim.

Two-GPU SFT alternative: use `--gpus 2,3 --grad-accum 64` with the other SFT flags unchanged. Four-GPU RL alternative (configurable, not GPU-validated by these CPU checks):

```bash
bash scripts/rl-agent0-2x4090-optimized.sh \
  --gpus 0,1,2,3 --tp 2 --batch-size 4 --rollout-n 8 --concurrency 32 \
  --warmup-epochs 3 --epochs 1 \
  --workers 1 --prefetch-factor 1 --memory-limit-percent 90
```

This four-GPU RL alternative generates 32 trajectories per prompt batch and implies 150 warm-up/50 formal steps; concurrency is a scheduling cap, not guaranteed simultaneous execution. Check selected GPUs are available before any launch. The CPU-loading guard pauses new batch reads at 90% pressure; model/optimizer offload and other processes can still push total memory higher. Append `--dry-run` to training commands to inspect settings without training. No live teacher generation or student training was launched while writing this runbook.

## Live SFT Failure Diagnosis (2026-09-27)

The 20260927_083721 builder exported 0 of its first 128 attempted Geometry3K candidates. Two small live diagnostic trajectories established two concrete failure modes: (1) the teacher correctly calculated base 20, its real SymPy tool succeeded and Verifier passed, but the final continuation was plain prose saying 20 / choice B; main's `scripts/prompt.txt` does not require boxed/FINAL_ANSWER whereas the unchanged extractor requires them, producing `no_final_answer_or_python`; (2) another response hit the default 2048-token response cap and was rejected as truncated, aggregated as `teacher_request_error`. These diagnostics do not establish that every request error is truncation. Another acceptance mismatch is original prompt's optional tool use versus local stage2 quality profile's mandatory actual tool call. Final-answer precedence skips executing code in a response that already contains a recognized final answer. Do not silently loosen extraction, force tools via modified original prompts or accept truncated answers under the user's main-format constraint. Explain these conflicts before selecting a policy change.

Builder now logs per-sample exception details and rejected last assistant response (bounded to 1200 characters), preserving prompts, extraction and acceptance gates. Existing already-imported running processes will not pick up this change until restarted. Full CPU checks after this diagnostic-only change: 21 passed in 4.72 seconds. Teacher service and user's running builder were left running; diagnostics did not write training data.

## Authorized SFT Generation Fixes (2026-09-27, supersedes original-prompt restrictions above)

The user explicitly authorized adding final-answer extraction instructions, raising teacher output limits and making tools optional. `Agent0-VL/scripts/prompt.txt` now requires exactly one final `\boxed{...}` (choice letter for MCQ), instructs code-only output when waiting for tools, and preserves JSON-only Verifier/Repair turns. This prompt is intentionally no longer byte-identical to main. Teacher public-response budget defaults to 8192 rather than 2048; configure `TEACHER_MAX_TOKENS` in the rebuild/teacher wrapper. Truncated/empty responses still fail. Accepted rows must still match references and pass confidence/repair checks; unused tools do not require tool_check=true. When tools are used, successful real execution and tool_check=true remain required. Code plus an answer in one Solver response now executes the code and retains its observation instead of skipping it. Final merge and training-data audits permit correct no-tool rows; tool/code evidence must be consistent when present.

Generation state versions are now row_audit_version=2 and solver_verification_flow_version=3. Teacher max tokens and prompt hash are recorded for resume compatibility. Do not resume prior failed attempts under the old acceptance rules. Stop the previous generation with Ctrl+C, then archive only per-source generation outputs/state (preserving all normalized candidates) and restart:

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/rebuild-balanced-data.sh reset-generation
bash scripts/rebuild-balanced-data.sh preflight
TEACHER_MAX_TOKENS=8192 CUDA_VISIBLE_DEVICES=0,1,2,3 TEACHER_TP_SIZE=4 TEACHER_CONCURRENCY=64 \
  bash scripts/build-balanced-sft-with-teacher.sh
```

The reset command refuses while a builder process is active and moves generation files into a timestamped generation_archive directory rather than deleting them. Raw candidates do not need another prepare run. After fixes, CPU checks: 22 passed in 4.39 seconds, including valid no-tool export and real execution of code+final-answer output; shell syntax and git diff whitespace checks passed. A live single-candidate diagnostic could not validate the new prompt because the teacher had stopped (connection refused); no teacher restart was performed automatically. Four-GPU acceptance yield and context budget under 8192 response tokens remain to be measured on the user's new run.

## Multi-Step Teacher and Image Tool Clarification (2026-09-27)

User authorized increasing teacher Solver steps and documenting image tools. Teacher generation defaults to 16 Solver segments (previously 8, not 2); Verifier/Repair requests are additional calls. `TEACHER_MAX_STEPS=16` flows through the teacher wrapper, balanced builder and build_stream; `TEACHER_MAX_TOKENS=8192` remains the per-request response budget. Nonempty text-only intermediate reasoning is now verified and continued rather than failing merely because no code/final marker was present. Confidence-triggered Repair and final/reference checks still apply. The model's 24576-token context limit still applies across accumulated conversation; 16 steps is a cap, not a guarantee every long conversation fits.

Prompt now explicitly describes Pillow cropping/resizing/rotation/contrast, ImageOps.mirror/flip, NumPy/OpenCV pixel operations and pytesseract OCR. These run as Python code inside the existing sandbox, not new API tool names. Each new tool invocation needs its own imports/variables. Text observations are returned; processed images are not automatically reattached. Real sandbox smoke succeeded: mirror 400x100, flip/crop 100x50 and OCR output HELLO 42. All 23 CPU tests passed in 4.51 seconds, including three consecutive no-tool reasoning segments. Shell syntax and git diff whitespace checks passed. Generation flow version is now 4 and max_reasoning_steps is included in resume configuration.

If a generation run was started before these edits, stop it with Ctrl+C; changes do not hot-reload. Archive prior per-source outputs/state while preserving candidates, then start with current settings:

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/rebuild-balanced-data.sh reset-generation
TEACHER_MAX_STEPS=16 TEACHER_MAX_TOKENS=8192 \
CUDA_VISIBLE_DEVICES=0,1,2,3 TEACHER_TP_SIZE=4 TEACHER_CONCURRENCY=64 \
bash scripts/build-balanced-sft-with-teacher.sh
```

This change targets teacher SFT-data generation; RL's separate rollout reasoning-step configuration is unchanged. No live teacher or training service was restarted for this edit.

## Tool Calling Prompt Audit (2026-09-27)

The user requested confirmation of tool invocation instructions. Audited prompt against builder execution and corrected ambiguous path-only output guidance. Prompt now gives a complete fenced-Python Pillow+pytesseract crop/grayscale/resize/OCR example, says to print useful text/numbers, clarifies image_path is the first local image in the data-building sandbox (not injected for text-only tasks), prohibits invented filenames/input overwrites, and explains temporary file lifetime and empty/error OCR evidence. The example's full-image crop is illustrative, not a mandatory ROI. These builder-specific injected variables must not be assumed for the separate main RL plain-subprocess sandbox. Prompt hash changes require archiving any earlier generation state after stopping its active builder before rebuilding; do not mix traces from different prompts. This audit did not run teacher generation or training.

## Default Launch Parameters Confirmed (2026-09-27)

User requested persisting the tool-prompt changes and using the new values as script defaults. No environment-variable prefix is needed for the main teacher data-generation entry point. `build-balanced-sft-with-teacher.sh` defaults to GPU 0,1,2,3 / TP4, 64 concurrent teacher requests / 64 engine sequences, 16 Solver segments, 8192 response tokens, 600-second request timeout, 24576 context and 16384 batched tokens. It prints the resolved Solver/token settings and validates positive integer step/token/timeout values before loading. Environment overrides remain supported. `rebuild-balanced-data.sh sft` passes the same request/step defaults. Python builder CLI defaults are also 16 steps / 8192 tokens. Standalone `start_qwen3_8_teacher_4gpu.sh` was aligned to 24576 context / 64 sequences / 16384 batched tokens; service-level settings do not themselves impose the per-request response or Solver-step limits.

Current prompt includes the authorized final boxed-answer requirement, optional tools, non-final text reasoning, actual OCR/Pillow/OpenCV invocation instructions, and explicit text-only observation/file-lifetime constraints. These are intentional user-authorized changes from original main. Preserve strict reference checking, verification and confidence-triggered Repair. Keep prior GPU/QLoRA/local loading adaptations.

After stopping a pre-edit builder, run once to preserve old outputs and start from fresh generation state (no raw-data prepare required):

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/rebuild-balanced-data.sh reset-generation
bash scripts/build-balanced-sft-with-teacher.sh
```

For an already fresh compatible generation directory, use only the second command; repeated reset discards resume progress into an archive. The wrapper generates SFT first then builds RL data. No generation, service startup or training was run while applying these defaults.

## SFT Context and JSON/Audit Fixes (2026-09-27)

Investigated the 09:14 diagnostics from logs/balanced_sft_build_20260927_090737/generation.log. Confirmed: old regex forbade braces inside JSON string fields, incorrectly rejecting Repair f-strings; task 82's critique contained an unsupported JSON LaTeX backslash escape; audit counted code quoted inside Repair JSON as unexecuted Solver code; a final no-tool Verifier result could override earlier valid tool verification. These are now fixed in the builder/auditor. JSON object parsing uses a string-aware decoder, preserves braces, handles raw newlines and unsupported backslash escapes without inventing missing structure, and canonicalizes accepted Verifier/Repair objects to valid JSON in exported messages. Actual complete task 197 and task 82 logged responses now parse successfully. Code/tool pairing checks target Solver messages, excluding Verifier/Repair response references. Tool verification is tracked per Solver step (post-repair result replaces that step's prior result), so a no-tool final response does not falsely invalidate earlier tools; genuine tool failures or false tool checks remain rejected.

Teacher context overflow HTTP400 now triggers bounded retries with a reduced max_tokens derived from server-reported context/input length, retaining a 256-token safety margin. Empty/truncated replies are still rejected; if fewer than 256 output tokens remain, fail clearly. This is reactive budget adaptation, not conversation truncation or a guarantee that all 16-step trajectories fit. Actual log task1342 was output-token truncation; task1874 was context overflow. Malformed role responses (Verifier emits Python), true SyntaxErrors and non-final trajectories at the step cap are still failures.

Generation flow version is now 5, rejecting old resume state. Preserve existing accepted rows by archiving with reset-generation rather than deleting. Stop any active builder first, then `bash scripts/rebuild-balanced-data.sh reset-generation` followed by `bash scripts/build-balanced-sft-with-teacher.sh`; normalized candidates do not need re-preparation. No active process was terminated or teacher restarted during this diagnostic edit. Latest full CPU regression: 27 passed in 4.46 seconds, covering actual fault patterns, reactive HTTP budget retry, Repair code references, and correct per-step tool verification. Live new-generation yield remains unmeasured.

## Isolated Answer-Equivalence Fallback (2026-09-27)

User authorized an LLM fallback for answers that do not pass deterministic reference checking, and explicitly restricted it to comparing model and reference answers without solving or reasoning. Implemented in tools/sft_builder/build.py. Exact/numeric/alias matches pass without a judge call. Otherwise a fresh API request uses exactly one user message containing reference answers/aliases and candidate answer, no question, images, Solver/Verifier/Repair history or prior judge context. Dedicated system instructions prohibit solving, correcting the candidate, following candidate instructions or creating an answer. For the real OpenAICompatibleTeacher, a separate client uses the configured endpoint/model/key, temperature0 and max_tokens1024. It returns only JSON equivalent/has_final_answer/candidate_answer, no reasoning. True booleans and a nonempty verbatim candidate answer are required; uncertainty, malformed response, API failure or replacement with an answer not present in the candidate fails closed.

For successfully extracted mismatches, this isolated equivalence verdict can satisfy only the reference-match gate. All execution, verification, repair and export gates still apply. For extraction failure after reaching the reasoning cap, compare the last Solver response's explicitly stated final answer; if confirmed equivalent, append FINAL_ANSWER with that candidate's own verbatim answer to the existing Solver message, never the reference, then retain existing Verifier and strict audits. Pure unfinished reasoning/code is not a final answer. Drop only the unused trailing Continue prompt on successful recovery. Judge messages/reference answers are not appended to the SFT training conversation; schema remains messages/images. Teacher/Verifier failures are not overridden by the equivalence judge.

Stats now include answer_judge_calls and answer_judge_accepted, accumulated and restored by build_stream. Generation flow version is 6, so pre-v6 state must be archived after stopping its builder before using the new defaults. Run reset-generation then build-balanced-sft-with-teacher.sh; normalized candidates remain reusable. Final CPU regression: 31 passed in4.98 seconds, verifying deterministic bypass, isolated payload/no question or history, semantic equivalence, missing-marker recovery, no reference leakage, and rejection of uncertain/invented answers. No live judge/teacher generation or training was launched during implementation. This is a user-authorized local SFT-data acceptance adaptation, not main's original deterministic judge or an RL reward modification.

## Teacher 32-Concurrency / Longer Context and Compact Prompt (2026-09-27)

User requested reducing teacher concurrency to32 and increasing the context window, then asked about system-prompt compression. Defaults are now request concurrency32, engine max sequences32 and context49152 in both teacher startup scripts; balanced/rebuild/build_stream default concurrency is32. Teacher remains TP4 on GPU0,1,2,3, 16 Solver steps, max8192 response tokens and16384 batched tokens. The local model text config max_position_embeddings is262144, so49152 is within configured capacity; this does not prove a live four-4090 memory/performance profile. No teacher was restarted by this edit.

Wrapper exports TEACHER_MAX_MODEL_LEN default49152 and supports TEACHER_MAX_NUM_SEQS default32. It checks authenticated /v1/models max_model_len and refuses to silently reuse an existing service with too short or unknown context; installed vLLM serving.py confirms this endpoint returns max_model_len. Old short-context services must be stopped/restarted by their owner. Runtime setting/prompt edits do not hot-reload.

Observed overflow input_tokens=16385,16642,16899 was a sequence of lower bounds, not necessarily full input length. Context retry now halves the requested output budget (also bounded by reported remaining context and256-token margin) instead of reducing only257 tokens repeatedly. It remains bounded and fails closed on exhausted context or response truncation. Increasing context does not resolve a separate finish_reason=length at the8192 output cap.

System prompt was compacted from3699 to approximately1580 characters (character count, not measured tokens), removing the long executable OCR demonstration/repetition while retaining code fences, optional tools, input-image variable and no-persistence constraints, OCR/Pillow/OpenCV operations, text-only results, boxed final answers, and JSON-only role responses. Large accumulated Solver/Verifier/Repair history is still the principal window pressure; compacting the initial prompt alone does not bound it. Full CPU checks31 passed in4.40 seconds, shell syntax and diff whitespace checks passed. Teacher32/49152 GPU execution and acceptance yield remain unvalidated. New prompt hash makes old resume states incompatible; after stopping the old builder, archive with reset-generation then launch build-balanced-sft-with-teacher.sh. Previously accepted output files are preserved in the archive; no raw candidate prepare is required.

## LLaVA Low-Yield Investigation / Reference Pairing Fix (2026-09-27)

Latest observed LLaVA progress224 attempted/35 written (15.625%). Of224 candidates,116 have images and108 are text-only; accepted35 comprise25 image/10 text-only. Reference responses average1278 characters (max5498), often long source assistant replies rather than unique gold answers. Candidate examples include London itineraries, poetry, story writing, robot design and knowledge explanations. These open-ended tasks are poorly matched to strict answer-equivalence against a single reference and concise boxed final answers. Do not equate a rejected different creative answer with a factually incorrect answer or silently switch the user-requested comparison-only judge to solving/grading task satisfaction. Original source assistant content is not guaranteed gold.

A concrete normalization defect was confirmed and fixed: _extract_question takes the first user question, but _raw_ground_truth previously took the last assistant response via reversed(messages). In a magpie_pro(l3_80b_mt) sample, the first question asks for the five R's of zero waste while selected reference answered a later request about household implementation. All32 raw sampled rows in that multi-turn shard have multiple assistant turns; this proves the defect in that shard, not the entire224 candidate set. sources.py now selects the first assistant response corresponding to the first user turn and stops before a second user if no first answer exists. Explicit gold fields remain authoritative. Regression checks33 passed in4.53 seconds.

Existing llava_ov_image_candidates.jsonl was normalized before this fix and is not automatically repaired. Its running build process and files were left untouched during investigation. Stop generation before rebuilding candidates with prepare; do not hot-edit a file a running job is consuming. Existing normalized candidates and generated trajectories require re-evaluation for multi-turn pairing; changing Python code alone does not fix serialized references. Archive old generation state before a new build because candidate hashes change. Strict tool errors, low Repair confidence and role-format errors remain additional causes of low yield. Selecting answerable visual/objective examples within the paper's LLaVA source could improve yield, but changes sampling policy and requires user authorization; no such filtering was applied here.

## Preserve First Three Completed Sources (2026-09-27)

User asked how to retain the first three sources after global reset. They were found in generation_archive_20260927_112351_676447. Geometry3K112, GeoQA111 and Mulberry111 rows passed the current full audit; their source/prompt hashes match current candidates/prompt. Restored both output JSONL and state JSON for these three by copying from archive (archive retained). Total334 accepted rows preserved. No teacher generation was started.

balanced_1000 supports --sources for selective prepare and reset-generation, and restore-completed with the same selector. Candidate/state mutations refuse while builders are active. Selective preparation retains the manifest/seed and all nonselected candidate files, deduplicates against their questions, and replaces selected files after complete writing. restore-completed verifies hashes, completed quotas and audits before copying, refuses overwriting current files. Build explicitly skips completed compatible audited sources with a reuse-completed message; changed prompt/source/model/budget settings prevent silent reuse.

Current remaining sources may still have old serialized multi-turn references, so rebuild only those six under the corrected first-user/first-assistant rule, preserving all334 accepted rows and their states:

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/rebuild-balanced-data.sh prepare --sources llava_ov_image mm_rlhf smr mmeureka retool arxivqa
bash scripts/rebuild-balanced-data.sh reset-generation --sources llava_ov_image mm_rlhf smr mmeureka retool arxivqa
bash scripts/build-balanced-sft-with-teacher.sh
```

Do not run unqualified prepare/reset-generation here: that would rewrite/archive retained sources too. Only if those three current outputs/state are missing and their compatible archive remains, restore via `bash scripts/rebuild-balanced-data.sh restore-completed --sources geometry3k geoqa mulberry`. This restoration has already been performed; rerunning refuses overwrite. Earlier full-rebuild advice is superseded by this selective runbook. Existing CPU suite33 passed in4.46 seconds; current output audits and hash comparisons were performed. LLaVA open-ended task mismatch remains; no sampling policy was changed.

## ReTool Low Acceptance: Same-Question Tool Turns (2026-09-27)

Current low-yield source was ReTool:96 attempted/5 written, with58 reference mismatches and28 historical tool execution error rejections. Raw ReTool messages are system,user,assistant(reasoning),tool,assistant(final answer); they are not necessarily multiple independent questions. Prior first-assistant pairing fix for LLaVA mistakenly picked ReTool's intermediate reasoning as the reference. Confirmed raw ordinal901 final answer30 versus serialized2421-character intermediate reasoning reference. Recomputed first96 references under corrected logic:95 differ. This is a preprocessing defect and does not mean95 teacher answers are incorrect. No ReTool OOM/context400 found in the inspected stage;2 responses were truncated.

sources.py now takes the last assistant response within the first question's group, traversing tool messages and recognized user tool observations/Continue prompts, and stops at a subsequent genuine user question. Thus LLaVA first-question pairing remains correct while ReTool final answers after tools are included. Explicit gold fields remain primary. Regression35 passed in4.79 seconds, covering both multi-question separation and same-question tool continuations. Streaming SMR first200 raw records showed single human/gpt pairs (0 multi-assistant rows), so this specific ReTool defect was not found in that SMR sample; no claim about all SMR rows.

No running process or serialized candidate/output was changed. ReTool requires selective candidate regeneration and output/state archival after stopping the current builder. Keep the seven completed sources and any other compatible completed source:

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/rebuild-balanced-data.sh prepare --sources retool
bash scripts/rebuild-balanced-data.sh reset-generation --sources retool
bash scripts/build-balanced-sft-with-teacher.sh
```

Do not reset all sources. True tool errors, reference mismatches, role failures and low repair confidence remain strict rejections. Increasing window/concurrency does not repair serialized reference errors. Existing completed source reuse validates hashes and audits.

## Four-GPU SFT Failure Diagnosis (2026-09-27)

Run checkpoints/sft_2x4090/sft_2x4090_mixed_20260927_202729 failed at data loading with swift.template.base.MaxLengthError, not CUDA OOM. Actual multimodal encoded rows17089,27845,32679 exceeded max_length16384; strict=true and lazy tokenization propagate the exception. Earlier batch2/gradaccum16/context16384 suggestion was an experimental memory profile and cannot cover all1000 accepted trajectories. Format/quality audit does not verify the student's multimodal token budget. Text-only tokenizer measurement for the1000 rows: p501605,p909762,p9518425,p9927011,max44105;62 already exceed16384 before vision tokens. Student local max_position_embeddings128000; image processor max_pixels12845056 (large potential vision-token overhead). These are capacity settings, not proof that long-context training fits RTX4090 memory.

Do not silently set strict=false or truncate/drop long samples just to make the launcher run: lazy strict=false resamples another row, changing actual sample coverage and source balance. Left/right truncation can remove question/image context, answer or Verifier/Repair evidence. Next launch needs a student multimodal length audit and an explicit long-sample policy; simply raising max_length or reducing batch does not establish correct/full1000-row coverage or no OOM. Failed run already registered statusfailed with exitcode1 in its README. No retraining, destructive cleanup, data edit or driver change was performed for this diagnosis.

## Mixed SFT checkpoint policy (2026-09-27)

`scripts/sft-agent0-2x4090-mixed.sh` now saves every **10 optimizer steps** (not epochs or micro-batches), with `save_strategy=steps`, `save_steps=10`, `save_total_limit=1`, `save_only_model=false`, and Swift checkpoint symlinks enabled. Full DeepSpeed optimizer/scheduler, RNG and trainer state are retained for resume; normal checkpoint rotation removes the old checkpoint after the new save returns. Keep one latest checkpoint rather than destructively overwriting files in place. The final merged model still exports after training completes.

Resume by repeating the original training settings and adding `--resume-from-checkpoint checkpoints/sft_2x4090/<run>/mixed_adapter/last-checkpoint`. The script resolves the link, checks training/adapter/optimizer state, reuses the original run directory and appends logs and README resume history; passes `resume_only_model=false`. Keep GPU count, batch, accumulation, data, model, prompt and total epoch settings consistent with the original run. `--epochs` means total epochs, not extra epochs. Before the first saved checkpoint there is no resumable state; a crash can lose progress since the last save. This change does not resolve the previously diagnosed overlength SFT samples. Shell syntax was checked; no training was launched.


## Four-GPU mixed SFT OOM diagnosis (2026-09-27 21:06)

Run `sft_2x4090_mixed_20260927_210523` with batch1/accum32/max_length65536 failed in backward before optimizer step1: GPU0 requested4.35GiB with4.06GiB free, GPU1 requested4.30GiB with4.13GiB free. No checkpoint was saved; GPUs were released. Raising max_length did not establish memory feasibility. The mixed launcher now enables installed Swift `use_liger_kernel=true` (override `USE_LIGER_KERNEL=false`), whose installed Qwen2.5-VL implementation defaults to fused linear cross entropy without materializing full training logits, and defaults allocator to expandable_segments:True. Preserve full rows, strict mode, QLoRA, frozen vision/aligner, and every10 optimizer step checkpoint policy. Shell syntax checked; no GPU training or numerical equivalence test launched, so memory feasibility for the longest rows remains unverified.


## RL KV-cache release before actor update (2026-09-27)

FSDP/vLLM already calls `sleep(level=sleep_level)` when leaving rollout generation. Added idempotent `ensure_sleeping_for_actor()` and an explicit call at the start of `update_actor`, before actor parameters/optimizer are loaded on GPU. If awake, the engine sleeps; if already asleep, no duplicate sleep is issued. CUDA synchronization and empty_cache follow; warning logs show sleep level and free GPU memory before/after. Next rollout uses the existing wake_up and full actor-weight synchronization path. This guard requires modern vLLM sleep-mode support and fails explicitly for legacy0.4.2/0.5.4/0.6.3. No teacher service or SFT flow changed, no GPU training launched. Existing runs must restart to load the change; existing rollout-end sleep means this guard may release no additional memory in normal operation.

## RL source audit and fixes (2026-09-27)

Audit confirmed actor response-only logits, activation CPU offload, microbatch1, actor/optimizer phase offload, eager vLLM, dummy_hf full weight synchronization and sleep_level2 in optimized profile. Prompt6144+response3072 fits configured9216; default max_num_seqs16 is scheduling cap, not proof of16 concurrent sequences fitting. Warmup defaults3 epochs and formal1; RL save_freq remains5 (SFT10-step change is separate).

Fixed RL rollout: execute code even when a Solver segment also contains a final answer; execute regenerated PATCH code before post-repair Verifier and replace stale tool evidence; include repaired-code tool counts. Fixed balanced-brace boxed-answer extraction and string-aware JSON decoding for Verifier/Repair (Python f-string braces previously broke PATCH parsing). Fixed checkpoint tracker and dataloader file publication using temporary-file+os.replace. Added checkpoint training-data source metadata: retain model/optimizer on warmup-to-formal transition but avoid loading warmup dataloader state onto a different dataset. Legacy checkpoints lack source metadata; their dataset match cannot be established by this new field. Moved FSDP old-actor-checkpoint deletion until all new actor shards are saved (not a complete controller-level transaction across data/critic files).

Validation: Python syntax +3 isolated parser checks passed; tests/test_dp_actor_batching.py4 passed. No GPU rollout equivalence/training test performed. Default latest_mixed_merged still resolves to sft_2x4090_mixed_20260923_211724/mixed_merged; Sept27 SFT attempts have not produced a successful new model, so RL must not be described as using newly trained Sept27 data/model.

Outstanding image-tool issue: RL passes bare Python code to sandbox, with no real image_path injection. Proposed image serialization into sandbox request was rejected by automatic approval review because SANDBOX_ENDPOINT may transmit images to an unauthorized destination. No image-transfer patch was applied; user clarification pending (local only versus configured service). Do not claim OCR/crop calls are correctly wired yet. Current math_verify scorer is deterministic; SFT isolated answer-judge fallback is not automatically enabled for RL. GPU weight synchronization checks parameter coverage, not numerical equivalence.

## Latest SFT failure (2026-09-27 21:24)

Run sft_2x4090_mixed_20260927_210754 used four GPUs, QLoRA rank16/alpha64, batch1/accum32/max_length65536, Liger enabled. It reached4/24 optimizer steps then OOM in gradient-checkpoint recomputation of Qwen language MLP, peft/tuners/lora/bnb.py output=lora_B(lora_A(dropout(x)))*scaling. Requested1.57GiB with413.69MiB free on GPU0. Unlike the preceding immediate backward OOM, fusion allowed several steps but does not solve longest-sequence activation/intermediate memory. No checkpoint-* exists because firstsave isstep10; only args.json/logging.jsonl remain. GPUs are released. Do not claim memory leak or broken model based on this log. Do not recommend lowering max_length without a full-row policy (strict mode would fail long rows). Swift's built-in activation_cpu_offload callback only enables on FSDP/FSDP2 with fsdp_config; blindly adding it to current DeepSpeed ZeRO3 does not enable offload. Need a compatible activation-offload/sequence-parallel implementation or explicitly authorized shorter trajectories before promising stable full1000-row training. No training was auto-launched.

## SFT saved-activation CPU offload (2026-09-27)

User requested CPU offload after the4/24-step OOM. Mixed SFT launcher now defaults `SFT_ACTIVATION_CPU_OFFLOAD=true`, configurable with `--activation-cpu-offload true|false`. It passes `--external_plugins tools/training/sft_activation_offload.py` to Swift only when enabled. The plugin idempotently wraps installed Swift Seq2SeqTrainer.training_step in `torch.autograd.graph.save_on_cpu(pin_memory=False)`, preserving Swift forward_context and DeepSpeed training_step. Context covers forward and backward, including gradient-checkpoint recomputation. Per-process first-step log `[sft-activation-offload pid=...] enabled` is evidence the plugin is invoked. No installed dependency files were edited. This offloads saved autograd tensors, not arbitrary live intermediate allocations or the quantized base; longest-sequence peaks may still OOM. Host RAM and PCIe transfer costs increase; existing90% data-loading guard is not a hard total-memory limit. QLoRA/Liger/batch/sequence/data formats and10-step full checkpoint policy remain. Shell/Python syntax checked; no GPU correctness or full training test run. User launches training manually with the same four-GPU command plus `--activation-cpu-offload true`.

## SFT on-demand activation offload (2026-09-27)

Latest user instruction changes default from forced offload to `--activation-cpu-offload auto`. At each Swift training microbatch, before forward, inspect expanded `input_ids.numel()` (includes image tokens and padded batch volume) and current rank GPU free memory. Default CPU offload triggers at >=8192 batch tokens OR <8GiB free; missing input_ids conservatively triggers offload. Otherwise saved tensors remain on GPU. Thresholds configurable via `--offload-token-threshold` / SFT_OFFLOAD_TOKEN_THRESHOLD and `--offload-min-free-gb` / SFT_OFFLOAD_MIN_FREE_GB. `true` forces offload, `false` disables plugin. These are conservative provisional thresholds, not measured guarantees or precise memory predictions. The offload context covers forward/backward/reentrant recomputation; it does not move arbitrary live compute intermediates. Log every microbatch with optimizer step, mode, enabled, batch_tokens, current free GiB and reason. Decisions are rank-local; they do not change distributed collective ordering. CLI/environment values inherited by all training ranks, dry-run and run README expose thresholds. Syntax checked; no GPU tests/training launched. Resume users must inspect these new settings; existing running processes retain the old plugin. Same fourGPU batch1/accum32/max_length65536 command can use `--activation-cpu-offload auto --offload-token-threshold 8192 --offload-min-free-gb 8`.

## Separate pipeline SFT launcher (2026-09-27)

Experimental script `Agent0-VL/scripts/sft-agent0-pipeline-4090.sh` uses Megatron-SWIFT, default GPUs0,1,2,3 with PP1/TP4/DP1, sequence parallel, global and micro batch1, 3 epochs, context65536, BF16 LoRA rank16/alpha64/all-linear, frozen vision/aligner, per-layer recomputation, and checkpoint interval10 steps retaining two checkpoints plus optimizer/RNG. It is NOT the NF4 QLoRA flow; `--allow-bf16-lora` is required. No DeepSpeed ZeRO3, Liger, or activation-offload plugin is reused. It audits exactly1000 SFT rows and checks GPU occupancy before launch. Outputs contain README and training log; no automatic merge/export or update to `latest_mixed_merged`; old Swift/ZeRO3 checkpoints cannot resume here.

The launcher currently sets `--attention_backend auto`. The CUDA13 FlashAttention wheel reports `2.8.3+cu130torch2.13`; TE2.11 incorrectly compares that local build suffix above its `2.8.3` upper bound and reports FlashAttention unavailable. `tools/runtime_guard/normalize_te_flashattn_version.py` applies a one-line, idempotent patch to the installed `.venv-pipeline` TE capability check to compare only the public version. It leaves the binary and package metadata untouched. Direct TE FlashAttention2.8.3 forward/backward smoke passed on RTX4090 (BF16, GQA, THD,1024 tokens). Forced cuDNN FusedAttention is not usable in the current overlay: execution errors because both libcudart.so.12 and libcudart.so.13 are mapped. Without the shim, Megatron's strict `flash` selection gave no backend; `auto` before shim fell back to memory-heavy Unfused attention and OOMed at update8/3000 during backward (requested2.66GiB with2.35GiB free). No checkpoint was saved because saving begins at step10. The corrected Flash path has not yet completed a training step.

Current isolated environment `.venv-pipeline`: ms-swift4.5.3, megatron-core0.16.1, mcore-bridge1.6.4, Transformer Engine2.11.0, reusing PyTorch2.13.0+cu130 from `.venv`; CUDA_HOME and CUDNN_HOME point to existing local CUDA13/cuDNN. The original TP4 attempt `...230705` used Unfused attention and OOMed at step8 before the flash-version shim existed. The later TP2 run `checkpoints/sft_pipeline/sft_pipeline_20260928_014720` used GPUs2,3 and automatic FlashAttention; it reached step12, saved checkpoint-10 successfully, then OOMed on a long row requesting4.63GiB with4.38GiB free. Its step10 checkpoint is retained; no later checkpoint exists. TP2 at context65536 is therefore not viable for this dataset on these cards. GPU0/1 remain occupied by processes not visible in this container, while GPU2/3 are free (at01:51 Sep28: 9046MiB, 3230MiB, 15MiB, 15MiB respectively). Do not terminate processes with unknown ownership; the launcher’s >512MiB guard is intentional.

A one-shot TP4 waiter is active since01:51 Sep28, logging to `/tmp/sft_pipeline_tp4_gpu_waiter_20260928.log`. It polls every10min and launches only after all four cards report <=512MiB; it does not kill existing processes and does not retry after a training failure. The corrected full run has not restarted yet. Earlier failed run dirs remain `...225932` (save limit startup error), `...230224` (no attention backend), and `...230705` (pre-shim backward OOM); no successful pipeline SFT model exists yet.

Run from `Agent0-VL/` when all four requested GPUs are free:

```bash
bash scripts/sft-agent0-pipeline-4090.sh --dry-run
bash scripts/sft-agent0-pipeline-4090.sh --allow-bf16-lora --preflight-only
bash scripts/sft-agent0-pipeline-4090.sh --allow-bf16-lora --gpus 0,1,2,3 --pp 1 --tp 4 --batch-size 1 --micro-batch-size 1 --max-length 65536 --epochs 3
```

The launcher uses `truncation_strategy=left` because this Swift version rejects `raise`; any record longer than65536 tokens can be truncated. Confirm no such records in the run log before claiming the whole SFT row content was trained. User requested status checks every10 minutes; do not poll more frequently unless a startup/error event requires diagnosis.

## Background pipeline dependency installation (2026-09-27)

Created `.venv-pipeline` as a separate dependency overlay; `agent0_existing_runtime.pth` references the existing `.venv` packages. Installed pip26.2.1, ms-swift4.5.3, megatron-core0.16.1, mcore-bridge1.6.4, transformers5.16.1 and Transformer Engine2.11.0; original PyTorch2.13.0+cu130 is reused. CUDA_HOME/CUDNN_HOME point to existing CUDA13 and cuDNN. Imports passed. The local `flash-attn 2.8.3+cu130torch2.13` wheel imports and its forward/backward path passes after the TE public-version compatibility shim described above. Do not refer to the previous TE2.19.0 pending install attempt; it was not the version retained in the working environment.
