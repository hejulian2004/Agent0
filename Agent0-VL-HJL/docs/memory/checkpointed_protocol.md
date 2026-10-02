# Context-Isolated Checkpointed Verification and Repair

Target: Agent0-VL-HJL. New opt-in profile: `local_4090_checkpointed`.
Existing artifacts and the legacy `local_4090` protocol are not converted.

## Implemented

`agent0_protocol/checkpointed.py` defines complete-before solving, fresh independent
Verifier sessions, evidence-addressed checkpoints, prefix retention/suffix
regeneration, predetermined main/shadow branches, and delayed transition rewards.
Reference labels are handled by a separate scorer callback and never select a
branch. Every shadow intervention gets its own Repair group. Terminal verification
after the repair budget is audit-only. Infrastructure failure masks rewards.

`checkpointed_runtime.py` executes real multi-round Responses tools in separate
contexts/output directories. A deployed multimodal token counter is mandatory;
overlength fails explicitly. Python variables do not persist between snippets.
Image state is copied from the repair boundary; arbitrary filesystem side effects
are not physically undone. This is context isolation, not a filesystem security
boundary for arbitrary Python.

The balanced builder retains source quotas and atomic stream recovery. Accepted
problem flows export `roles/solve.jsonl`, `roles/repair.jsonl`, and
`roles/verify.jsonl`. Role row counts differ from accepted problem counts. Repair
prefix/checkpoint inputs are excluded from supervision by an explicit item boundary
and checked token-prefix alignment in the Swift template. Solver-before supervision
requires a correct before answer; no additional clean solve is synthesized.

SFT command printing: `.venv/bin/python -m scripts.launch sft-local --profile
local_4090_checkpointed --mode solve --dry-run` (also repair/verify). Adapter export
uses `--merge_lora false`; three adapters are never summed.
Register the three independent HF PEFT exports using
`python -m tools.checkpointed_bundle --base-model BASE --solve S --repair R --verify V --output BUNDLE`.
The manifest binds adapter content hashes and frozen base configuration/weight-file
identity. Atomic plain-PEFT checkpoints preserve three optimizer states, policy
versions, optional schedulers and RNG; they reject incompatible fingerprints.

All initial budgets, sampling counts, reward coefficients, adapter slots and
statistics settings live under this profile's `checkpointed` mapping in
`config.yaml`. Teacher model window is taken from its existing serve profile;
teacher per-request output cap remains omitted when `teacher_reply_tokens=0`.
Read-only statistics: `.venv/bin/python -m tools.checkpointed_statistics --audit
data/sft/checkpointed_v1/audit`. Suggestions do not mutate config. Budget exhaustion
is reported as censoring rather than treated as an observed complete length.

## Role-specific RL integration

The VERL worker loads three independent SFT adapters into Actor and frozen reference
models. Native vLLM loads immutable role snapshots after waking and receives an
explicit LoRARequest for each generation. Frozen base weights are synchronized
separately; no dense adapter merge is used. Sessions refill dynamically across
Solver, Verifier and Repair while sharing the configured concurrency limit.

Every rollout and reference computation finishes before S/R/V updates. Complete
role groups preserve sampled tokens/logprobs and mask unavailable rewards; padded
groups have zero policy loss. Rank-local role optimizer shards, scheduler states
and versions accompany VERL model/RNG checkpoints. Resume rejects old protocol,
changed fingerprints and changed shard layouts. Reference policies remain the
initial SFT adapters. Vision remains frozen.

Print the manual command without loading models or allocating a GPU:

```bash
.venv/bin/python -m tools.checkpointed_rl \
  --base-model /path/to/frozen/base --adapter-bundle /path/to/bundle --dry-run
```

`--preflight-only` checks actual bundle/data identity without launching training.
Tune limits, role group sizes, rewards and scheduling in config.yaml. Existing
datasets/checkpoints are not converted. New data and adapter exports are required.
Shared-adapter/non-default ablation execution and formal-from-checkpoint remain
explicitly unsupported; compatible full-run resume is supported.

CPU/Gloo session coverage verifies one real tool execution on the leader, image
state broadcast, and matching token/logprob histories across ranks. A real two-rank
CPU/FSDP test updates and restores three independent optimizers. A fake-engine
end-to-end test runs 4 problems × 8 initial samples, drives all main/shadow repairs,
re-checks and packs separate role batches. These checks do not validate GPU Ray,
distributed vLLM startup or GPU memory feasibility.

No generation, SFT, RL, model-service restart, GPU allocation or existing artifact
rewrite was performed during implementation. CPU tests do not establish GPU memory
feasibility or distributed vLLM startup correctness.
