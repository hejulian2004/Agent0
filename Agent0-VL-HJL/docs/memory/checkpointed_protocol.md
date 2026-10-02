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

SFT retains all three mode files and additionally exports `roles/shared.jsonl`
with per-row mode metadata, source hashes and mode counts. The shared dataset
contains the original positive rows, without resampling or joining conversations.
SFT command printing: `.venv/bin/python -m scripts.launch sft-local --profile
local_4090_checkpointed --adapter shared --dry-run`. One SFT run learns all three
modes on one LoRA initialized from the frozen base. Per-mode prompts, row length
limits and supervision boundaries remain distinct. Export uses `--merge_lora false`.
Register the single HF PEFT export using
`python -m tools.checkpointed_bundle --base-model BASE --shared SHARED --output BUNDLE`.
Former `--mode` training/export commands fail explicitly.
The manifest binds adapter content hashes and frozen base configuration/weight-file
identity. Atomic plain-PEFT checkpoints preserve one optimizer state, a shared policy
version, optional scheduler and RNG; they reject incompatible fingerprints.

All initial budgets, sampling counts, reward coefficients, adapter slots and
statistics settings live under this profile's `checkpointed` mapping in
`config.yaml`. Teacher model window is taken from its existing serve profile;
teacher per-request output cap remains omitted when `teacher_reply_tokens=0`.
Read-only statistics: `.venv/bin/python -m tools.checkpointed_statistics --audit
data/sft/checkpointed_v1/audit`. Suggestions do not mutate config. Budget exhaustion
is reported as censoring rather than treated as an observed complete length.

## Role-specific RL integration

The VERL worker loads one shared SFT adapter into Actor and one frozen copy into
the Reference model. The Actor has one optimizer, scheduler and version dictionary
`{"shared": n}`. Solve, Repair and Verify all map to this same adapter. Native vLLM loads immutable role snapshots after waking and receives an
explicit LoRARequest for each generation. Frozen base weights are synchronized
separately; no dense adapter merge is used. Sessions refill dynamically across
Solver, Verifier and Repair while sharing the configured concurrency limit.

Every rollout and reference computation finishes before a joint update. The three
GRPO groups remain independent. Each mode loss is averaged over its valid groups,
then weighted by `prior[mode] * sqrt(global_valid_group_count)` and normalized over
active modes. Default priors are Solve=0.5, Repair=0.3, Verify=0.2; configure them
under `checkpointed.joint_training.rl_weighting`. Counts exclude unavailable rewards,
incomplete groups, no-action groups and dummy padding, and are computed globally
before rank partitioning. Valid zero-variance groups retain zero advantages and the
existing entropy/KL losses. Zero priors omit that mode's RL objective.

All mode micro-batches accumulate gradients with unchanged parameters; clipping,
optimizer and scheduler happen once after the full rollout batch is consumed.
A successful update advances the shared version once. Non-finite gradients skip
the update without advancing scheduler/version; empty objectives also skip it.
An already consumed phase or stale policy version is rejected. `ppo_epochs=1` is
required, and the current profile is unchanged. Native vLLM registers only one
immutable LoRARequest per rollout phase; mode switches never reload the adapter.
 Complete
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
datasets/checkpoints are not converted. New jointly trained adapter exports are required; existing valid checkpointed data is not rewritten.
Shared-adapter execution is the checkpointed architecture; formal-from-checkpoint remains explicitly unsupported;
compatible full-run resume is supported.

### Action-conditioned rewards and ablations

`transition_reward` requires the parsed Verifier action. Wrong-answer accept gets
`false_accept=-1`; correct-answer accept gets preservation plus keep-tool bonus.
Successful intervention gets fix plus fix-tool bonus; failed intervention gets
`failed_fix=0` (configurable); intervention preserving an already correct answer
gets `unnecessary_revision=0`, without any tool bonus. Corruption gets -1.
`uncertain` is a repair-triggering intervention and follows the revise reward rules;
its action remains distinct in audit/statistics. These are final-answer outcomes,
not correctness labels for individual critiques.

`rewards.repair_credit_mode=main` uses the predetermined completion; `mean` averages
the action-conditioned reward across every Repair completion under the same feedback.
Any unavailable/infrastructure outcome masks mean credit. The continuing trajectory
always uses the fixed main completion. Logs keep the main binary transition separately
from `repair_outcomes` and `repair_success_rate`.

The three boolean ablation switches now execute different behavior:

| Variant | context_isolation | suffix_repair | train_verifier_rl | tool_bonus_enabled |
| --- | --- | --- | --- | --- |
| A | true | false | false | false |
| B | true | true | false | false |
| C | true | true | true | false |
| Full | true | true | true | true |

The original baseline remains the legacy `local_4090` profile. For an isolation-only
comparison against Full, set only `context_isolation=false`.
False isolation inherits current Solver messages and current image state, while
Python remains stateless between snippets. Inherited Solver calls never earn V tool
bonus or V policy loss. False suffix repair supplies ordinary verification feedback
and regenerates from the original problem, with no accepted prefix or checkpoint in
the Repair model input. A checkpoint may still be retained for audit/branch identity.
False Verifier RL keeps all checks and delayed-reward auditing but builds no Verify
Actor batch and performs no Verify reference computation. It removes only the
Verify loss; Solve/Repair updates still change the shared parameters used during
verification. This ablation does not freeze a separate Verifier model. Use fixed RL problems and the same initial adapter bundle across variants.
Configuration/code fingerprints reject incompatible resume; no stored data is rewritten.

CPU/Gloo session coverage verifies one real tool execution on the leader, image
state broadcast, and matching token/logprob histories across ranks. A real two-rank
CPU/FSDP test updates and restores the single shared optimizer. A fake-engine
end-to-end test runs 4 problems × 8 initial samples, drives all main/shadow repairs,
re-checks and packs separate role batches. These checks do not validate GPU Ray,
distributed vLLM startup or GPU memory feasibility.

No generation, SFT, RL, model-service restart, GPU allocation or existing artifact
rewrite was performed during implementation. CPU tests do not establish GPU memory
feasibility or distributed vLLM startup correctness.

## Shared Adapter compatibility

Training layout `agent0.checkpointed.shared_lora.v1` is separate from the unchanged
`agent0.checkpointed.v1` data/Verification Checkpoint protocol. Bundle and optimizer
sidecars require the exact shared layout and mode mapping, and training fingerprints
include code, layout, config and bundle identity. Old three-adapter bundles and RL
checkpoints are rejected without copying, averaging or summing their weights.
Existing stored datasets and historical artifacts remain untouched. Legacy
`local_4090` infrastructure is unchanged. FSDP adapter-only export recognizes the
shared sidecar; legacy default-LoRA export retains its existing behavior.

Logs keep `solve/*`, `repair/*`, `verify/*` and add per-mode `valid_groups` and
`loss_weight`, plus `adapter/shared_version` and `adapter/update_applied`.
