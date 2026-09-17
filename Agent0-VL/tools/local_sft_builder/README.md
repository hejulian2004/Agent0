# Local SFT Builder

This is an additive, source-compatible construction package for Agent0-VL.
It does not modify upstream prompts, evaluators, sandbox implementations, or
training scripts.

The Solver runtime follows the upstream evaluator's fenced Python extraction.
Verifier and Repair audit responses are JSON. Final training rows contain only
the upstream Swift-compatible `messages` and `images` fields. Verifier,
Repair, and Regeneration units remain audit-only.

Run state and generated data should be placed outside the Git worktree, for
example under `Agent0-VL/data/runs/<run-id>/`. API keys are read only from
environment variables and must never be written to logs or manifests.
