# Agent0-VL SFT builder

This is a deliberately small reproduction of the unpublished SFT data
construction bridge. It starts from a local source row, runs the Solver with
the official `scripts/prompt.txt`, executes fenced Python through the released
`sandbox` backend, and appends a Verifier after each Solver segment. When a
Verifier is low-confidence, the same row continues with a PATCH repair,
corrected Solver segment, real tool re-execution, and a post-repair Verifier.
Rows are exported only when the final verification passes.

The exported row is exactly:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "images": []}
```

There is no `system` message because the official ms-swift scripts already
pass `scripts/prompt.txt` with `--system`.

## Supported sources

The source stage is fixed by the adapter:

* `geometry3k` (Stage 1): raw Geometry3K directories containing
  `.../<problem>/data.json` and `img_diagram.png`, or JSON/JSONL/Parquet
  exports with `problem`/`question`, `choices`, `answer`, and image fields.
* `retool` (Stage 2): JSON/JSONL/Parquet exports, including verl-style rows
  with `prompt[0].content` and `reward_model.ground_truth`.
* `mulberry` (Stage 1): generic JSON/JSONL/Parquet rows with a user
  conversation and image field. Existing assistant answers are never added to
  the Solver input; rows without a reliable reference can be retained only
  with `--keep-unverified`.

Images are resolved to existing local paths (or retained as URLs/data URLs),
and the initial user message contains exactly one `<image>` marker per image.

## Run

From the `Agent0-VL` directory:

```bash
python3 -m tools.sft_builder.build \
  --source geometry3k \
  --source-path /path/to/geometry3k/raw/train \
  --stage 1 \
  --output data/sft/stage1_tool_usage.jsonl \
  --teacher-base-url http://127.0.0.1:8000/v1 \
  --teacher-model qwen2.5-vl-7b \
  --max-tasks 10
```

For ReTool-style math-code data, use the Stage 2 output expected by the
official entry point:

```bash
python3 -m tools.sft_builder.build \
  --source retool \
  --source-path /path/to/retool.parquet \
  --stage 2 \
  --output data/sft/stage2_math_code.jsonl \
  --teacher-base-url http://127.0.0.1:8000/v1 \
  --teacher-model qwen2.5-vl-7b \
  --max-tasks 10
```

`OPENAI_API_KEY` is read by default; override the environment variable name
with `--teacher-api-key-env`. The source reference is used only after the
Solver/Verifier/Repair calls for exact normalized or strict numeric matching.
Rows with no reliable reference are skipped unless `--keep-unverified` is
given.

The provided raw data can be used in place; the builder does not copy the
116GB raw tree into the Git worktree. For example, the checked Geometry3K
probe is under:

```text
/mnt/d/Agent0/Agent0-VL/data/raw/.staging/geometry3k-probe/raw/train
```
