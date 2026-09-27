# Agent0-VL SFT builder

For the new sequential 500 + 500 paper-source rebuild, use
[`docs/rebuild-paper-1000.md`](../../docs/rebuild-paper-1000.md). The rebuild
counts downloaded training records before calling the teacher and records the
quota for every source; the older examples below describe retained low-level
commands and historical datasets.

The upstream `main` branch publishes the SFT launch scripts and their
`scripts/prompt.txt`, but does not publish the data-generation program or the
paper's 200k SFT trajectories. This directory is a local reconstruction, not
the authors' original data pipeline. It now passes the released system prompt
unchanged to both SFT stages, without adding stage-specific output rules.

The paper's Appendix C illustrates JSON tool calls and a `FINAL_ANSWER` field.
The released SFT script instead passes `scripts/prompt.txt`, which asks for
Python inside `<code>` and a fenced block, and does not require an answer tag.
The released rollout executes fenced Python. These published formats differ;
the builder follows the actual SFT script's prompt and tool syntax.

The local builder starts from a source row, runs the Solver, executes fenced
Python, and appends Verifier and optional Repair turns. Its filtering rules
are local reconstruction choices; successful output must be audited before
training.

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
.venv/bin/python -m tools.sft_builder.build \
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
.venv/bin/python -m tools.sft_builder.build \
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

Before training, validate the completed JSONL without rewriting it:

```bash
.venv/bin/python -m tools.sft_builder.validate_sft \
  --stage 1 \
  --input data/sft/large/stage1_500_local.jsonl \
  --expected-rows 500

.venv/bin/python -m tools.sft_builder.validate_sft \
  --stage 2 \
  --input data/sft/large/stage2_500.jsonl \
  --expected-rows 500
```

The validator rejects malformed JSON, invalid role ordering, duplicate rows,
undecodable local images, Stage-1 rows without images, and incomplete
Solver/Verifier/Repair flows. Stage 2 is text-only by default, so rows with
images are rejected unless `--allow-stage2-images` is passed. This explicit
flag is used when adding multimodal MM-Eureka Stage-2 rows. The two-GPU SFT
launch script forwards the same choice through `ALLOW_STAGE2_IMAGES=1`.

To keep a fixed-size training set while adding missing sources, use
`tools.sft_builder.sample_sft`. It audits and deduplicates each input pool,
then allocates the requested row count proportionally with a fixed seed. For
image paths from Mulberry, `--stratify-mulberry-sources` creates per-dataset
buckets; `--ensure-source-coverage` reserves at least one row per nonempty
bucket; and `--require-all-repairs` forces every valid Self-Repair example into
the sample. `--normalize-form-feed` converts OCR page-break characters to
newlines before strict audit.

The current two-GPU Stage-1 set was selected with:

```bash
.venv/bin/python -m tools.sft_builder.sample_sft \
  --stage 1 \
  --input data/sft/large/stage1_multisource_500.jsonl \
         data/sft/full/supplement_4gpu_20260923_stage1_diversity_balanced/stage1_mulberry.jsonl \
  --output data/sft/large/stage1_multisource_500_repaired_20260923.jsonl \
  --count 500 --seed 20260923 \
  --stratify-mulberry-sources --ensure-source-coverage \
  --require-all-repairs --normalize-form-feed
```

For an optional single-run mixed-data experiment (500 Stage-1 + 500 Stage-2
rows), use:

```bash
bash scripts/sft-agent0-2x4090-mixed.sh
```

The script strictly audits both input files, requires all six selected Stage-1
Repair rows, rejects exact duplicates, writes a run-scoped concatenated JSONL,
then trains and exports one mixed-data model. It defaults to the selected
Stage-1 and Stage-2 files above; set `STAGE2_DATA=...` to use a refreshed
Stage-2 file. This is a single mixed-data run, not the sequential
Stage-1-then-Stage-2 curriculum used by `scripts/sft-agent0-2x4090-full.sh`.

Large raw sources are not committed. Put or copy the required source under the
current workspace, for example:

```text
data/raw/.staging/geometry3k-probe/raw/train
```
