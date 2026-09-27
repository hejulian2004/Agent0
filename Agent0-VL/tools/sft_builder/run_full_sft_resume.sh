#!/usr/bin/env bash
set -euo pipefail

# Run all paper-listed SFT sources serially after an already-running GeoQA
# build. The teacher is deliberately stopped only after merge and strict-load
# validation succeed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TEACHER_PID="${TEACHER_PID:?Set TEACHER_PID to the current vLLM API server PID}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3.8-27b}"
RETOOL_SOURCE="${RETOOL_SOURCE:-data/raw/.staging/retool-proxy/train_2000.parquet}"
MULBERRY_SOURCE="${MULBERRY_SOURCE:-data/raw/.staging/mulberry-proxy/mulberry_sft.json}"
MMEUREKA_SOURCE="${MMEUREKA_SOURCE:-data/raw/.staging/mmeureka-proxy/dataset.jsonl}"
COMMON_ARGS=(
  --concurrency 64
  --batch-size 64
  --sandbox-timeout 60
  --teacher-timeout 300
  --teacher-retries 2
  --teacher-base-url "$TEACHER_BASE_URL"
  --teacher-model "$TEACHER_MODEL"
  --teacher-temperature 0.0
  --teacher-top-p 1.0
)

state_complete() {
  local state="$1"
  [[ -f "$state" ]] && [[ "$(jq -r '.complete // false' "$state")" == "true" ]]
}

wait_for_geoqa() {
  local output="data/sft/full/stage1_geoqa_train.jsonl"
  local state="${output}.state.json"
  while ! state_complete "$state"; do
    if ! pgrep -f 'tools\.sft_builder\.build_stream --source geoqa' >/dev/null; then
      echo "GeoQA builder is no longer running and its state is incomplete: $state" >&2
      exit 1
    fi
    sleep 60
  done
}

run_source() {
  local source="$1"
  local source_path="$2"
  local stage="$3"
  local max_tasks="$4"
  local output="$5"
  local quality_profile="$6"
  local state="${output}.state.json"
  local log="${output%.jsonl}.log"
  local args=(
    --source "$source"
    --source-path "$source_path"
    --source-split train
    --stage "$stage"
    --quality-profile "$quality_profile"
    --output "$output"
    --max-tasks "$max_tasks"
    "${COMMON_ARGS[@]}"
  )

  if state_complete "$state"; then
    return 0
  fi
  if [[ -f "$state" ]]; then
    args+=(--resume)
  fi
  echo "Starting $source stage=$stage output=$output" >&2
  .venv/bin/python -m tools.sft_builder.build_stream "${args[@]}" >> "$log" 2>&1
  state_complete "$state"
}

wait_for_geoqa

run_source retool \
  "$RETOOL_SOURCE" \
  1 2000 data/sft/full/stage1_retool_train.jsonl stage1
run_source mulberry \
  "$MULBERRY_SOURCE" \
  1 272775 data/sft/full/stage1_mulberry_train.jsonl stage1
run_source retool \
  "$RETOOL_SOURCE" \
  2 2000 data/sft/full/stage2_retool_train.jsonl stage2
run_source mulberry \
  "$MULBERRY_SOURCE" \
  2 272775 data/sft/full/stage2_mulberry_train.jsonl stage2
run_source mmeureka \
  "$MMEUREKA_SOURCE" \
  2 54931 data/sft/full/stage2_mmeureka_train.jsonl stage2

.venv/bin/python -m tools.sft_builder.merge_sft --stage 1 \
  --input data/sft/full/stage1_geometry3k_train.jsonl \
          data/sft/full/stage1_geoqa_train.jsonl \
          data/sft/full/stage1_retool_train.jsonl \
          data/sft/full/stage1_mulberry_train.jsonl \
  --output data/sft/stage1_full.jsonl \
  --manifest data/sft/stage1_full.manifest.json
.venv/bin/python -m tools.sft_builder.merge_sft --stage 2 \
  --allow-stage2-images \
  --input data/sft/full/stage2_retool_train.jsonl \
          data/sft/full/stage2_mulberry_train.jsonl \
          data/sft/full/stage2_mmeureka_train.jsonl \
  --output data/sft/stage2_full.jsonl \
  --manifest data/sft/stage2_full.manifest.json

.venv/bin/python - <<'PY'
from pathlib import Path
from swift.dataset import load_dataset

for path in (Path("data/sft/stage1_full.jsonl"), Path("data/sft/stage2_full.jsonl")):
    train_dataset, val_dataset = load_dataset([str(path)], split_dataset_ratio=0, strict=True)
    assert val_dataset is None
    print(f"validated {path}: {len(train_dataset)} rows")
PY

if ps -p "$TEACHER_PID" -o args= 2>/dev/null | grep -F -- '--port 8000' >/dev/null; then
  kill -TERM "$TEACHER_PID"
fi
echo "SFT build, merge, and strict-load validation completed; teacher stop requested."
