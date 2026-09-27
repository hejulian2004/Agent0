#!/usr/bin/env bash
set -Eeuo pipefail

# Build the missing paper-source SFT supplements with the local Qwen teacher.
# The teacher is already running with TP=2 on physical GPUs 2 and 3.  Each
# source is processed serially, while independent trajectories within a
# source use a bounded worker pool.  State files make interrupted sources
# resumable without changing the selected source rows.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3.8-27b}"
TEACHER_API_KEY_FILE="${TEACHER_API_KEY_FILE:-/mnt/d/Agent0/bench/.api_key}"
CONCURRENCY="${CONCURRENCY:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_TASKS="${MAX_TASKS:-300}"
MAX_REASONING_STEPS="${MAX_REASONING_STEPS:-8}"
SANDBOX_TIMEOUT="${SANDBOX_TIMEOUT:-60}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-300}"
TEACHER_RETRIES="${TEACHER_RETRIES:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/sft/full/supplement_2gpu_20260922}"
LOG_FILE="${LOG_FILE:-logs/sft_supplement_2gpu_concurrency${CONCURRENCY}_max${MAX_TASKS}_20260922.log}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing virtualenv Python: $PYTHON_BIN" >&2; exit 1; }
[[ -r "$TEACHER_API_KEY_FILE" ]] || {
    echo "Missing teacher API key file: $TEACHER_API_KEY_FILE" >&2
    exit 1
}
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "CONCURRENCY must be positive" >&2; exit 1; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be positive" >&2; exit 1; }
[[ "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_TASKS must be positive" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG_FILE")"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="$(tr -d '\r\n' < "$TEACHER_API_KEY_FILE")"

log() {
    printf '[sft-supplement] %s\n' "$*"
}

state_complete() {
    local state_path="$1"
    [[ -f "$state_path" ]] && "$PYTHON_BIN" - "$state_path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    state = json.load(handle)
raise SystemExit(0 if state.get("complete") is True else 1)
PY
}

run_source() {
    local source_name="$1"
    local source_path="$2"
    local stage="$3"
    local quality_profile="$4"
    local output_path="$5"
    local state_path="${output_path}.state.json"
    local source_log="${output_path%.jsonl}.log"
    local -a args=(
        -m tools.sft_builder.build_stream
        --source "$source_name"
        --source-path "$source_path"
        --source-split all
        --stage "$stage"
        --quality-profile "$quality_profile"
        --output "$output_path"
        --max-tasks "$MAX_TASKS"
        --concurrency "$CONCURRENCY"
        --batch-size "$BATCH_SIZE"
        --max-reasoning-steps "$MAX_REASONING_STEPS"
        --sandbox-timeout "$SANDBOX_TIMEOUT"
        --teacher-base-url "$TEACHER_BASE_URL"
        --teacher-model "$TEACHER_MODEL"
        --teacher-timeout "$TEACHER_TIMEOUT"
        --teacher-retries "$TEACHER_RETRIES"
        --repo-root "$ROOT_DIR"
    )

    if state_complete "$state_path"; then
        log "skip complete source=$source_name stage=$stage output=$output_path"
        return 0
    fi
    if [[ -f "$state_path" ]]; then
        args+=(--resume)
    fi

    log "start source=$source_name stage=$stage max_tasks=$MAX_TASKS concurrency=$CONCURRENCY batch_size=$BATCH_SIZE"
    "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "$source_log"
    state_complete "$state_path"
    log "complete source=$source_name stage=$stage output=$output_path"
}

run_source mulberry \
    data/sft/source_subsets/mulberry_stage1_candidates_1000.jsonl \
    1 stage1 "$OUTPUT_ROOT/stage1_mulberry.jsonl"

run_source mulberry \
    data/sft/source_subsets/mulberry_stage2_candidates_1000.jsonl \
    2 stage2 "$OUTPUT_ROOT/stage2_mulberry.jsonl"

run_source mmeureka \
    data/sft/source_subsets/mmeureka_stage2_candidates_1000.jsonl \
    2 stage2 "$OUTPUT_ROOT/stage2_mmeureka.jsonl"

log "sampling final Stage-1 set proportionally from the current and supplemental pools"
"$PYTHON_BIN" -m tools.sft_builder.sample_sft \
    --stage 1 \
    --input data/sft/large/stage1_500_broad_local.jsonl \
           "$OUTPUT_ROOT/stage1_mulberry.jsonl" \
    --output data/sft/large/stage1_multisource_500.jsonl \
    --count 500 \
    --seed 20260922 \
    --manifest data/sft/large/stage1_multisource_500.manifest.json \
    2>&1 | tee -a "$LOG_FILE"

log "sampling final Stage-2 set proportionally from the current and supplemental pools"
"$PYTHON_BIN" -m tools.sft_builder.sample_sft \
    --stage 2 \
    --allow-stage2-images \
    --input data/sft/large/stage2_500_broad.jsonl \
           "$OUTPUT_ROOT/stage2_mulberry.jsonl" \
           "$OUTPUT_ROOT/stage2_mmeureka.jsonl" \
    --output data/sft/large/stage2_multisource_500.jsonl \
    --count 500 \
    --seed 20260922 \
    --manifest data/sft/large/stage2_multisource_500.manifest.json \
    2>&1 | tee -a "$LOG_FILE"

"$PYTHON_BIN" -m tools.sft_builder.validate_sft \
    --stage 1 \
    --input data/sft/large/stage1_multisource_500.jsonl \
    2>&1 | tee -a "$LOG_FILE"

"$PYTHON_BIN" -m tools.sft_builder.validate_sft \
    --stage 2 \
    --allow-stage2-images \
    --input data/sft/large/stage2_multisource_500.jsonl \
    2>&1 | tee -a "$LOG_FILE"

log "all source builds and audits completed; teacher service was left running"
