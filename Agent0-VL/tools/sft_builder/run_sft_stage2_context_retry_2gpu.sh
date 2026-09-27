#!/usr/bin/env bash
set -Eeuo pipefail

# Serial Stage-2 retry for large multimodal source rows. The teacher is started
# separately; the service may use the two-GPU or four-GPU TP profile.
# The builder scans up to the full 1,000-row candidate pool per source and
# stops only after the requested number of valid exported rows is reached.
# Existing state/output files are resumed so an interrupted run is not lost.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3.8-27b}"
TEACHER_API_KEY_FILE="${TEACHER_API_KEY_FILE:-/mnt/d/Agent0/bench/.api_key}"
CONCURRENCY="${CONCURRENCY:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_TASKS="${MAX_TASKS:-1000}"
TARGET_ROWS="${TARGET_ROWS:-300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/sft/full/supplement_2gpu_20260922_ctx20480}"
LOG_FILE="${LOG_FILE:-logs/sft_stage2_context20480_target300_2gpu_serial_20260922.log}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing virtualenv Python: $PYTHON_BIN" >&2; exit 1; }
[[ -r "$TEACHER_API_KEY_FILE" ]] || { echo "Missing teacher API key file: $TEACHER_API_KEY_FILE" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG_FILE")"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="$(tr -d '\r\n' < "$TEACHER_API_KEY_FILE")"

run_source() {
    local source_name="$1"
    local source_path="$2"
    local output_path="$3"
    local source_log="${output_path%.jsonl}.log"
    local state_path="${output_path}.state.json"
    local -a resume_args=()
    if [[ -f "$state_path" ]]; then
        resume_args+=(--resume)
    fi
    "$PYTHON_BIN" -m tools.sft_builder.build_stream \
        --source "$source_name" \
        --source-path "$source_path" \
        --source-split all \
        --stage 2 \
        --quality-profile stage2 \
        --output "$output_path" \
        --max-tasks "$MAX_TASKS" \
        --target-exported "$TARGET_ROWS" \
        --concurrency "$CONCURRENCY" \
        --batch-size "$BATCH_SIZE" \
        --max-reasoning-steps 8 \
        --sandbox-timeout 60 \
        --teacher-base-url "$TEACHER_BASE_URL" \
        --teacher-model "$TEACHER_MODEL" \
        --teacher-timeout 300 \
        --teacher-retries 1 \
        --repo-root "$ROOT_DIR" \
        "${resume_args[@]}" \
        2>&1 | tee -a "$source_log"
}

printf '[sft-stage2-retry] context=20480 concurrency=%s batch_size=%s max_tasks=%s target_exported=%s\n' \
    "$CONCURRENCY" "$BATCH_SIZE" "$MAX_TASKS" "$TARGET_ROWS" | tee -a "$LOG_FILE"

run_source mulberry \
    data/sft/source_subsets/mulberry_stage2_candidates_1000.jsonl \
    "$OUTPUT_ROOT/stage2_mulberry.jsonl"

run_source mmeureka \
    data/sft/source_subsets/mmeureka_stage2_candidates_1000.jsonl \
    "$OUTPUT_ROOT/stage2_mmeureka.jsonl"

"$PYTHON_BIN" -m tools.sft_builder.sample_sft \
    --stage 2 \
    --allow-stage2-images \
    --input data/sft/large/stage2_500_broad.jsonl \
           data/sft/full/supplement_2gpu_20260922/stage2_mulberry.jsonl \
           data/sft/full/supplement_2gpu_20260922_ctx20480/stage2_mulberry.jsonl \
           "$OUTPUT_ROOT/stage2_mulberry.jsonl" \
           "$OUTPUT_ROOT/stage2_mmeureka.jsonl" \
    --output data/sft/large/stage2_multisource_500.jsonl \
    --count 500 \
    --seed 20260922 \
    --manifest data/sft/large/stage2_multisource_500.manifest.json \
    2>&1 | tee -a "$LOG_FILE"

"$PYTHON_BIN" -m tools.sft_builder.validate_sft \
    --stage 2 \
    --allow-stage2-images \
    --input data/sft/large/stage2_multisource_500.jsonl \
    2>&1 | tee -a "$LOG_FILE"

echo "[sft-stage2-retry] completed; final Stage-2 file was audited" | tee -a "$LOG_FILE"
