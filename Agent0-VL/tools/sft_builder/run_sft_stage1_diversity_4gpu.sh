#!/usr/bin/env bash
set -Eeuo pipefail

# Build paper-aligned Stage-1 visual/tool trajectories from a fresh,
# source-stratified Mulberry subset. This only constructs data; it does not
# change the released SFT trainer, curriculum, or model code.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3.8-27b}"
TEACHER_API_KEY_FILE="${TEACHER_API_KEY_FILE:-/mnt/d/Agent0/bench/.api_key}"
CONCURRENCY="${CONCURRENCY:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_TASKS="${MAX_TASKS:-1200}"
TARGET_EXPORTED="${TARGET_EXPORTED:-250}"
MAX_REASONING_STEPS="${MAX_REASONING_STEPS:-8}"
SANDBOX_TIMEOUT="${SANDBOX_TIMEOUT:-60}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-300}"
TEACHER_RETRIES="${TEACHER_RETRIES:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/sft/full/supplement_4gpu_20260923_stage1_diversity_balanced}"
SOURCE_PATH="${SOURCE_PATH:-data/sft/source_subsets/mulberry_stage1_diversity_candidates_1200_20260923.jsonl}"
SOURCE_MANIFEST="${SOURCE_PATH%.jsonl}.manifest.json"
OUTPUT_PATH="$OUTPUT_ROOT/stage1_mulberry.jsonl"
STATE_PATH="$OUTPUT_PATH.state.json"
LOG_FILE="${LOG_FILE:-logs/sft_stage1_mulberry_diversity_4gpu_conc${CONCURRENCY}_target${TARGET_EXPORTED}_20260923.log}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing virtualenv Python: $PYTHON_BIN" >&2; exit 1; }
[[ -r "$TEACHER_API_KEY_FILE" ]] || { echo "Missing teacher API key file: $TEACHER_API_KEY_FILE" >&2; exit 1; }
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "CONCURRENCY and BATCH_SIZE must be positive integers" >&2
    exit 1
}
[[ "$MAX_TASKS" =~ ^[1-9][0-9]*$ && "$TARGET_EXPORTED" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_TASKS and TARGET_EXPORTED must be positive integers" >&2
    exit 1
}

mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG_FILE")"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="$(tr -d '\r\n' < "$TEACHER_API_KEY_FILE")"
export TEACHER_BASE_URL TEACHER_MODEL
[[ -n "$OPENAI_API_KEY" ]] || { echo "Teacher API key file is empty" >&2; exit 1; }

log() { printf '[sft-stage1-diversity] %s\n' "$*"; }

if [[ ! -s "$SOURCE_PATH" || ! -s "$SOURCE_MANIFEST" ]]; then
    log "preparing 1,200 unused Stage-1 Mulberry candidates, balanced across image subdatasets"
    "$PYTHON_BIN" -m tools.sft_builder.prepare_partitioned_source \
        --project-root "$ROOT_DIR" \
        --dataset mulberry \
        --usage-partition sft_stage1 \
        --partition data/processed/global_partition_v1/assigned_sft_stage1.jsonl \
        --raw data/raw/.staging/mulberry-proxy/mulberry_sft.json \
        --output "$SOURCE_PATH" \
        --count "$MAX_TASKS" \
        --seed 20260923 \
        --exclude-manifest data/sft/source_subsets/mulberry_stage1_candidates_1000.manifest.json \
        --stratify-image-subdataset
fi

log "candidate_source=$SOURCE_PATH target_valid=$TARGET_EXPORTED max_candidates=$MAX_TASKS concurrency=$CONCURRENCY batch_size=$BATCH_SIZE"
log "paper Stage-1 task: image-grounded tool use; source rows are train/train_inferred only; no forced-error or repair quota"
log "waiting for authenticated teacher model endpoint; no API key will be logged"

ready=false
for attempt in $(seq 1 180); do
    if "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

url = os.environ.get("TEACHER_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/") + "/models"
request = urllib.request.Request(
    url,
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
)
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=8) as response:
        data = json.loads(response.read().decode("utf-8"))
    models = {item.get("id") for item in data.get("data", [])}
    if os.environ.get("TEACHER_MODEL", "qwen3.8-27b") in models:
        print("ready")
        sys.exit(0)
except (OSError, urllib.error.URLError, ValueError, KeyError):
    pass
sys.exit(1)
PY
    then
        ready=true
        break
    fi
    if (( attempt % 6 == 0 )); then
        log "teacher still initializing/unavailable; retry=$attempt/180"
    fi
    sleep 5
done

[[ "$ready" == true ]] || { log "teacher did not become ready within 15 minutes"; exit 2; }
log "teacher ready; starting resumable Stage-1 trajectory generation"

args=(
    -m tools.sft_builder.build_stream
    --source mulberry
    --source-path "$SOURCE_PATH"
    --source-split all
    --stage 1
    --quality-profile stage1
    --output "$OUTPUT_PATH"
    --max-tasks "$MAX_TASKS"
    --target-exported "$TARGET_EXPORTED"
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
if [[ -f "$STATE_PATH" ]]; then
    args+=(--resume)
fi

"$PYTHON_BIN" "${args[@]}"
log "Stage-1 Mulberry generation finished; output=$OUTPUT_PATH state=$STATE_PATH"
