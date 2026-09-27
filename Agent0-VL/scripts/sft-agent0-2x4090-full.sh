#!/usr/bin/env bash
set -Eeuo pipefail

# One-click two-stage SFT for two RTX 4090 GPUs.
# Stage 2 starts only after Stage 1 succeeds and consumes that run's adapter.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/checkpoints/base/Qwen2.5-VL-7B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
RUN_ID="${RUN_ID:-sft_2x4090_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-checkpoints/sft_2x4090/$RUN_ID}"
STAGE1_OUTPUT="${STAGE1_OUTPUT:-$RUN_ROOT/stage1}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-$RUN_ROOT/stage2}"
MERGED_OUTPUT="${MERGED_OUTPUT:-$RUN_ROOT/stage2_merged}"
STAGE1_DATA="${STAGE1_DATA:-data/sft/large/stage1_500.jsonl}"
STAGE2_DATA="${STAGE2_DATA:-data/sft/large/stage2_500.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BSZ="${BSZ:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
REPORT_TO="${REPORT_TO:-none}"
ALLOW_STAGE2_IMAGES="${ALLOW_STAGE2_IMAGES:-1}"
LOG_FILE="$RUN_ROOT/training.log"
RUN_README="$RUN_ROOT/README.md"

[[ -x "$ROOT_DIR/.venv/bin/python" ]] || {
    echo "Missing project virtual environment: $ROOT_DIR/.venv" >&2
    exit 1
}
[[ -x "$ROOT_DIR/.venv/bin/swift" ]] || {
    echo "Missing ms-swift executable: $ROOT_DIR/.venv/bin/swift" >&2
    exit 1
}
[[ -f "$BASE_MODEL/config.json" && -f "$BASE_MODEL/model.safetensors.index.json" ]] || {
    echo "Incomplete local base model: $BASE_MODEL" >&2
    echo "Set BASE_MODEL to a complete local Qwen2.5-VL checkpoint." >&2
    exit 1
}
for shard in "$BASE_MODEL"/model-0000{1..5}-of-00005.safetensors; do
    [[ -f "$shard" ]] || { echo "Missing model shard: $shard" >&2; exit 1; }
done

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
(( ${#GPU_IDS[@]} == 2 )) || {
    echo "CUDA_VISIBLE_DEVICES must contain exactly two GPU IDs; got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
}

mkdir -p "$RUN_ROOT"
START_TIME="$(date '+%F %T %z')"
printf '%s\n' \
    "# Two-GPU SFT run: $RUN_ID" \
    "" \
    "- Status: running" \
    "- Created: $START_TIME" \
    "- Base model: $BASE_MODEL" \
    "- Stage 1 output: $STAGE1_OUTPUT" \
    "- Stage 2 output: $STAGE2_OUTPUT" \
    "- Stage 2 merged model: $MERGED_OUTPUT" \
    "- Log: $LOG_FILE" \
    "- Stage 1 dataset: $STAGE1_DATA" \
    "- Stage 2 dataset: $STAGE2_DATA" \
    "- GPUs: $CUDA_VISIBLE_DEVICES (2 x RTX 4090 profile)" \
    "- Precision: QLoRA NF4, BF16 compute/storage, double quantization" \
    "- LoRA: rank 16, alpha 64, all-linear; vision tower and aligner frozen" \
    "- Limits: max_length=$MAX_LENGTH, per-device batch=$BSZ, grad_accum=$GRAD_ACCUM_STEPS" \
    "- Distributed: 2 processes, DeepSpeed ZeRO-3" \
    "- Command: bash scripts/sft-agent0-2x4090-full.sh" \
    >"$RUN_README"

RUN_STATUS="failed"
CURRENT_STAGE_PID=""

terminate_tree() {
    local parent_pid="$1"
    local child_pid
    while read -r child_pid; do
        [[ -n "$child_pid" ]] || continue
        terminate_tree "$child_pid"
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
    kill -TERM "$parent_pid" 2>/dev/null || true
}

on_interrupt() {
    trap - INT TERM
    echo "[sft-full] interrupted; terminating the active training process tree" | tee -a "$LOG_FILE" >&2
    if [[ -n "$CURRENT_STAGE_PID" ]] && kill -0 "$CURRENT_STAGE_PID" 2>/dev/null; then
        terminate_tree "$CURRENT_STAGE_PID"
        wait "$CURRENT_STAGE_PID" 2>/dev/null || true
    fi
    exit 130
}

record_exit() {
    local exit_code=$?
    sed -i "s/^- Status: running$/- Status: $RUN_STATUS/" "$RUN_README"
    printf '%s\n' \
        "" \
        "- Final status: $RUN_STATUS" \
        "- Last update: $(date '+%F %T %z')" \
        "- Exit code: $exit_code" \
        >>"$RUN_README"
}
trap record_exit EXIT
trap on_interrupt INT TERM

printf '[sft-full] run=%s\n' "$RUN_ID" | tee -a "$LOG_FILE"
printf '[sft-full] GPUs=%s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
printf '[sft-full] Stage 1 output=%s\n' "$STAGE1_OUTPUT" | tee -a "$LOG_FILE"
printf '[sft-full] Stage 2 output=%s\n' "$STAGE2_OUTPUT" | tee -a "$LOG_FILE"

echo "[sft-full] starting Stage 1" | tee -a "$LOG_FILE"
env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    NPROC_PER_NODE=2 \
    MODEL="$BASE_MODEL" \
    SFT_DATA="$STAGE1_DATA" \
    OUTPUT_DIR="$STAGE1_OUTPUT" \
    MAX_LENGTH="$MAX_LENGTH" \
    BSZ="$BSZ" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    REPORT_TO="$REPORT_TO" \
    ALLOW_STAGE2_IMAGES="$ALLOW_STAGE2_IMAGES" \
    PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" \
    bash "$ROOT_DIR/scripts/sft_stage1.sh" > >(tee -a "$LOG_FILE") 2>&1 &
CURRENT_STAGE_PID=$!
wait "$CURRENT_STAGE_PID"
CURRENT_STAGE_PID=""

if [[ "${PREFLIGHT_ONLY:-0}" != "1" ]]; then
    [[ -f "$STAGE1_OUTPUT/last/adapter_config.json" ]] || {
        echo "Stage 1 finished without a usable adapter: $STAGE1_OUTPUT/last" | tee -a "$LOG_FILE" >&2
        exit 1
    }
fi

echo "[sft-full] starting Stage 2" | tee -a "$LOG_FILE"
env -u MODEL \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    NPROC_PER_NODE=2 \
    BASE_MODEL="$BASE_MODEL" \
    STAGE1_ADAPTER="$STAGE1_OUTPUT/last" \
    SFT_DATA="$STAGE2_DATA" \
    ALLOW_STAGE2_IMAGES="$ALLOW_STAGE2_IMAGES" \
    OUTPUT_DIR="$STAGE2_OUTPUT" \
    MAX_LENGTH="$MAX_LENGTH" \
    BSZ="$BSZ" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    REPORT_TO="$REPORT_TO" \
    PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" \
    bash "$ROOT_DIR/scripts/sft_stage2.sh" > >(tee -a "$LOG_FILE") 2>&1 &
CURRENT_STAGE_PID=$!
wait "$CURRENT_STAGE_PID"
CURRENT_STAGE_PID=""

if [[ "${PREFLIGHT_ONLY:-0}" != "1" ]]; then
    echo "[sft-full] merging Stage 2 adapter into a standalone HF model" | tee -a "$LOG_FILE"
    "$ROOT_DIR/.venv/bin/swift" export \
        --model "$BASE_MODEL" \
        --adapters "$STAGE2_OUTPUT/last" \
        --merge_lora true \
        --safe_serialization true \
        --output_dir "$MERGED_OUTPUT" \
        2>&1 | tee -a "$LOG_FILE"
    [[ -f "$MERGED_OUTPUT/config.json" ]] || {
        echo "Stage 2 merge did not produce a valid model: $MERGED_OUTPUT" | tee -a "$LOG_FILE" >&2
        exit 1
    }
    ln -sfn "$(realpath --relative-to="$(dirname "$RUN_ROOT")" "$MERGED_OUTPUT")" \
        "$(dirname "$RUN_ROOT")/latest_stage2_merged"
    echo "[sft-full] merged Stage 2 model=$MERGED_OUTPUT" | tee -a "$LOG_FILE"
fi

echo "[sft-full] Stage 1 and Stage 2 completed successfully" | tee -a "$LOG_FILE"
RUN_STATUS="success"
