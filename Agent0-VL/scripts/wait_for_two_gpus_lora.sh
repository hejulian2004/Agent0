#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for two genuinely idle GPUs, then launch the pending two-GPU LoRA SFT.
# The monitor is intentionally local: it sleeps between checks and never
# invokes the assistant or a model API.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLL_SECONDS="${POLL_SECONDS:-10}"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
MAX_USED_MIB="${MAX_USED_MIB:-512}"
MAX_UTIL="${MAX_UTIL:-5}"

MODEL_PATH="${MODEL_PATH:-/mnt/d/Agent0/models/Qwen2.5-VL-7B-Instruct}"
SFT_DATA="${SFT_DATA:-data/sft/large/stage1_500.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/paper_500/sft_stage1_lora_2gpu}"
MAX_LENGTH="${MAX_LENGTH:-3840}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
LOG_DIR="${LOG_DIR:-logs/lora_monitor}"

mkdir -p "$LOG_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%F %T %z')" "$*" | tee -a "$LOG_DIR/monitor.log"
}

idle_gpus() {
    nvidia-smi \
        --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
    | awk -F, -v min_free="$MIN_FREE_MIB" -v max_used="$MAX_USED_MIB" -v max_util="$MAX_UTIL" '
        {
            for (i = 1; i <= 4; i++) gsub(/[[:space:]]/, "", $i)
            if (($2 + 0) <= max_used && ($3 + 0) >= min_free && ($4 + 0) <= max_util)
                print $1
        }'
}

log "waiting for two idle GPUs (free >= ${MIN_FREE_MIB} MiB, used <= ${MAX_USED_MIB} MiB)"

while true; do
    mapfile -t candidates < <(idle_gpus || true)
    if (( ${#candidates[@]} >= 2 )); then
        selected="${candidates[0]},${candidates[1]}"
        log "found idle GPUs: ${selected}; launching 2-GPU LoRA SFT"

        export CUDA_VISIBLE_DEVICES="$selected"
        export NPROC_PER_NODE=2
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
        export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-10}"
        export MAX_PIXELS="${MAX_PIXELS:-3211264}"
        export WANDB_MODE="${WANDB_MODE:-disabled}"

        exec "$ROOT_DIR/.venv/bin/swift" sft \
            --model "$MODEL_PATH" \
            --dataset "$SFT_DATA" \
            --tuner_type lora \
            --lora_rank 8 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --torch_dtype bfloat16 \
            --system scripts/prompt.txt \
            --num_train_epochs 3 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --learning_rate 1e-5 \
            --freeze_vit true \
            --gradient_checkpointing true \
            --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
            --save_strategy epoch \
            --max_length "$MAX_LENGTH" \
            --save_total_limit 5 \
            --logging_steps 5 \
            --output_dir "$OUTPUT_DIR" \
            --warmup_ratio 0.05 \
            --dataloader_num_workers 4 \
            --deepspeed zero3 \
            --attn_impl flash_attn \
            --report_to none
    fi

    if (( ${#candidates[@]} == 0 )); then
        log "no idle GPU pair; next check in ${POLL_SECONDS}s"
    else
        log "only ${#candidates[@]} idle GPU(s) (${candidates[*]}); next check in ${POLL_SECONDS}s"
    fi
    sleep "$POLL_SECONDS"
done
