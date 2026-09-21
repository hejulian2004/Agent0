#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for two genuinely idle GPUs, probe the largest safe training context on
# the longest Stage-1 example, then launch the two-GPU LoRA SFT.  The monitor
# is local-only and never invokes the assistant or a model API.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLL_SECONDS="${POLL_SECONDS:-10}"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
MAX_USED_MIB="${MAX_USED_MIB:-512}"
MAX_UTIL="${MAX_UTIL:-5}"

MODEL_PATH="${MODEL_PATH:-/mnt/d/Agent0/models/Qwen2.5-VL-7B-Instruct}"
SFT_DATA="${SFT_DATA:-data/sft/large/stage1_500.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/paper_500/sft_stage1_lora_2gpu}"
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

make_probe_data() {
    local probe_path
    probe_path="$(mktemp /tmp/agent0-stage1-longest.XXXXXX.jsonl)"
    "$ROOT_DIR/.venv/bin/python" - "$SFT_DATA" "$probe_path" <<'PY'
import json
import sys

source, target = sys.argv[1:3]
best = None
best_score = -1
with open(source, encoding='utf-8') as handle:
    for line in handle:
        row = json.loads(line)
        messages = row.get('messages', [])
        score = len(json.dumps(messages, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best = row
if best is None:
    raise SystemExit('no SFT rows available for context probing')
with open(target, 'w', encoding='utf-8') as handle:
    handle.write(json.dumps(best, ensure_ascii=False) + '\n')
print(best_score)
PY
    printf '%s\n' "$probe_path"
}

run_sft() {
    local context_length="$1"
    local dataset="$2"
    local output_dir="$3"
    local epochs="$4"
    local grad_accum="$5"
    local max_steps="${6:-}"
    local save_strategy="epoch"
    if [[ -n "$max_steps" ]]; then
        save_strategy="no"
    fi

    local -a args=(
        --model "$MODEL_PATH"
        --dataset "$dataset"
        --tuner_type lora
        --lora_rank 8
        --lora_alpha 32
        --lora_dropout 0.05
        --torch_dtype bfloat16
        --system scripts/prompt.txt
        --num_train_epochs "$epochs"
        --per_device_train_batch_size 1
        --per_device_eval_batch_size 1
        --learning_rate 1e-5
        --freeze_vit true
        --gradient_checkpointing true
        --gradient_accumulation_steps "$grad_accum"
        --save_strategy "$save_strategy"
        --max_length "$context_length"
        --save_total_limit 5
        --logging_steps 5
        --output_dir "$output_dir"
        --warmup_ratio 0.05
        --dataloader_num_workers 1
        --deepspeed zero3
        --attn_impl flash_attn
        --report_to none
    )
    if [[ -n "$max_steps" ]]; then
        args+=(--max_steps "$max_steps" --logging_steps 1)
    fi

    env PATH="$ROOT_DIR/.venv/bin:$PATH" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        NPROC_PER_NODE=2 \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
        FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-10}" \
        MAX_PIXELS="${MAX_PIXELS:-3211264}" \
        WANDB_MODE=disabled \
        "$ROOT_DIR/.venv/bin/swift" sft "${args[@]}"
}

probe_and_train() {
    local probe_data="$1"
    local best=0
    local context
    local rc

    # The upper bound includes the repository's 10240 setting and one larger
    # candidate.  The first successful value is retained while later values
    # are tested, so the final run uses the largest successful context.
    for context in 2048 3072 3840 4096 5120 6144 8192 10240 12288; do
        log "probing max_length=${context} on the longest Stage-1 row"
        set +e
        run_sft "$context" "$probe_data" "checkpoints/paper_500/context_probe_2gpu_${context}" 1 1 1 \
            >"$LOG_DIR/probe_${context}.log" 2>&1
        rc=$?
        set -e
        if (( rc == 0 )); then
            best="$context"
            log "probe passed at max_length=${context}"
        else
            if grep -Eiq 'out.?of.?memory|cuda out of memory' "$LOG_DIR/probe_${context}.log"; then
                log "probe OOM at max_length=${context}; stop probing higher values"
            else
                log "probe failed at max_length=${context} with exit=${rc}; stop probing"
            fi
            break
        fi
    done

    if (( best == 0 )); then
        log "no tested context fits; leaving monitor active for a later retry"
        return 1
    fi

    log "starting formal 3-epoch 2-GPU LoRA SFT with max_length=${best}"
    run_sft "$best" "$SFT_DATA" "$OUTPUT_DIR" 3 "$GRAD_ACCUM_STEPS"
}

log "waiting for two idle GPUs; local check interval=${POLL_SECONDS}s"
while true; do
    mapfile -t candidates < <(idle_gpus || true)
    if (( ${#candidates[@]} >= 2 )); then
        selected="${candidates[0]},${candidates[1]}"
        log "found idle GPUs=${selected}; preparing context probe"
        export CUDA_VISIBLE_DEVICES="$selected"
        probe_data="$(make_probe_data | tail -n 1)"
        if probe_and_train "$probe_data"; then
            rm -f "$probe_data"
            log "formal Stage-1 LoRA training completed"
            exit 0
        fi
        rm -f "$probe_data"
    elif (( ${#candidates[@]} == 0 )); then
        log "no idle GPU pair; next check in ${POLL_SECONDS}s"
    else
        log "only ${#candidates[@]} idle GPU (${candidates[*]}); next check in ${POLL_SECONDS}s"
    fi
    sleep "$POLL_SECONDS"
done
