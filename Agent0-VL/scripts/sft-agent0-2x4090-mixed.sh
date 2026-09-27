#!/usr/bin/env bash
set -Eeuo pipefail

# Single-stage balanced SFT on two or four RTX 4090s.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"


usage() {
    cat <<'HELP'
Usage: bash scripts/sft-agent0-2x4090-mixed.sh [options]
  --gpus 2,3                GPU IDs; process count follows GPU count
  --batch-size 1            Per-GPU training batch
  --grad-accum 64            Accumulation steps (default targets global batch 128)
  --epochs 3 --max-length 10240 --learning-rate 1e-5 --workers 4 (or --concurrency; CPU dataloader workers)
  --data PATH --model PATH --output-dir PATH --run-id NAME
  --activation-cpu-offload auto|true|false  Saved tensors offload (default auto)
  --offload-token-threshold 8192 --offload-min-free-gb 8
  --resume-from-checkpoint PATH  Resume full training state (use mixed_adapter/last-checkpoint)
  --preflight-only           Audit data without training
  --dry-run                  Print resolved settings without creating a run
  --workers 1 --prefetch-factor 1 --memory-limit-percent 90 --memory-wait-seconds 180
CLI options override environment variables. SFT uses data parallelism;
its effective batch is GPU count * batch-size * grad-accum.
HELP
}
while (($#)); do
    option="$1"
    case "$option" in
        -h|--help) usage; exit 0 ;;
        --memory-limit-percent) variable=DATA_MEMORY_PERCENT ;;
        --memory-wait-seconds) variable=DATA_MEMORY_WAIT_SECONDS ;;
        --prefetch-factor) variable=DATA_PREFETCH_FACTOR ;;
        --dry-run) DRY_RUN=1; shift; continue ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift; continue ;;
        --resume-from-checkpoint) variable=RESUME_CHECKPOINT ;;
        --activation-cpu-offload) variable=SFT_ACTIVATION_CPU_OFFLOAD ;;
        --offload-token-threshold) variable=SFT_OFFLOAD_TOKEN_THRESHOLD ;;
        --offload-min-free-gb) variable=SFT_OFFLOAD_MIN_FREE_GB ;;
        --gpus) variable=CUDA_VISIBLE_DEVICES ;;
        --batch-size) variable=BSZ ;;
        --grad-accum) variable=GRAD_ACCUM_STEPS ;;
        --epochs) variable=NUM_TRAIN_EPOCHS ;;
        --max-length) variable=MAX_LENGTH ;;
        --data) variable=SFT_DATA ;;
        --model) variable=BASE_MODEL ;;
        --run-id) variable=RUN_ID ;;
        --output-dir) variable=RUN_ROOT ;;
        --workers|--concurrency) variable=DATALOADER_WORKERS ;;
        --learning-rate) variable=LEARNING_RATE ;;
        *) echo "Unknown option: $option" >&2; usage >&2; exit 2 ;;
    esac
    (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || { echo "Missing value for $option" >&2; exit 2; }
    printf -v "$variable" '%s' "$2"
    shift 2
done
SFT_ACTIVATION_CPU_OFFLOAD="${SFT_ACTIVATION_CPU_OFFLOAD:-auto}"
SFT_OFFLOAD_TOKEN_THRESHOLD="${SFT_OFFLOAD_TOKEN_THRESHOLD:-8192}"
SFT_OFFLOAD_MIN_FREE_GB="${SFT_OFFLOAD_MIN_FREE_GB:-8}"
[[ "$SFT_ACTIVATION_CPU_OFFLOAD" == auto || "$SFT_ACTIVATION_CPU_OFFLOAD" == true || "$SFT_ACTIVATION_CPU_OFFLOAD" == false ]] || {
    echo "activation-cpu-offload must be auto, true or false" >&2; exit 2;
}
[[ "$SFT_OFFLOAD_TOKEN_THRESHOLD" =~ ^[1-9][0-9]*$ && "$SFT_OFFLOAD_MIN_FREE_GB" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] || {
    echo "Invalid offload token threshold/free GiB" >&2; exit 2;
}
export SFT_ACTIVATION_CPU_OFFLOAD SFT_OFFLOAD_TOKEN_THRESHOLD SFT_OFFLOAD_MIN_FREE_GB
DATA_MEMORY_PERCENT="${DATA_MEMORY_PERCENT:-90}"
DATA_MEMORY_WAIT_SECONDS="${DATA_MEMORY_WAIT_SECONDS:-180}"
DATA_PREFETCH_FACTOR="${DATA_PREFETCH_FACTOR:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-1}"
[[ "$DATA_MEMORY_PERCENT" =~ ^[1-9][0-9]*$ ]] && (( DATA_MEMORY_PERCENT <= 90 )) || { echo "memory-limit-percent must be 1..90" >&2; exit 2; }
[[ "$DATA_MEMORY_WAIT_SECONDS" =~ ^[1-9][0-9]*$ && "$DATA_PREFETCH_FACTOR" =~ ^[1-9][0-9]*$ && "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] || { echo "Invalid loading workers/prefetch/timeout" >&2; exit 2; }
export AGENT0_DATA_MEMORY_GUARD=1
export AGENT0_DATA_MEMORY_PERCENT="$DATA_MEMORY_PERCENT"
export AGENT0_DATA_MEMORY_WAIT_SECONDS="$DATA_MEMORY_WAIT_SECONDS"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
SWIFT_BIN="$ROOT_DIR/.venv/bin/swift"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/checkpoints/base/Qwen2.5-VL-7B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${#GPU_IDS[@]}"
RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_CHECKPOINT="$(realpath -e "$RESUME_CHECKPOINT")"
    [[ -s "$RESUME_CHECKPOINT/trainer_state.json" && -f "$RESUME_CHECKPOINT/adapter_config.json" ]] || {
        echo "Checkpoint is missing training state or adapter config: $RESUME_CHECKPOINT" >&2; exit 1;
    }
    [[ -n "$(find "$RESUME_CHECKPOINT" -name '*optim_states.pt' -size +0c -print -quit)" ]] || {
        echo "Checkpoint has no DeepSpeed optimizer state: $RESUME_CHECKPOINT" >&2; exit 1;
    }
    RESUME_RUN_ROOT="$(dirname "$(dirname "$RESUME_CHECKPOINT")")"
    if [[ -n "${RUN_ROOT:-}" && "$(realpath -m "$RUN_ROOT")" != "$RESUME_RUN_ROOT" ]]; then
        echo "Resume must use the original run directory: $RESUME_RUN_ROOT" >&2; exit 2
    fi
    RUN_ROOT="$RESUME_RUN_ROOT"
    RUN_ID="$(basename "$RUN_ROOT")"
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_CHECKPOINT" --resume_only_model false)
fi
RUN_ID="${RUN_ID:-sft_2x4090_mixed_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-checkpoints/sft_2x4090/$RUN_ID}"
ADAPTER_OUTPUT="${ADAPTER_OUTPUT:-$RUN_ROOT/mixed_adapter}"
MERGED_OUTPUT="${MERGED_OUTPUT:-$RUN_ROOT/mixed_merged}"
MIXED_DATA="${SFT_DATA:-data/sft/large/mixed_balanced_1000.jsonl}"
MIXED_MANIFEST="data/sft/balanced_1000_v2/manifest.json"
MAX_LENGTH="${MAX_LENGTH:-10240}"
BSZ="${BSZ:-1}"
[[ "$BSZ" =~ ^[1-9][0-9]*$ ]] || { echo "batch-size must be positive" >&2; exit 2; }
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$(((128 + NPROC_PER_NODE * BSZ - 1) / NPROC_PER_NODE / BSZ))}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-1}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU list" >&2; exit 2; }
declare -A seen_gpus=()
for gpu_id in "${GPU_IDS[@]}"; do
    [[ -z "${seen_gpus[$gpu_id]:-}" ]] || { echo "Duplicate GPU ID: $gpu_id" >&2; exit 2; }
    seen_gpus[$gpu_id]=1
done
[[ "$GRAD_ACCUM_STEPS" =~ ^[1-9][0-9]*$ && "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid accumulation/length" >&2; exit 2; }
[[ "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] || { echo "workers must be a nonnegative integer" >&2; exit 2; }
[[ "${NUM_TRAIN_EPOCHS:-3}" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || { echo "epochs must be positive" >&2; exit 2; }
[[ "$LEARNING_RATE" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ && "$LEARNING_RATE" != 0 ]] || { echo "Invalid learning rate" >&2; exit 2; }
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'activation_cpu_offload=%s token_threshold=%s min_free_gb=%s\n' "$SFT_ACTIVATION_CPU_OFFLOAD" "$SFT_OFFLOAD_TOKEN_THRESHOLD" "$SFT_OFFLOAD_MIN_FREE_GB"
    printf 'loading: workers=%s prefetch=%s memory_threshold=%s%% wait_timeout=%ss\n' "$DATALOADER_WORKERS" "$DATA_PREFETCH_FACTOR" "$DATA_MEMORY_PERCENT" "$DATA_MEMORY_WAIT_SECONDS"
    printf 'GPUs=%s processes=%s per_gpu_batch=%s grad_accum=%s global_batch=%s max_length=%s epochs=%s lr=%s workers=%s data=%s\n' "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$BSZ" "$GRAD_ACCUM_STEPS" "$((NPROC_PER_NODE * BSZ * GRAD_ACCUM_STEPS))" "$MAX_LENGTH" "${NUM_TRAIN_EPOCHS:-3}" "$LEARNING_RATE" "$DATALOADER_WORKERS" "$MIXED_DATA"
    exit 0
fi
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
REPORT_TO="${REPORT_TO:-none}"
LOG_FILE="$RUN_ROOT/training.log"
RUN_README="$RUN_ROOT/README.md"
PROMPT_SHA256="$(sha256sum scripts/prompt.txt | awk '{print $1}')"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing project venv Python: $PYTHON_BIN" >&2; exit 1; }
[[ -x "$SWIFT_BIN" ]] || { echo "Missing ms-swift executable: $SWIFT_BIN" >&2; exit 1; }
[[ -f "$BASE_MODEL/config.json" && -f "$BASE_MODEL/model.safetensors.index.json" ]] || {
    echo "Incomplete local base model: $BASE_MODEL" >&2
    exit 1
}
for shard in "$BASE_MODEL"/model-0000{1..5}-of-00005.safetensors; do
    [[ -f "$shard" ]] || { echo "Missing model shard: $shard" >&2; exit 1; }
done
[[ -f "$MIXED_DATA" ]] || { echo "Missing generated SFT data: $MIXED_DATA" >&2; exit 1; }
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo "Select at least one GPU" >&2; exit 1; }
[[ "$BSZ" =~ ^[1-9][0-9]*$ ]] || { echo "BSZ must be a positive integer" >&2; exit 1; }
[[ "$GRAD_ACCUM_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "GRAD_ACCUM_STEPS must be a positive integer" >&2; exit 1; }
[[ "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_LENGTH must be a positive integer" >&2; exit 1; }
[[ "$NUM_TRAIN_EPOCHS" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || {
    echo "NUM_TRAIN_EPOCHS must be a positive number" >&2
    exit 1
}
(( ${#GPU_IDS[@]} == NPROC_PER_NODE )) || { echo "GPU count differs from NPROC_PER_NODE" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" || -n "${RESUME_CHECKPOINT:-}" ]] || {
    echo "Run directory already exists; choose a new RUN_ID/RUN_ROOT: $RUN_ROOT" >&2
    exit 1
}

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-10}"
export MAX_PIXELS="${MAX_PIXELS:-3211264}"
export PYTHONPATH="$ROOT_DIR/tools/runtime_guard:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT_DIR/.cache/modelscope}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$RUN_ROOT" "$HF_HOME" "$MODELSCOPE_CACHE" "$(dirname "$MIXED_DATA")" "$(dirname "$ADAPTER_OUTPUT")"
START_TIME="$(date '+%F %T %z')"
RUN_STATUS="failed"
training_pid=""

record_exit() {
    local exit_code=$?
    if [[ -n "$training_pid" ]]; then
        kill -- "-$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
    fi
    if [[ -f "$RUN_README" ]]; then
        sed -i "s/^- Status: running$/- Status: $RUN_STATUS/" "$RUN_README"
        printf '%s\n' "" "- Last update: $(date '+%F %T %z')" "- Exit code: $exit_code" >> "$RUN_README"
    fi
}
trap record_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "${RESUME_CHECKPOINT:-}" && -f "$RUN_README" ]]; then
    sed -i 's/^- Status: .*/- Status: running/' "$RUN_README"
    printf '\n- Resumed: %s; checkpoint: %s; GPUs: %s; batch: %s; accumulation: %s; max_length: %s; total epochs: %s\n' \
        "$START_TIME" "$RESUME_CHECKPOINT" "$CUDA_VISIBLE_DEVICES" "$BSZ" "$GRAD_ACCUM_STEPS" "$MAX_LENGTH" "$NUM_TRAIN_EPOCHS" >> "$RUN_README"
else
printf '%s\n' \
    "# Mixed SFT run: $RUN_ID" \
    "" \
    "- Status: running" \
    "- Created: $START_TIME" \
    "- Training mode: one single-stage balanced nine-source dataset" \
    "- Base model: $BASE_MODEL" \
    "- Mixed dataset: $MIXED_DATA" \
    "- Dataset manifest: $MIXED_MANIFEST" \
    "- System prompt: scripts/prompt.txt (sha256=$PROMPT_SHA256)" \
    "- Adapter output: $ADAPTER_OUTPUT" \
    "- Merged model: $MERGED_OUTPUT" \
    "- Log: $LOG_FILE" \
    "- GPUs: $CUDA_VISIBLE_DEVICES ($NPROC_PER_NODE x RTX 4090 profile)" \
    "- Precision: QLoRA NF4, BF16 compute/storage, double quantization" \
    "- LoRA: rank 16, alpha 64, all-linear; vision tower and aligner frozen" \
    "- Learning rate: $LEARNING_RATE; dataloader workers: $DATALOADER_WORKERS" \
    "- Data loading: workers=$DATALOADER_WORKERS, prefetch=$DATA_PREFETCH_FACTOR, memory pause threshold=$DATA_MEMORY_PERCENT%, wait timeout=$DATA_MEMORY_WAIT_SECONDS seconds" \
    "- Limits: max_length=$MAX_LENGTH, per-device batch=$BSZ, grad_accum=$GRAD_ACCUM_STEPS, epochs=$NUM_TRAIN_EPOCHS" \
    "- Activation CPU offload: $SFT_ACTIVATION_CPU_OFFLOAD; saved tensors, non-pinned CPU copies; token threshold=$SFT_OFFLOAD_TOKEN_THRESHOLD; min free GiB=$SFT_OFFLOAD_MIN_FREE_GB" \
    "- Memory kernels: use_liger_kernel=${USE_LIGER_KERNEL:-true}; allocator=$PYTORCH_CUDA_ALLOC_CONF" \
    "- Distributed: $NPROC_PER_NODE processes, DeepSpeed ZeRO-3" \
    "- Command: bash scripts/sft-agent0-2x4090-mixed.sh" \
    > "$RUN_README"
fi
printf '\n- Checkpoint policy: every 10 optimizer steps, keep latest checkpoint; model, optimizer, scheduler, RNG and trainer state saved.\n' >> "$RUN_README"

log() {
    printf '[sft-mixed] %s\n' "$*" | tee -a "$LOG_FILE"
}

log "run=$RUN_ID GPUs=$CUDA_VISIBLE_DEVICES"
log "auditing single-stage balanced SFT data"
"$PYTHON_BIN" -m tools.sft_builder.validate_sft --stage 2 --allow-stage2-images \
    --input "$MIXED_DATA" --expected-rows 1000 2>&1 | tee -a "$LOG_FILE"
log "training rows=1000 model=$BASE_MODEL GPUs=$CUDA_VISIBLE_DEVICES"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    log "preflight passed; training not started"
    RUN_STATUS="success"
    exit 0
fi

OFFLOAD_ARGS=()
if [[ "$SFT_ACTIVATION_CPU_OFFLOAD" != false ]]; then
    [[ -f "$ROOT_DIR/tools/training/sft_activation_offload.py" ]] || { echo "Missing activation offload plugin" >&2; exit 1; }
    OFFLOAD_ARGS=(--external_plugins "$ROOT_DIR/tools/training/sft_activation_offload.py")
fi
log "activation_cpu_offload=$SFT_ACTIVATION_CPU_OFFLOAD token_threshold=$SFT_OFFLOAD_TOKEN_THRESHOLD min_free_gb=$SFT_OFFLOAD_MIN_FREE_GB"
LOADING_ARGS=(--dataloader_pin_memory false --dataset_num_proc 1 --lazy_tokenize true)
if (( DATALOADER_WORKERS > 0 )); then
    LOADING_ARGS+=(--dataloader_prefetch_factor "$DATA_PREFETCH_FACTOR")
fi
log "loading workers=$DATALOADER_WORKERS prefetch=$DATA_PREFETCH_FACTOR memory_threshold=$DATA_MEMORY_PERCENT%"
setsid "$SWIFT_BIN" sft \
    --model "$BASE_MODEL" \
    --use_hf true \
    --template qwen2_5_vl \
    --dataset "$MIXED_DATA" \
    --strict true \
    --tuner_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_storage bfloat16 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --lora_rank 16 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --system scripts/prompt.txt \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$BSZ" \
    --per_device_eval_batch_size 1 \
    --learning_rate "$LEARNING_RATE" \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --use_liger_kernel "${USE_LIGER_KERNEL:-true}" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --save_strategy steps \
    --save_steps 10 \
    --save_only_model false \
    --max_length "$MAX_LENGTH" \
    --save_total_limit 1 \
    --logging_steps 1 \
    --output_dir "$ADAPTER_OUTPUT" \
    --add_version false \
    --create_checkpoint_symlink true \
    --warmup_ratio 0.05 \
    --dataloader_num_workers "$DATALOADER_WORKERS" \
    "${LOADING_ARGS[@]}" \
    "${OFFLOAD_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --report_to "$REPORT_TO" \
    > >(tee -a "$LOG_FILE") 2>&1 &
training_pid=$!
wait "$training_pid"
training_pid=""

LATEST_CHECKPOINT="$(find "$ADAPTER_OUTPUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -n 1)"
[[ -n "$LATEST_CHECKPOINT" ]] || {
    echo "Training completed without a checkpoint in $ADAPTER_OUTPUT" | tee -a "$LOG_FILE" >&2
    exit 1
}
[[ ! -e "$ADAPTER_OUTPUT/last" || -L "$ADAPTER_OUTPUT/last" ]] || {
    echo "$ADAPTER_OUTPUT/last exists and is not a symlink" | tee -a "$LOG_FILE" >&2
    exit 1
}
ln -sfn "$LATEST_CHECKPOINT" "$ADAPTER_OUTPUT/last"

log "merging the trained adapter into a standalone model"
"$SWIFT_BIN" export \
    --model "$BASE_MODEL" \
    --adapters "$ADAPTER_OUTPUT/last" \
    --merge_lora true \
    --safe_serialization true \
    --output_dir "$MERGED_OUTPUT" \
    2>&1 | tee -a "$LOG_FILE"
[[ -f "$MERGED_OUTPUT/config.json" ]] || {
    echo "Adapter merge did not produce a valid model: $MERGED_OUTPUT" | tee -a "$LOG_FILE" >&2
    exit 1
}
ln -sfn "$(realpath --relative-to="$(dirname "$RUN_ROOT")" "$MERGED_OUTPUT")" \
    "$(dirname "$RUN_ROOT")/latest_mixed_merged"

log "mixed SFT completed; merged model=$MERGED_OUTPUT"
RUN_STATUS="success"
