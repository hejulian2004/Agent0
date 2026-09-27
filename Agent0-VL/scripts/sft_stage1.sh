#!/usr/bin/env bash
set -Eeuo pipefail

# Two-RTX-4090 Stage-1 SFT profile. The 4-bit base and frozen vision modules
# keep the 7B vision-language model within 24 GiB per GPU.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
SWIFT_BIN="$ROOT_DIR/.venv/bin/swift"
MODEL="${MODEL:-$ROOT_DIR/checkpoints/base/Qwen2.5-VL-7B-Instruct}"
SFT_DATA="${SFT_DATA:-data/sft/large/stage1_500.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_stage1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BSZ="${BSZ:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
EXPECTED_ROWS="${EXPECTED_ROWS:-500}"
REPORT_TO="${REPORT_TO:-none}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing project virtualenv Python: $PYTHON_BIN" >&2; exit 1; }
[[ -x "$SWIFT_BIN" ]] || { echo "Missing project virtualenv Swift: $SWIFT_BIN" >&2; exit 1; }
[[ -f "$SFT_DATA" ]] || { echo "Missing Stage-1 dataset: $SFT_DATA" >&2; exit 1; }
[[ -f scripts/prompt.txt ]] || { echo "Missing system prompt: scripts/prompt.txt" >&2; exit 1; }
if [[ -d "$MODEL" ]]; then
    [[ -f "$MODEL/config.json" && -f "$MODEL/model.safetensors.index.json" ]] || {
        echo "Incomplete local model: $MODEL" >&2
        exit 1
    }
fi
[[ "$NPROC_PER_NODE" == "2" ]] || { echo "This profile requires NPROC_PER_NODE=2" >&2; exit 1; }
[[ "$BSZ" =~ ^[1-9][0-9]*$ ]] || { echo "BSZ must be a positive integer" >&2; exit 1; }
[[ "$GRAD_ACCUM_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "GRAD_ACCUM_STEPS must be a positive integer" >&2; exit 1; }

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
(( ${#GPU_IDS[@]} == 2 )) || {
    echo "CUDA_VISIBLE_DEVICES must contain exactly two GPU IDs; got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
}

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-10}"
export MAX_PIXELS="${MAX_PIXELS:-3211264}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT_DIR/.cache/modelscope}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$(dirname "$OUTPUT_DIR")"

echo "[sft-stage1] validating $SFT_DATA"
"$PYTHON_BIN" -m tools.sft_builder.validate_sft \
    --stage 1 --input "$SFT_DATA" --expected-rows "$EXPECTED_ROWS"

echo "[sft-stage1] model=$MODEL"
echo "[sft-stage1] GPUs=$CUDA_VISIBLE_DEVICES, QLoRA NF4/BF16, ZeRO-3"
echo "[sft-stage1] batch=$BSZ, grad_accum=$GRAD_ACCUM_STEPS, max_length=$MAX_LENGTH"
echo "[sft-stage1] output=$OUTPUT_DIR"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[sft-stage1] preflight passed"
    exit 0
fi

# Paper Appendix B: both SFT stages use 1e-5.
"$SWIFT_BIN" sft \
    --model "$MODEL" \
    --use_hf true \
    --template qwen2_5_vl \
    --dataset "$SFT_DATA" \
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
    --num_train_epochs 3 \
    --per_device_train_batch_size "$BSZ" \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --save_strategy epoch \
    --max_length "$MAX_LENGTH" \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir "$OUTPUT_DIR" \
    --add_version false \
    --create_checkpoint_symlink false \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --report_to "$REPORT_TO"

LATEST_CHECKPOINT="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -n 1)"
[[ -n "$LATEST_CHECKPOINT" ]] || { echo "Training finished without a checkpoint in $OUTPUT_DIR" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR/last" || -L "$OUTPUT_DIR/last" ]] || {
    echo "$OUTPUT_DIR/last exists and is not a symlink" >&2
    exit 1
}
ln -sfn "$LATEST_CHECKPOINT" "$OUTPUT_DIR/last"
echo "[sft-stage1] completed; latest adapter=$OUTPUT_DIR/last"
