#!/usr/bin/env bash
# Experimental Megatron-SWIFT BF16 LoRA pipeline. NOT bitsandbytes QLoRA.
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
usage() {
    cat <<'HELP'
Usage: bash scripts/sft-agent0-pipeline-4090.sh [options]
  --gpus 0,1,2,3 --pp 1 --tp 4
  --batch-size 1             Global batch across data-parallel replicas
  --micro-batch-size 1 --epochs 3 --max-length 65536
  --model PATH --data PATH --output-dir PATH --workers 1
  --cross-entropy-impl te|native  Cross-entropy backend; te is the default
  --save-steps N              Save a checkpoint every N optimizer steps; 0 disables saving (default)
  --allow-bf16-lora          Explicitly select BF16 LoRA, not NF4 QLoRA
  --preflight-only           Check environment/data without training
  --dry-run                 Print command; no dependency imports or run creation
This is BF16 LoRA (not QLoRA), TP=4 with sequence parallel. Existing Swift/ZeRO3 checkpoints and
activation-offload plugins cannot be resumed here. By default no checkpoints are saved;
pass --save-steps N to save every N optimizer steps, retaining the latest two checkpoints
with optimizer and RNG state (saving is synchronous). The default TE cross-entropy backend
avoids the native backend's full-sequence FP32 vocabulary buffer;
uses Megatron's automatic Flash/cuDNN/unfused attention fallback; no automatic HF merge or
latest-model update.
HELP
}
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PP=1 TP=4 BATCH=1 MICRO=1 EPOCHS=3 LENGTH=65536 WORKERS=1 CE_IMPL=te
SAVE_STEPS=0 MODEL="$ROOT_DIR/checkpoints/base/Qwen2.5-VL-7B-Instruct"
DATA="$ROOT_DIR/data/sft/large/mixed_balanced_1000.jsonl"
OUTPUT="" ALLOW=0 PREFLIGHT=0 DRY=0
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --allow-bf16-lora) ALLOW=1; shift; continue ;;
        --preflight-only) PREFLIGHT=1; shift; continue ;;
        --dry-run) DRY=1; shift; continue ;;
        --gpus) var=GPU_LIST ;; --pp) var=PP ;; --tp) var=TP ;;
        --batch-size) var=BATCH ;; --micro-batch-size) var=MICRO ;;
        --epochs) var=EPOCHS ;; --max-length) var=LENGTH ;;
        --cross-entropy-impl) var=CE_IMPL ;;
        --save-steps) var=SAVE_STEPS ;;
        --model) var=MODEL ;; --data) var=DATA ;; --output-dir) var=OUTPUT ;;
        --workers) var=WORKERS ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    (($# >= 2)) && [[ "$2" != --* ]] || { echo "Missing value: $1" >&2; exit 2; }
    printf -v "$var" '%s' "$2"; shift 2
done
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPUs' >&2; exit 2; }
IFS=, read -r -a GPU_IDS <<< "$GPU_LIST"
NPROC=${#GPU_IDS[@]}
declare -A seen=()
for gpu in "${GPU_IDS[@]}"; do
    [[ -z "${seen[$gpu]:-}" ]] || { echo 'Duplicate GPU' >&2; exit 2; }
    seen[$gpu]=1
done
for name in PP TP BATCH MICRO EPOCHS LENGTH; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $name" >&2; exit 2; }
done
[[ "$SAVE_STEPS" =~ ^[0-9]+$ ]] || { echo 'Invalid --save-steps (expected a non-negative integer)' >&2; exit 2; }
SAVE_STEPS=$((10#$SAVE_STEPS))
[[ "$CE_IMPL" == te || "$CE_IMPL" == native ]] || { echo 'Invalid --cross-entropy-impl (expected te or native)' >&2; exit 2; }
[[ "$WORKERS" =~ ^[0-9]+$ ]] || { echo 'Invalid workers' >&2; exit 2; }
(( NPROC % (PP * TP) == 0 )) || { echo 'GPU count must be divisible by PP*TP' >&2; exit 2; }
DP=$((NPROC / PP / TP))
(( BATCH % (MICRO * DP) == 0 )) || { echo 'Global batch must divide micro-batch*DP' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-pipeline/bin/python}"
MEGATRON_BIN="$(dirname "$PYTHON_BIN")/megatron"
OUTPUT="${OUTPUT:-$ROOT_DIR/checkpoints/sft_pipeline/sft_pipeline_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST" NPROC_PER_NODE="$NPROC"
CUDA_TOOLKIT="$ROOT_DIR/.venv/lib/python3.10/site-packages/nvidia/cu13"
if [[ -x "$CUDA_TOOLKIT/bin/nvcc" ]]; then
    export CUDA_HOME="${CUDA_HOME:-$CUDA_TOOLKIT}"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
if [[ -d "$ROOT_DIR/.venv/lib/python3.10/site-packages/nvidia/cudnn" ]]; then
    export CUDNN_HOME="${CUDNN_HOME:-$ROOT_DIR/.venv/lib/python3.10/site-packages/nvidia/cudnn}"
    export LD_LIBRARY_PATH="$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MAX_PIXELS="${MAX_PIXELS:-3211264}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AGENT0_DATA_MEMORY_GUARD=1 AGENT0_DATA_MEMORY_PERCENT=90 AGENT0_DATA_MEMORY_WAIT_SECONDS=180
export PYTHONPATH="$ROOT_DIR/tools/runtime_guard:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
SAVE_ARGS=(--save_strategy steps --save_steps "$SAVE_STEPS" --async_save false)
if (( SAVE_STEPS > 0 )); then
    SAVE_ARGS+=(--save_total_limit 2 --no_save_optim false --no_save_rng false)
else
    SAVE_ARGS+=(--no_save_optim true --no_save_rng true)
fi
CMD=("$MEGATRON_BIN" sft --model "$MODEL" --dataset "$DATA" --system "$ROOT_DIR/scripts/prompt.txt"
    --external_plugins "$ROOT_DIR/tools/training/megatron_save_control.py"
    --tuner_type lora --lora_rank 16 --lora_alpha 64 --target_modules all-linear
    --torch_dtype bfloat16 --freeze_vit true --freeze_aligner true
    --pipeline_model_parallel_size "$PP" --tensor_model_parallel_size "$TP" --sequence_parallel true
    --attention_backend auto
    --micro_batch_size "$MICRO" --global_batch_size "$BATCH" --num_train_epochs "$EPOCHS"
    --max_length "$LENGTH" --truncation_strategy left --strict true --packing false --split_dataset_ratio 0
    --lr 1e-5 --lr_warmup_fraction 0.05
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1
    --cross_entropy_loss_fusion true --cross_entropy_fusion_impl "$CE_IMPL" --vit_gradient_checkpointing true
    "${SAVE_ARGS[@]}" --output_dir "$OUTPUT/adapter" --logging_steps 1
    --dataloader_num_workers "$WORKERS" --dataloader_prefetch_factor 1 --dataloader_pin_memory false
    --dataset_num_proc 1)
printf '[sft-pipeline] BF16 LoRA (NOT QLoRA), GPUs=%s PP=%s TP=%s DP=%s global_batch=%s ce_impl=%s save_steps=%s\n' "$GPU_LIST" "$PP" "$TP" "$DP" "$BATCH" "$CE_IMPL" "$SAVE_STEPS"
if (( DRY )); then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
(( ALLOW )) || { echo 'This backend uses BF16 LoRA. Pass --allow-bf16-lora explicitly; current NF4 QLoRA script is unchanged.' >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -x "$MEGATRON_BIN" && -f "$DATA" && -f "$MODEL/config.json" ]] || { echo 'Missing Python/megatron/model/data' >&2; exit 1; }
"$PYTHON_BIN" - "$CE_IMPL" <<'PY'
import importlib, sys
for module in ('megatron.core', 'mcore_bridge', 'transformer_engine.pytorch'):
    try:
        importlib.import_module(module)
    except Exception as error:
        raise SystemExit(f'Megatron dependency unavailable: {module}: {error}. Install a compatible Megatron-SWIFT environment first; no automatic installation.')
from swift.megatron.arguments.megatron_args import MegatronArguments
if sys.argv[1] == 'te':
    from megatron.core.models.common.language_module.language_module import te_parallel_cross_entropy
    if te_parallel_cross_entropy is None:
        raise SystemExit('Transformer Engine parallel cross entropy is unavailable')
print('[sft-pipeline] Megatron imports passed')
PY
"$PYTHON_BIN" -m tools.sft_builder.validate_sft --stage 2 --allow-stage2-images --input "$DATA" --expected-rows 1000
"$PYTHON_BIN" - "$MODEL" "$PP" <<'PY'
import json, sys
from pathlib import Path
config=json.loads((Path(sys.argv[1])/'config.json').read_text())
if config.get('model_type') != 'qwen2_5_vl': raise SystemExit('Expected Qwen2.5-VL model')
layers=config.get('text_config',config).get('num_hidden_layers')
if layers is None or layers % int(sys.argv[2]): raise SystemExit('Decoder layer count must divide PP')
PY
for gpu in "${GPU_IDS[@]}"; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    (( used <= 512 )) || { echo "GPU $gpu occupied: ${used}MiB" >&2; exit 1; }
done
if (( PREFLIGHT )); then echo '[sft-pipeline] preflight passed; model execution and memory feasibility unverified'; exit 0; fi
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 1; }
mkdir -p "$OUTPUT"
STATUS=failed CHILD=""
cleanup() {
    code=$?
    if [[ -n "$CHILD" ]]; then kill -- "-$CHILD" 2>/dev/null || true; wait "$CHILD" 2>/dev/null || true; fi
    sed -i "s/^- Status: running$/- Status: $STATUS/" "$OUTPUT/README.md"
    printf '\n- Finished: %s; exit code: %s\n' "$(date -Iseconds)" "$code" >> "$OUTPUT/README.md"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$PYTHON_BIN" tools/runtime_guard/normalize_te_flashattn_version.py
{
    printf '# Experimental pipeline SFT\n\n- Status: running\n- Created: %s\n' "$(date -Iseconds)"
    printf -- '- Model: %s\n- Dataset: %s\n- GPUs: %s; PP=%s TP=%s DP=%s\n' "$MODEL" "$DATA" "$GPU_LIST" "$PP" "$TP" "$DP"
    printf -- '- Precision: BF16 base and LoRA; no NF4 quantization; rank16/alpha64/all-linear, frozen vision/aligner\n'
    printf -- '- Attention: auto backend; TE compares FlashAttention public version after stripping only the local CUDA/Torch build suffix\n'
    printf -- '- Cross-entropy: %s fused vocabulary-parallel implementation\n' "$CE_IMPL"
    printf -- '- Limits: context=%s global_batch=%s micro_batch=%s epochs=%s workers=%s\n' "$LENGTH" "$BATCH" "$MICRO" "$EPOCHS" "$WORKERS"
    if (( SAVE_STEPS > 0 )); then
        printf -- '- Checkpoints: every %s optimizer steps, latest two retained, optimizer/RNG retained; synchronous Megatron save\n' "$SAVE_STEPS"
    else
        printf -- '- Checkpoints: disabled (including Swift final-step auto-save via local callback guard)\n'
    fi
    printf -- '- Log: training.log\n- Command: '
    printf '%q ' "${CMD[@]}"; printf '\n'
} > "$OUTPUT/README.md"
setsid "${CMD[@]}" > >(tee -a "$OUTPUT/training.log") 2>&1 &
CHILD=$!
wait "$CHILD"
CHILD="" STATUS=success
echo "[sft-pipeline] complete: $OUTPUT; export/merge must be performed separately before RL"
