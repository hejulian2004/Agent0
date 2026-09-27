#!/usr/bin/env bash
set -Eeuo pipefail

# Local four-GPU teacher launcher for SFT data construction.
# This intentionally lives in Agent0-VL so the external teacher launcher and
# system CUDA installation remain untouched.

MODEL_DIR="${MODEL_DIR:-/mnt/d/qwen3.8-27B/model}"
VLLM_BIN="${VLLM_BIN:-/mnt/d/qwen3.8-27B/.venv/bin/vllm}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP_SIZE="${VLLM_TP_SIZE:-4}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${VLLM_MODEL_NAME:-qwen3.8-27b}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-49152}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MM_PROCESSOR_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-0.5}"
TEACHER_API_KEY_FILE="${TEACHER_API_KEY_FILE:-/mnt/d/Agent0/bench/.api_key}"

[[ -r "$TEACHER_API_KEY_FILE" ]] || {
    echo "Teacher API key file is not readable: $TEACHER_API_KEY_FILE" >&2
    exit 2
}
[[ -x "$VLLM_BIN" ]] || {
    echo "vLLM executable not found: $VLLM_BIN" >&2
    exit 2
}
[[ -d "$MODEL_DIR" ]] || {
    echo "Model directory not found: $MODEL_DIR" >&2
    exit 2
}

teacher_api_key="$(tr -d '\r\n' < "$TEACHER_API_KEY_FILE")"
[[ -n "$teacher_api_key" ]] || {
    echo "Teacher API key file is empty" >&2
    exit 2
}

echo "Starting ${MODEL_NAME} with TP=${TP_SIZE} on ${HOST}:${PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}; max_model_len=${MAX_MODEL_LEN}; max_num_seqs=${MAX_NUM_SEQS}; gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "API key is read from the configured file and is not printed"

exec env \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    VLLM_API_KEY="$teacher_api_key" \
    "$VLLM_BIN" serve "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$teacher_api_key"
