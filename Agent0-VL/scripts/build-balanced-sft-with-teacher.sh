#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-4}"
export TEACHER_CONCURRENCY="${TEACHER_CONCURRENCY:-32}"
export TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-600}"
export TEACHER_MAX_STEPS="${TEACHER_MAX_STEPS:-16}"
export TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-8192}"
export TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-49152}"
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-32}"
TEACHER_BATCHED_TOKENS="${TEACHER_BATCHED_TOKENS:-16384}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
(( ${#GPU_IDS[@]} == TEACHER_TP_SIZE )) || { echo "GPU count must match teacher TP size" >&2; exit 1; }
[[ "$TEACHER_TP_SIZE" == 4 || "$TEACHER_TP_SIZE" == 2 ]] || { echo "Use 2 or 4 teacher GPUs" >&2; exit 1; }
[[ "$TEACHER_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] && (( TEACHER_CONCURRENCY <= 64 )) || { echo "TEACHER_CONCURRENCY must be 1..64" >&2; exit 1; }
for numeric_setting in TEACHER_MAX_STEPS TEACHER_MAX_TOKENS TEACHER_TIMEOUT TEACHER_MAX_MODEL_LEN TEACHER_MAX_NUM_SEQS; do
    [[ "${!numeric_setting}" =~ ^[1-9][0-9]*$ ]] || { echo "$numeric_setting must be a positive integer" >&2; exit 2; }
done
# Check data before loading a large teacher or creating run logs.
bash scripts/rebuild-balanced-data.sh preflight
RUN_NAME="balanced_sft_build_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PWD/logs/$RUN_NAME"
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/README.md" <<EOF
# Balanced SFT data generation

- Status: running
- Created: $(date '+%F %T %z')
- Teacher/base model: /mnt/d/qwen3.8-27B/model (qwen3.8-27b)
- Dataset: nine paper SFT sources, 112 + 8*111 rows
- Output: data/sft/large/mixed_balanced_1000.jsonl
- GPU setup: teacher GPUs $CUDA_DEVICES, TP=$TEACHER_TP_SIZE, BF16, FP8 KV; no LoRA
- Teacher limits: context $TEACHER_MAX_MODEL_LEN, max sequences $TEACHER_MAX_NUM_SEQS, batched tokens $TEACHER_BATCHED_TOKENS
- Builder: concurrent requests $TEACHER_CONCURRENCY, max response tokens $TEACHER_MAX_TOKENS, max Solver steps $TEACHER_MAX_STEPS; main prompts with user-authorized final-answer constraint
- Logs: generation.log, teacher.log
- Command: bash scripts/build-balanced-sft-with-teacher.sh
EOF
teacher_pid=""
builder_pid=""
cleanup() {
    status=$?
    for job_pid in "$builder_pid" "$teacher_pid"; do
        [[ -n "$job_pid" ]] || continue
        kill -- "-$job_pid" 2>/dev/null || true
        for attempt in $(seq 1 20); do
            kill -0 "$job_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$job_pid" 2>/dev/null || true
        wait "$job_pid" 2>/dev/null || true
    done
    if ((status == 0)); then label=success; else label=failed; fi
    sed -i "s/^- Status: running$/- Status: $label/" "$RUN_DIR/README.md"
    printf '\n- Last write: %s\n- Exit code: %s\n' "$(date '+%F %T %z')" "$status" >> "$RUN_DIR/README.md"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$RUN_DIR/generation.log") 2>&1
printf '[balanced-sft] logs=%s\n' "$RUN_DIR"
KEY_FILE="${TEACHER_API_KEY_FILE:-/mnt/d/Agent0/bench/.api_key}"
[[ -r "$KEY_FILE" ]] || { echo "Teacher API key file missing"; exit 1; }
export OPENAI_API_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
export VLLM_API_KEY="$OPENAI_API_KEY"
if ! curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    while IFS=',' read -r gpu used; do
        (( ${used// /} < 512 )) || { echo "Teacher GPU $gpu is occupied"; exit 1; }
    done < <(nvidia-smi -i "$CUDA_DEVICES" --query-gpu=index,memory.used --format=csv,noheader,nounits)
    setsid env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" /mnt/d/qwen3.8-27B/.venv/bin/vllm serve /mnt/d/qwen3.8-27B/model \
      --served-model-name qwen3.8-27b --tensor-parallel-size "$TEACHER_TP_SIZE" \
      --max-model-len "$TEACHER_MAX_MODEL_LEN" --max-num-seqs "$TEACHER_MAX_NUM_SEQS" --max-num-batched-tokens "$TEACHER_BATCHED_TOKENS" \
      --enable-chunked-prefill --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 \
      --enable-prefix-caching --mamba-cache-mode align \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
      --mm-processor-cache-gb 0.5 --host 127.0.0.1 --port 8000 > "$RUN_DIR/teacher.log" 2>&1 &
    teacher_pid=$!
    for attempt in $(seq 1 180); do
        kill -0 "$teacher_pid" 2>/dev/null || { echo "Teacher startup failed; see teacher.log"; exit 1; }
        if curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
        printf '[balanced-sft] teacher loading, elapsed ~%ss\n' "$((attempt * 5))"
        tail -n 1 "$RUN_DIR/teacher.log" | tr '\r' '\n' | tail -n 1
        sleep 5
    done
    curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null
fi
# Validate the existing or newly started endpoint without printing credentials.
.venv/bin/python - <<'PY_CHECK'
import json
import os
import urllib.request
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/models",
    headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=15) as response:
    models = json.load(response)["data"]
    ids = [row["id"] for row in models]
expected = os.environ.get("TEACHER_MODEL", "qwen3.8-27b")
if expected not in ids:
    raise SystemExit("Teacher endpoint model does not match " + expected)
model = next(row for row in models if row["id"] == expected)
required_context = int(os.environ["TEACHER_MAX_MODEL_LEN"])
actual_context = model.get("max_model_len")
if actual_context is None or int(actual_context) < required_context:
    raise SystemExit(f"Teacher context {actual_context} does not meet required {required_context}; stop the old teacher and restart with the new profile")
print(f"[balanced-sft] teacher context verified: {actual_context}", flush=True)
print("[balanced-sft] teacher endpoint verified: " + expected, flush=True)
PY_CHECK
run_builder() {
    setsid bash scripts/rebuild-balanced-data.sh "$1" &
    builder_pid=$!
    wait "$builder_pid"
    builder_pid=""
}
printf '[balanced-sft] GPUs=%s TP=%s concurrent_requests=%s max_model_len=%s max_num_seqs=%s max_solver_steps=%s max_response_tokens=%s\n' "$CUDA_DEVICES" "$TEACHER_TP_SIZE" "$TEACHER_CONCURRENCY" "$TEACHER_MAX_MODEL_LEN" "$TEACHER_MAX_NUM_SEQS" "$TEACHER_MAX_STEPS" "$TEACHER_MAX_TOKENS"
run_builder sft
run_builder rl
