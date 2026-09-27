#!/usr/bin/env bash
set -Eeuo pipefail

# Configurable RTX 4090 RL pipeline (default: two GPUs).
#
# Default:
#   external correctness warm-up (3 epochs)
#   -> checkpoint under one run directory
#   -> SERC RL resumed from that checkpoint (1 epoch)
#
# Skip the warm-up with:
#   SKIP_WARMUP=1 bash scripts/rl-agent0-2x4090-optimized.sh
# Resume a failed warm-up from its last saved step with:
#   RESUME_RUN_DIR=/path/to/previous/run bash scripts/rl-agent0-2x4090-optimized.sh
#
# PHASE=external_warmup and PHASE=serc remain available for running one phase
# explicitly. The script never changes the host CUDA driver or system Python.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"


usage() {
    cat <<'HELP'
Usage: bash scripts/rl-agent0-2x4090-optimized.sh [options]
  --gpus 2,3 --tp 2           GPU IDs and tensor parallelism (TP defaults to GPU count)
  --batch-size 2              Global prompt batch (not per GPU)
  --concurrency 16            vLLM max_num_seqs scheduling cap
  --rollout-n 8               Trajectories per prompt
  --mini-batch-size 1 --micro-batch-size 1
  --warmup-epochs 3 --epochs 1 --warmup-steps N --steps N
  --max-model-len 9216 --max-prompt-length 6144 --max-response-length 3072
  --max-batched-tokens N --ppo-max-tokens N --gpu-memory-utilization 0.65
  --data PATH --warmup-data PATH --val-data PATH --model PATH
  --output-dir PATH --resume-dir PATH --phase full|external_warmup|serc
  --save-freq 5 --progress-interval 30 --skip-warmup --dry-run
  --workers 1 --prefetch-factor 1 --memory-limit-percent 90 --memory-wait-seconds 180
Step targets default to floor(dataset rows / batch-size) * phase epochs.
--steps specifies additional formal steps. CLI overrides environment variables.
Dry-run uses 200 rows per phase if data is not yet present.
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
        --skip-warmup) SKIP_WARMUP=1; shift; continue ;;
        --workers) variable=DATALOADER_WORKERS ;;
        --gpus) variable=CUDA_DEVICES ;;
        --batch-size) variable=TRAIN_BATCH_SIZE ;;
        --concurrency|--max-num-seqs) variable=MAX_NUM_SEQS ;;
        --rollout-n) variable=ROLLOUT_N ;;
        --tp) variable=TENSOR_MODEL_PARALLEL_SIZE ;;
        --mini-batch-size) variable=PPO_MINI_BATCH_SIZE ;;
        --micro-batch-size) variable=PPO_MICRO_BATCH_SIZE_PER_GPU ;;
        --max-model-len) variable=MAX_MODEL_LEN ;;
        --max-batched-tokens) variable=MAX_NUM_BATCHED_TOKENS ;;
        --max-prompt-length) variable=MAX_PROMPT_LENGTH ;;
        --max-response-length) variable=MAX_TOTAL_RESPONSE_LENGTH ;;
        --ppo-max-tokens) variable=PPO_MAX_TOKEN_LEN_PER_GPU ;;
        --gpu-memory-utilization) variable=GPU_MEMORY_UTILIZATION ;;
        --warmup-epochs) variable=WARMUP_EPOCHS ;;
        --epochs) variable=FORMAL_EPOCHS ;;
        --warmup-steps) variable=WARMUP_STEPS ;;
        --steps) variable=FORMAL_STEPS ;;
        --save-freq) variable=SAVE_FREQ ;;
        --progress-interval) variable=PROGRESS_INTERVAL_SECONDS ;;
        --data) variable=FORMAL_TRAIN_DATA ;;
        --warmup-data) variable=WARMUP_TRAIN_DATA ;;
        --val-data) variable=VAL_DATA ;;
        --model) variable=MODEL_PATH ;;
        --output-dir) variable=CKPT_ROOT ;;
        --resume-dir) variable=RESUME_RUN_DIR ;;
        --phase) variable=PHASE ;;
        *) echo "Unknown option: $option" >&2; usage >&2; exit 2 ;;
    esac
    (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || { echo "Missing value for $option" >&2; exit 2; }
    printf -v "$variable" '%s' "$2"
    shift 2
done
DATA_MEMORY_PERCENT="${DATA_MEMORY_PERCENT:-90}"
DATA_MEMORY_WAIT_SECONDS="${DATA_MEMORY_WAIT_SECONDS:-180}"
DATA_PREFETCH_FACTOR="${DATA_PREFETCH_FACTOR:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-1}"
[[ "$DATA_MEMORY_PERCENT" =~ ^[1-9][0-9]*$ ]] && (( DATA_MEMORY_PERCENT <= 90 )) || { echo "memory-limit-percent must be 1..90" >&2; exit 2; }
[[ "$DATA_MEMORY_WAIT_SECONDS" =~ ^[1-9][0-9]*$ && "$DATA_PREFETCH_FACTOR" =~ ^[1-9][0-9]*$ && "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]] || { echo "Invalid loading workers/prefetch/timeout" >&2; exit 2; }
export AGENT0_DATA_MEMORY_GUARD=1
export AGENT0_DATA_MEMORY_PERCENT="$DATA_MEMORY_PERCENT"
export AGENT0_DATA_MEMORY_WAIT_SECONDS="$DATA_MEMORY_WAIT_SECONDS"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-2,3}}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
NUM_GPUS=${#GPU_IDS[@]}
FORMAL_TRAIN_DATA="${FORMAL_TRAIN_DATA:-${PROJECT_ROOT}/data/rl/rl_200_multisource.parquet}"
WARMUP_TRAIN_DATA="${WARMUP_TRAIN_DATA:-${PROJECT_ROOT}/data/rl/rl_warmup_200_multisource.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/rl/validation_10_rebuilt.parquet}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/sft_2x4090/latest_mixed_merged}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/paper_500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-0}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-9216}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
MAX_TOTAL_RESPONSE_LENGTH="${MAX_TOTAL_RESPONSE_LENGTH:-3072}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${NUM_GPUS}}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-1}"
[[ "$CUDA_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid GPU list" >&2; exit 2; }
declare -A seen_gpus=()
for gpu_id in "${GPU_IDS[@]}"; do
    [[ -z "${seen_gpus[$gpu_id]:-}" ]] || { echo "Duplicate GPU ID: $gpu_id" >&2; exit 2; }
    seen_gpus[$gpu_id]=1
done
for variable in TRAIN_BATCH_SIZE SAVE_FREQ PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU PPO_MAX_TOKEN_LEN_PER_GPU MAX_NUM_SEQS MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_PROMPT_LENGTH MAX_TOTAL_RESPONSE_LENGTH ROLLOUT_N TENSOR_MODEL_PARALLEL_SIZE PROGRESS_INTERVAL_SECONDS WARMUP_EPOCHS FORMAL_EPOCHS; do
    [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || { echo "$variable must be a positive integer" >&2; exit 2; }
done
(( NUM_GPUS % TENSOR_MODEL_PARALLEL_SIZE == 0 )) || { echo "GPU count must be divisible by TP" >&2; exit 2; }
(( TRAIN_BATCH_SIZE * ROLLOUT_N % NUM_GPUS == 0 )) || { echo "batch-size * rollout-n must be divisible by GPU count" >&2; exit 2; }
(( TRAIN_BATCH_SIZE >= PPO_MINI_BATCH_SIZE && PPO_MINI_BATCH_SIZE * ROLLOUT_N % NUM_GPUS == 0 && (PPO_MINI_BATCH_SIZE * ROLLOUT_N / NUM_GPUS) % PPO_MICRO_BATCH_SIZE_PER_GPU == 0 )) || { echo "Invalid actor mini/micro batch for GPU count and rollout-n" >&2; exit 2; }
(( MAX_PROMPT_LENGTH + MAX_TOTAL_RESPONSE_LENGTH <= MAX_MODEL_LEN )) || { echo "Prompt + response length exceeds model context" >&2; exit 2; }
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0[.][0-9]+$ || "$GPU_MEMORY_UTILIZATION" == 1 || "$GPU_MEMORY_UTILIZATION" == 1.0 ]] || { echo "gpu-memory-utilization must be in (0,1]" >&2; exit 2; }
[[ "$GPU_MEMORY_UTILIZATION" != 0.0 && "$GPU_MEMORY_UTILIZATION" != 0.00 ]] || { echo "gpu-memory-utilization must be positive" >&2; exit 2; }
rows() {
    if [[ -f "$1" ]]; then
        "${PROJECT_ROOT}/.venv/bin/python" -c 'import sys,pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "$1"
    elif [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo 200
    else
        echo "Missing dataset: $1" >&2; return 1
    fi
}
WARMUP_ROWS=$(rows "$WARMUP_TRAIN_DATA")
FORMAL_ROWS=$(rows "$FORMAL_TRAIN_DATA")
WARMUP_STEPS="${WARMUP_STEPS:-$((WARMUP_ROWS / TRAIN_BATCH_SIZE * WARMUP_EPOCHS))}"
FORMAL_STEPS="${FORMAL_STEPS:-$((FORMAL_ROWS / TRAIN_BATCH_SIZE * FORMAL_EPOCHS))}"
[[ "$WARMUP_STEPS" =~ ^[1-9][0-9]*$ && "$FORMAL_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid step counts" >&2; exit 2; }
(( WARMUP_STEPS <= WARMUP_ROWS / TRAIN_BATCH_SIZE * WARMUP_EPOCHS && FORMAL_STEPS <= FORMAL_ROWS / TRAIN_BATCH_SIZE * FORMAL_EPOCHS )) || { echo "Step cap exceeds available epochs; increase --warmup-epochs/--epochs" >&2; exit 2; }
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'loading: workers=%s prefetch=%s memory_threshold=%s%% wait_timeout=%ss\n' "$DATALOADER_WORKERS" "$DATA_PREFETCH_FACTOR" "$DATA_MEMORY_PERCENT" "$DATA_MEMORY_WAIT_SECONDS"
    printf 'GPUs=%s num_gpus=%s TP=%s global_batch=%s rollout_n=%s trajectories=%s concurrency=%s mini_batch=%s micro_batch=%s warmup_steps=%s formal_steps=%s max_model_len=%s model=%s\n' "$CUDA_DEVICES" "$NUM_GPUS" "$TENSOR_MODEL_PARALLEL_SIZE" "$TRAIN_BATCH_SIZE" "$ROLLOUT_N" "$((TRAIN_BATCH_SIZE * ROLLOUT_N))" "$MAX_NUM_SEQS" "$PPO_MINI_BATCH_SIZE" "$PPO_MICRO_BATCH_SIZE_PER_GPU" "$WARMUP_STEPS" "$FORMAL_STEPS" "$MAX_MODEL_LEN" "$MODEL_PATH"
    exit 0
fi
MODEL_PATH="$(realpath -e -- "$MODEL_PATH")"
PHASE="${PHASE:-full}"
SKIP_WARMUP="${SKIP_WARMUP:-0}"
case "${PHASE}" in
    external_warmup)
        RUN_MODE="warmup"
        ;;
    serc)
        RUN_MODE="formal_only"
        ;;
    full)
        if [[ "${SKIP_WARMUP}" == "1" || "${SKIP_WARMUP,,}" == "true" ]]; then
            RUN_MODE="formal_only"
        else
            RUN_MODE="full"
        fi
        ;;
    *)
        echo "[rl-agent0] PHASE must be full, external_warmup, or serc" >&2
        exit 2
        ;;
esac

if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
    [[ "${RUN_MODE}" != "formal_only" ]] || {
        echo "[rl-agent0] RESUME_RUN_DIR requires the warm-up phase" >&2
        exit 2
    }
    readonly RUN_DIR="$(realpath -- "${RESUME_RUN_DIR}")"
    [[ -f "${RUN_DIR}/latest_checkpointed_iteration.txt" && -f "${RUN_DIR}/README.md" ]] || {
        echo "[rl-agent0] resume directory lacks its checkpoint tracker or README: ${RUN_DIR}" >&2
        exit 2
    }
    readonly RUN_NAME="$(basename -- "${RUN_DIR}")"
    readonly WARMUP_RESUME_MODE="auto"
else
    readonly RUN_NAME="rl_${FORMAL_ROWS}_${NUM_GPUS}gpu_qlora_nf4_ctx${MAX_MODEL_LEN}_resp${MAX_TOTAL_RESPONSE_LENGTH}_n${ROLLOUT_N}_maxseq${MAX_NUM_SEQS}_gpuutil${GPU_MEMORY_UTILIZATION}_$(date +%Y%m%d_%H%M%S)"
    readonly RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"
    readonly WARMUP_RESUME_MODE="disable"
fi
readonly LAUNCH_LOG="${PROJECT_ROOT}/logs/${RUN_NAME}.log"
readonly RUN_README="${RUN_DIR}/README.md"

if [[ "${WARMUP_RESUME_MODE}" == "auto" ]]; then
    recorded_model="$(sed -n 's/^- Resolved base model: //p' "${RUN_README}" | head -n 1)"
    if [[ -z "${recorded_model}" ]]; then
        recorded_model="$(sed -n 's/^- Base model: //p' "${RUN_README}" | head -n 1)"
    fi
    if [[ -z "${recorded_model}" || "$(realpath -e -- "${recorded_model}")" != "${MODEL_PATH}" ]]; then
        echo "[rl-agent0] resume model differs from the run's SFT model: ${recorded_model:-missing}" >&2
        echo "[rl-agent0] selected model: ${MODEL_PATH}" >&2
        exit 2
    fi
fi

export PYTHONPATH="${PROJECT_ROOT}/tools/runtime_guard:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export N_GPUS="${NUM_GPUS}"
export VAL_DATA
export MODEL_PATH
export CKPT_PATH="${CKPT_ROOT}"
export PYTHONUNBUFFERED=1
export AGENT0_ROLLOUT_VERBOSE=1
export AGENT0_ROLLOUT_LOG_SAMPLES="${AGENT0_ROLLOUT_LOG_SAMPLES:-16}"
export AGENT0_ROLLOUT_LOG_MAX_CHARS="${AGENT0_ROLLOUT_LOG_MAX_CHARS:-600}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Avoid the FlashInfer sampler CUDA-runtime mismatch in the local vLLM wheel.
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_DIR}"
if [[ "${WARMUP_RESUME_MODE}" == "auto" ]]; then
    sed -i 's/^- Status: .*/- Status: running/' "${RUN_README}"
    printf '%s\n' \
        "- Resume attempt: $(date '+%F %T %z') from step $(cat "${RUN_DIR}/latest_checkpointed_iteration.txt"); vLLM sleep level 2" \
        >>"${RUN_README}"
else
    printf '%s\n' \
    "# ${NUM_GPUS}x RTX 4090 RL run: ${RUN_NAME}" \
    "" \
    "- Status: running" \
    "- Created: $(date '+%F %T %z')" \
    "- Base model: ${MODEL_PATH}" \
    "- Resolved base model: ${MODEL_PATH}" \
    "- Warm-up dataset: ${WARMUP_TRAIN_DATA}" \
    "- Formal RL dataset: ${FORMAL_TRAIN_DATA}" \
    "- Validation dataset: ${VAL_DATA}" \
    "- Warm-up: external correctness reward, ${WARMUP_EPOCHS} epochs, ${WARMUP_STEPS} steps" \
    "- Formal phase: SERC/GRPO, ${FORMAL_EPOCHS} epochs, ${FORMAL_STEPS} additional steps" \
    "- GPUs: ${CUDA_DEVICES} (${NUM_GPUS} x RTX 4090, TP=${TENSOR_MODEL_PARALLEL_SIZE})" \
    "- QLoRA: NF4, BF16 compute/storage, double quantization, LoRA rank 8/alpha 32" \
    "- Memory: actor activation CPU offload, response-only logits, FSDP backward_post, actor/optimizer phase offload, vLLM sleep level 2" \
    "- Data loading: workers=$DATALOADER_WORKERS, prefetch=$DATA_PREFETCH_FACTOR, memory pause threshold=$DATA_MEMORY_PERCENT%, wait timeout=$DATA_MEMORY_WAIT_SECONDS seconds" \
    "- Batches: global=$TRAIN_BATCH_SIZE, actor mini=$PPO_MINI_BATCH_SIZE, per-GPU micro=$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    "- Rollout: n=${ROLLOUT_N}, max_num_seqs=${MAX_NUM_SEQS}, max_prompt_length=${MAX_PROMPT_LENGTH}, max_total_response_length=${MAX_TOTAL_RESPONSE_LENGTH}, max_model_len=${MAX_MODEL_LEN}" \
    "- Command: bash scripts/rl-agent0-2x4090-optimized.sh" \
    "- Log: ${LAUNCH_LOG}" \
        >"${RUN_README}"
fi

run_status="failed"
record_status() {
    local exit_code=$?
    sed -i "s/^- Status: running$/- Status: ${run_status}/" "${RUN_README}" 2>/dev/null || true
    printf '%s\n' "- Last update: $(date '+%F %T %z')" "- Exit code: ${exit_code}" >>"${RUN_README}" 2>/dev/null || true
}
trap record_status EXIT

training_pid=""
progress_pid=""

on_error() {
    local status=$?
    echo "[rl-agent0] failed (exit=${status}); stopping without retry." >&2
    exit "${status}"
}

stop_process_group() {
    local pid="${1:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
}

on_interrupt() {
    echo "[rl-agent0] interrupted; stopping active phase and progress monitor." >&2
    stop_process_group "${training_pid}"
    if [[ -n "${progress_pid}" ]] && kill -0 "${progress_pid}" 2>/dev/null; then
        kill "${progress_pid}" 2>/dev/null || true
    fi
    exit 130
}

trap on_error ERR
trap on_interrupt INT TERM

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "[rl-agent0] missing file: ${path}" >&2
        return 1
    fi
}

require_executable() {
    local path="$1"
    if [[ ! -x "${path}" ]]; then
        echo "[rl-agent0] missing executable: ${path}" >&2
        return 1
    fi
}

preflight() {
    if (( MAX_PROMPT_LENGTH + MAX_TOTAL_RESPONSE_LENGTH > MAX_MODEL_LEN )); then
        echo "[rl-agent0] prompt and response budgets exceed max_model_len" >&2
        return 1
    fi
    require_executable "${PROJECT_ROOT}/.venv/bin/python"
    require_executable "${PROJECT_ROOT}/.venv/bin/python3"
    require_file "${FORMAL_TRAIN_DATA}"
    require_file "${WARMUP_TRAIN_DATA}"
    require_file "${VAL_DATA}"
    require_file "${PROJECT_ROOT}/verl/trainer/config/agent0_trainer_2x4090.yaml"
    require_file "${PROJECT_ROOT}/verl/trainer/config/agent0_trainer_2x4090_external_warmup.yaml"
    require_file "${PROJECT_ROOT}/scripts/rl-agent0.sh"
    require_file "${MODEL_PATH}/config.json"

    if ! command -v setsid >/dev/null 2>&1; then
        echo "[rl-agent0] setsid is unavailable; refusing to start." >&2
        return 1
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[rl-agent0] nvidia-smi is unavailable; refusing to start." >&2
        return 1
    fi

    local gpu used
    for gpu in 2 3; do
        used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
        if [[ ! "${used}" =~ ^[0-9]+$ ]]; then
            echo "[rl-agent0] cannot read memory usage for GPU ${gpu}: ${used}" >&2
            return 1
        fi
        if (( used > 512 )); then
            echo "[rl-agent0] GPU ${gpu} already uses ${used} MiB; refusing to start." >&2
            return 1
        fi
    done
}

progress_heartbeat() {
    local active_pid="$1"
    local started_at="$2"
    local now elapsed session_dir progress_line latest_rollout_line latest_stage_line process_snapshot
    local stage gpu_stats worker_count worker_rss ram_stats
    set +e
    trap - ERR

    while kill -0 "${active_pid}" 2>/dev/null; do
        now="$(date +%s)"
        elapsed=$((now - started_at))
        session_dir="$(find /tmp/ray -mindepth 1 -maxdepth 1 -type d -name 'session_*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
        progress_line=""
        latest_rollout_line=""
        if [[ -n "${session_dir}" && -d "${session_dir}/logs" ]]; then
            progress_line="$(rg --no-heading --no-filename 'Training Progress:' "${session_dir}"/logs/worker-*.err 2>/dev/null | tail -n 1 | tr -d '\r')"
            latest_rollout_line="$(rg --no-heading --no-filename '\[Agent0-VL rollout|Training Progress:' "${session_dir}"/logs/worker-*.err 2>/dev/null | tail -n 1 | tr -d '\r')"
        fi
        latest_stage_line="$(rg --no-heading '\[Agent0-VL trainer\] (rollout|old_logprob|ref_logprob|actor_update)_(start|done)' "${LAUNCH_LOG}" 2>/dev/null | tail -n 1 | tr -d '\r')"
        process_snapshot="$(ps -eo args= 2>/dev/null)"
        if [[ "${latest_stage_line}" == *"actor_update_start"* ]]; then
            stage="actor-update/backward"
        elif [[ "${latest_stage_line}" == *"ref_logprob_start"* ]]; then
            stage="reference-log-prob"
        elif [[ "${latest_stage_line}" == *"old_logprob_start"* ]]; then
            stage="old-log-prob"
        elif [[ "${latest_rollout_line}" == *"verifier"* ]]; then
            stage="rollout/verifier"
        elif [[ "${latest_rollout_line}" == *"tool_"* ]]; then
            stage="rollout/tool"
        elif [[ "${latest_rollout_line}" == *"repair"* ]]; then
            stage="rollout/repair"
        elif [[ "${process_snapshot}" == *"actor_rollout_generate_sequences"* ]]; then
            stage="rollout/generation"
        elif [[ "${process_snapshot}" == *"compute_log_prob"* || "${process_snapshot}" == *"update_actor"* || "${process_snapshot}" == *"loss.backward"* ]]; then
            stage="actor-update/log-prob"
        elif [[ -n "${progress_line}" ]]; then
            stage="trainer"
        else
            stage="initialization/waiting"
        fi
        worker_count="$(printf '%s\n' "${process_snapshot}" | awk '/ray::WorkerDict.actor_rollout_generate_sequences/ {count++} END {print count + 0}')"
        worker_rss="$(ps -eo rss,args= 2>/dev/null | awk '/ray::WorkerDict.actor_rollout_generate_sequences/ {sum += $1; count++} END {if (count) printf "%d workers, %.1f GiB RSS", count, sum / 1048576; else print "unavailable"}')"
        ram_stats="$(free -h 2>/dev/null | awk '/^Mem:/ {printf "used=%s available=%s", $3, $7}')"
        gpu_stats="$(nvidia-smi -i "${CUDA_DEVICES}" --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd ';' -)"
        printf '[rl-agent0][heartbeat] elapsed=%ss stage=%s workers=%s gpu(index,usedMiB,freeMiB,util)=%s ram=%s rss=%s\n' \
            "${elapsed}" "${stage}" "${worker_count}" "${gpu_stats:-unavailable}" \
            "${ram_stats:-unavailable}" "${worker_rss}"
        [[ -n "${progress_line}" ]] && printf '[rl-agent0][progress] %s\n' "${progress_line:0:240}"
        [[ -n "${latest_rollout_line}" && "${latest_rollout_line}" != "${progress_line}" ]] && \
            printf '[rl-agent0][rollout] %s\n' "${latest_rollout_line:0:300}"
        sleep "${PROGRESS_INTERVAL_SECONDS}"
    done
}

run_phase() {
    local phase_name="$1" train_data="$2" config_name="$3" reward_manager="$4"
    local enable_verification="$5" enable_self_repair="$6" total_epochs="$7"
    local total_steps="$8" resume_mode="$9"

    export TRAIN_DATA="${train_data}"
    export CONFIG_NAME="${config_name}"
    export REWARD_MANAGER="${reward_manager}"
    export ENABLE_VERIFICATION="${enable_verification}"
    export ENABLE_SELF_REPAIR="${enable_self_repair}"
    export TOTAL_EPOCHS="${total_epochs}"

    echo "[rl-agent0] starting phase=${phase_name}"
    echo "[rl-agent0] data=${train_data} reward_manager=${reward_manager} epochs=${total_epochs} target_steps=${total_steps}"
    echo "[rl-agent0] checkpoint_dir=${RUN_DIR} resume_mode=${resume_mode}"

    setsid bash scripts/rl-agent0.sh \
        "data.train_files=${train_data}" \
        "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
        "+data.train_num_workers=${DATALOADER_WORKERS}" \
        "+data.train_prefetch_factor=${DATA_PREFETCH_FACTOR}" \
        "data.val_num_workers=${DATALOADER_WORKERS}" \
        "data.val_prefetch_factor=${DATA_PREFETCH_FACTOR}" \
        "data.filter_overlong_prompts_workers=1" \
        "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
        "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}" \
        "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
        "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}" \
        "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
        "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}" \
        "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
        "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}" \
        "actor_rollout_ref.rollout.max_total_response_length=${MAX_TOTAL_RESPONSE_LENGTH}" \
        "trainer.n_gpus_per_node=${NUM_GPUS}" \
        "trainer.logger=[console]" \
        "trainer.project_name=Agent0-VL" \
        "trainer.experiment_name=${RUN_NAME}_${phase_name}" \
        "trainer.default_local_dir=${RUN_DIR}" \
        "trainer.total_training_steps=${total_steps}" \
        "trainer.total_epochs=${total_epochs}" \
        "trainer.save_freq=${SAVE_FREQ}" \
        "trainer.test_freq=${TEST_FREQ}" \
        trainer.val_before_train=False \
        "trainer.resume_mode=${resume_mode}" \
        trainer.log_val_generations=0 &
    training_pid=$!

    progress_heartbeat "${training_pid}" "$(date +%s)" &
    progress_pid=$!
    local status=0
    if wait "${training_pid}"; then
        status=0
    else
        status=$?
    fi
    kill "${progress_pid}" 2>/dev/null || true
    wait "${progress_pid}" 2>/dev/null || true
    training_pid=""
    progress_pid=""
    if (( status != 0 )); then
        echo "[rl-agent0] phase=${phase_name} failed (exit=${status})" >&2
        return "${status}"
    fi
    echo "[rl-agent0] phase=${phase_name} finished successfully"
}

exec > >(tee -a "${LAUNCH_LOG}") 2>&1
preflight
echo "[rl-agent0] run=${RUN_NAME} mode=${RUN_MODE}"
echo "[rl-agent0] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, TP=${TENSOR_MODEL_PARALLEL_SIZE}, n=${ROLLOUT_N}"
echo "[rl-agent0] max_model_len=${MAX_MODEL_LEN}, max_num_seqs=${MAX_NUM_SEQS}, max_prompt_length=${MAX_PROMPT_LENGTH}, max_total_response_length=${MAX_TOTAL_RESPONSE_LENGTH}"
echo "[rl-agent0] MODEL_PATH=${MODEL_PATH}"
echo "[rl-agent0] launcher_log=${LAUNCH_LOG}"

if [[ "${RUN_MODE}" == "warmup" || "${RUN_MODE}" == "full" ]]; then
    run_phase "warmup" "${WARMUP_TRAIN_DATA}" \
        "agent0_trainer_2x4090_external_warmup" external False False "${WARMUP_EPOCHS}" "${WARMUP_STEPS}" "${WARMUP_RESUME_MODE}"
    [[ -f "${RUN_DIR}/latest_checkpointed_iteration.txt" ]] || {
        echo "[rl-agent0] warm-up did not write checkpoint tracker: ${RUN_DIR}" >&2
        exit 1
    }
    warmup_step="$(tr -d '[:space:]' <"${RUN_DIR}/latest_checkpointed_iteration.txt")"
    [[ "${warmup_step}" == "${WARMUP_STEPS}" ]] || {
        echo "[rl-agent0] expected warm-up checkpoint step ${WARMUP_STEPS}, got ${warmup_step}" >&2
        exit 1
    }
    [[ -d "${RUN_DIR}/global_step_${WARMUP_STEPS}/actor" ]] || {
        echo "[rl-agent0] warm-up actor checkpoint is missing" >&2
        exit 1
    }
    echo "[rl-agent0] warm-up checkpoint saved at ${RUN_DIR}/global_step_${WARMUP_STEPS}/actor"
    if [[ "${RUN_MODE}" == "warmup" ]]; then
        run_status="success"
        echo "[rl-agent0] external warm-up complete; checkpoints=${RUN_DIR}"
        exit 0
    fi
fi

if [[ "${RUN_MODE}" == "formal_only" ]]; then
    formal_resume="disable"
    formal_steps="${FORMAL_STEPS}"
else
    formal_resume="auto"
    formal_steps="$((WARMUP_STEPS + FORMAL_STEPS))"
fi

run_phase "serc" "${FORMAL_TRAIN_DATA}" agent0_trainer_2x4090 agent0 True True "${FORMAL_EPOCHS}" "${formal_steps}" "${formal_resume}"
[[ -d "${RUN_DIR}/global_step_${formal_steps}/actor" ]] || {
    echo "[rl-agent0] formal actor checkpoint is missing: ${RUN_DIR}/global_step_${formal_steps}/actor" >&2
    exit 1
}

run_status="success"
echo "[rl-agent0] complete; checkpoints=${RUN_DIR}"
