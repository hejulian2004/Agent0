#!/usr/bin/env bash
set -Eeuo pipefail

# Run the seven benchmark splits used by the Agent0-VL main results table.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
GPU_IDS="${GPU_IDS:-0,1}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/base/Qwen2.5-VL-7B-Instruct}"
TEST_DATA="${TEST_DATA:-${PROJECT_ROOT}/data/evaluation/agent0_vl_paper_main/paper_main_7.parquet}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/rl/rl_200_multisource.parquet}"
VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-11264}"
MAX_TOTAL_RESPONSE_LENGTH="${MAX_TOTAL_RESPONSE_LENGTH:-3072}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$((MAX_MODEL_LEN - MAX_TOTAL_RESPONSE_LENGTH))}"
# The full vision benchmark is host-RAM heavy; expose conservative defaults
# while keeping each limit overridable for a machine with more headroom.
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
VAL_PREFETCH_FACTOR="${VAL_PREFETCH_FACTOR:-1}"

if [[ "${TEST_DATA}" == "${PROJECT_ROOT}/data/rl/validation_10_rebuilt.parquet" \
    && "${ALLOW_VAL10_SMOKE:-0}" != "1" ]]; then
    cat >&2 <<'EOF'
The 10-row validation set is only for an internal smoke run, not the full
Agent0-VL paper benchmark suite. Refusing to present it as the requested test.
Set ALLOW_VAL10_SMOKE=1 only when an internal smoke run is intended.
EOF
    exit 2
fi

MODEL_LABEL="${MODEL_LABEL:-qwen2_5_vl_7b}"
MODEL_LABEL="${MODEL_LABEL// /_}"
RUN_NAME="${RUN_NAME:-serc_eval_${MODEL_LABEL}_paper_main_7_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/evaluation_results/${RUN_NAME}}"
RUN_README="${RESULT_DIR}/README.md"
RUN_LOG="${RESULT_DIR}/run.log"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing project interpreter: ${PYTHON}" >&2
    exit 1
fi
if [[ "${TEST_DATA}" == "${PROJECT_ROOT}/data/evaluation/agent0_vl_paper_main/paper_main_7.parquet" \
    && ! -f "${TEST_DATA}" ]]; then
    echo "Missing full paper evaluation Parquet. Prepare it with:" >&2
    echo "  ${PYTHON} tools/prepare_paper_eval_datasets.py" >&2
    exit 1
fi
for required_file in "${MODEL_PATH}/config.json" "${TEST_DATA}" "${TRAIN_DATA}" \
    "${PROJECT_ROOT}/verl/trainer/config/agent0_trainer_2x4090.yaml" \
    "${PROJECT_ROOT}/scripts/prompt.txt"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done
TEST_ROWS="$("${PYTHON}" -c 'import pyarrow.parquet as pq,sys; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "${TEST_DATA}")"
if [[ "${TEST_DATA}" == "${PROJECT_ROOT}/data/evaluation/agent0_vl_paper_main/paper_main_7.parquet" ]]; then
    if [[ "${TEST_ROWS}" != "14249" ]]; then
        echo "Expected 14,249 rows in the seven-benchmark paper evaluation set; found ${TEST_ROWS}." >&2
        exit 1
    fi
fi
if [[ -e "${RESULT_DIR}" ]]; then
    echo "Result directory already exists: ${RESULT_DIR}" >&2
    echo "Set a new RUN_NAME or RESULT_DIR to preserve the previous run." >&2
    exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if [[ "${#GPU_LIST[@]}" -ne 2 ]]; then
    echo "GPU_IDS must name exactly two physical GPUs (default: 0,1)." >&2
    exit 2
fi
if [[ ! "${VAL_N}" =~ ^[1-9][0-9]*$ || ( "${VAL_DO_SAMPLE}" != "True" && "${VAL_DO_SAMPLE}" != "False" ) ]]; then
    echo "VAL_N must be a positive integer and VAL_DO_SAMPLE must be True or False." >&2
    exit 2
fi
if [[ "${VAL_DO_SAMPLE}" == "False" && "${VAL_N}" != "1" ]]; then
    echo "Greedy evaluation requires VAL_N=1. Set VAL_DO_SAMPLE=True for multiple samples." >&2
    exit 2
fi
if (( MAX_PROMPT_LENGTH <= 0 || MAX_PROMPT_LENGTH + MAX_TOTAL_RESPONSE_LENGTH > MAX_MODEL_LEN )); then
    echo "Require 0 < MAX_PROMPT_LENGTH and MAX_PROMPT_LENGTH + MAX_TOTAL_RESPONSE_LENGTH <= MAX_MODEL_LEN." >&2
    exit 2
fi
if (( VAL_BATCH_SIZE <= 0 || MAX_NUM_SEQS <= 0 )); then
    echo "VAL_BATCH_SIZE and MAX_NUM_SEQS must be positive integers." >&2
    exit 2
fi
if [[ ! "${VAL_NUM_WORKERS}" =~ ^[0-9]+$ || ! "${VAL_PREFETCH_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
    echo "VAL_NUM_WORKERS must be non-negative and VAL_PREFETCH_FACTOR must be positive." >&2
    exit 2
fi
for gpu in "${GPU_LIST[@]}"; do
    used_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ ! "${used_mib}" =~ ^[0-9]+$ ]]; then
        echo "Could not read memory usage for GPU ${gpu}: ${used_mib}" >&2
        exit 1
    fi
    if (( used_mib > 512 )); then
        echo "GPU ${gpu} already uses ${used_mib} MiB; refusing to start." >&2
        exit 1
    fi
done

NUM_GPUS="${#GPU_LIST[@]}"
mkdir -p "${RESULT_DIR}"
run_status="failed"
record_status() {
    local exit_code=$?
    local updated_at
    updated_at="$(date '+%F %T %z')"
    if (( exit_code == 0 )); then
        run_status="success"
    fi
    if [[ -f "${RUN_README}" ]]; then
        sed -i "s/^- Status: running$/- Status: ${run_status}/" "${RUN_README}"
        printf '%s\n' "- Updated: ${updated_at}" "- Exit code: ${exit_code}" >> "${RUN_README}"
    fi
    return "${exit_code}"
}
trap record_status EXIT

created_at="$(date '+%F %T %z')"
cat > "${RUN_README}" <<EOF
# SERC evaluation: ${RUN_NAME}

- Status: running
- Created: ${created_at}
- Base model/checkpoint: ${MODEL_PATH}
- Run directory: ${RESULT_DIR}
- Dataset: ${TEST_DATA} (${TEST_ROWS} rows; val samples per prompt=${VAL_N})
- GPUs: physical ${GPU_IDS}, ${NUM_GPUS} x RTX 4090, tensor parallel size 2
- Precision/quantization: vLLM BF16 rollout; actor QLoRA NF4 with BF16 compute/storage and double quantization
- LoRA: rank 8, alpha 32; no optimizer update (trainer.val_only=True)
- Evaluation: Agent0-VL SERC verifier/reasoning/tool/repair path; val samples per prompt=${VAL_N}
- Decoding: do_sample=${VAL_DO_SAMPLE}; greedy when False
- Limits: max prompt length=${MAX_PROMPT_LENGTH}, max model length=${MAX_MODEL_LEN}, max total response length=${MAX_TOTAL_RESPONSE_LENGTH}, validation batch size=${VAL_BATCH_SIZE}, vLLM max_num_seqs=${MAX_NUM_SEQS}, validation DataLoader workers=${VAL_NUM_WORKERS}, prefetch factor=${VAL_PREFETCH_FACTOR}, GPU memory utilization=${GPU_MEMORY_UTILIZATION}
- Hydra run directory: ${RESULT_DIR}/hydra
- Validation results: ${RESULT_DIR}/validation_<timestamp>.json
- Aggregate metrics: ${RESULT_DIR}/summary.json
- Repair success definition: applied repair whose post-repair Verifier confidence is at least the configured threshold (0.7).
- Repair score improvement: post-repair Verifier score exceeds the pre-repair score.
- Log: ${RUN_LOG}
- Command/config: verl.trainer.main_ppo --config-name=agent0_trainer_2x4090 with overrides in scripts/eval_serc_2gpu.sh
EOF

export PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export N_GPUS="${NUM_GPUS}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a "${RUN_LOG}") 2>&1
echo "[$(date '+%F %T %z')] Starting evaluation-only SERC run ${RUN_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Paper benchmark set: ${TEST_DATA}"
echo "Physical GPUs: ${GPU_IDS}"
echo "Results: ${RESULT_DIR}"

"${PYTHON}" -m verl.trainer.main_ppo \
    --config-name=agent0_trainer_2x4090 \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${TEST_DATA}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.val_num_workers="${VAL_NUM_WORKERS}" \
    data.val_prefetch_factor="${VAL_PREFETCH_FACTOR}" \
    data.train_batch_size=2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_qlora=True \
    actor_rollout_ref.actor.lora_rank=8 \
    actor_rollout_ref.actor.lora_alpha=32 \
    actor_rollout_ref.actor.qlora_4bit_quant_type=nf4 \
    actor_rollout_ref.actor.qlora_4bit_compute_dtype=bf16 \
    actor_rollout_ref.actor.qlora_4bit_quant_storage=bf16 \
    actor_rollout_ref.actor.qlora_4bit_use_double_quant=True \
    actor_rollout_ref.rollout.load_format=dummy_hf \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_total_response_length="${MAX_TOTAL_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.project_name=Agent0-VL-evaluation \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.default_local_dir="${RESULT_DIR}/checkpoints" \
    trainer.logger=[console] \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.log_val_generations=0 \
    trainer.save_validation_results=True \
    trainer.validation_results_path="${RESULT_DIR}/validation.json" \
    hydra.run.dir="${RESULT_DIR}/hydra"

shopt -s nullglob
validation_files=("${RESULT_DIR}"/validation_*.json)
shopt -u nullglob
if [[ "${#validation_files[@]}" -ne 1 ]]; then
    echo "Expected one timestamped validation JSON in ${RESULT_DIR}; found ${#validation_files[@]}." >&2
    exit 1
fi
RESULT_JSON="${validation_files[0]}"
"${PYTHON}" scripts/summarize_serc_eval.py \
    --results-json "${RESULT_JSON}" \
    --output "${RESULT_DIR}/summary.json" \
    --model-path "${MODEL_PATH}" \
    --dataset "${TEST_DATA}" \
    --gpu-ids "${GPU_IDS}" \
    --val-n "${VAL_N}"

run_status="success"
echo "[$(date '+%F %T %z')] Evaluation completed. Results: ${RESULT_DIR}"
