#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="$PWD/.venv/bin/python"
ACTION="${1:-prepare}"
if (($#)); then shift; fi
case "$ACTION" in
  reset-generation) "$PYTHON" -m tools.sft_builder.balanced_1000 reset-generation "$@" ;;
  restore-completed) "$PYTHON" -m tools.sft_builder.balanced_1000 restore-completed "$@" ;;
  preflight) "$PYTHON" -m tools.sft_builder.balanced_1000 preflight ;;
  prepare) "$PYTHON" -m tools.sft_builder.balanced_1000 prepare "$@" ;;
  sft)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      KEY_FILE="${TEACHER_API_KEY_FILE:-/mnt/d/Agent0/bench/.api_key}"
      [[ -r "$KEY_FILE" ]] || { echo "Set OPENAI_API_KEY or TEACHER_API_KEY_FILE" >&2; exit 1; }
      export OPENAI_API_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
    fi
    "$PYTHON" -m tools.sft_builder.balanced_1000 build \
      --concurrency "${TEACHER_CONCURRENCY:-32}" \
      --max-reasoning-steps "${TEACHER_MAX_STEPS:-16}" \
      --teacher-max-tokens "${TEACHER_MAX_TOKENS:-8192}" \
      --teacher-timeout "${TEACHER_TIMEOUT:-600}" \
      --teacher-url "${TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}" \
      --teacher-model "${TEACHER_MODEL:-qwen3.8-27b}"
    ;;
  rl) "$PYTHON" -m tools.build_multisource_rl ;;
  *) echo "Usage: $0 {prepare|preflight|reset-generation|restore-completed|sft|rl} [--sources SOURCE ...]" >&2; exit 2 ;;
esac
