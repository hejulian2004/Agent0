#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
ACTION="${1:-preflight}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing project Python: $PYTHON" >&2
    exit 1
fi

preflight() {
    "$PYTHON" -m tools.sft_builder.rebuild_paper_1000 preflight
    "$PYTHON" -m tools.build_multisource_rl --preflight-only
}

prepare() {
    "$PYTHON" -m tools.sft_builder.rebuild_paper_1000 prepare
}

sft() {
    if [[ -z "${OPENAI_API_KEY:-}" && -n "${TEACHER_API_KEY_FILE:-}" ]]; then
        [[ -r "$TEACHER_API_KEY_FILE" ]] || { echo "Teacher API key file unreadable" >&2; exit 1; }
        export OPENAI_API_KEY="$(tr -d '\r\n' < "$TEACHER_API_KEY_FILE")"
    fi
    "$PYTHON" -m tools.sft_builder.rebuild_paper_1000 build \
        --teacher-url "${TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}" \
        --teacher-model "${TEACHER_MODEL:-qwen3.8-27b}" \
        --concurrency "${TEACHER_CONCURRENCY:-4}"
}

rl() {
    "$PYTHON" -m tools.build_multisource_rl
}

case "$ACTION" in
    preflight) preflight ;;
    prepare) prepare ;;
    sft) sft ;;
    rl) rl ;;
    all) preflight; prepare; sft; rl ;;
    *) echo "Usage: $0 {preflight|prepare|sft|rl|all}" >&2; exit 2 ;;
esac
