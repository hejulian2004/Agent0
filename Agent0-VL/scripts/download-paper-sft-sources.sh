#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCES=(smr mm_rlhf llava_ov_image)

usage() {
  cat <<'EOF'
Usage: bash scripts/download-paper-sft-sources.sh [smr|mm_rlhf|llava_ov_image ...]

Downloads the selected paper SFT sources in the foreground with progress
bars in this terminal. With no arguments, downloads all three sequentially.
Downloads resume from Hugging Face's local cache. Each source also has a log.
EOF
}

valid_source() {
  case "$1" in
    smr|mm_rlhf|llava_ov_image) return 0 ;;
    *) return 1 ;;
  esac
}

source_dir() {
  case "$1" in
    smr) printf '%s' 'data/raw/.staging/smr' ;;
    mm_rlhf) printf '%s' 'data/raw/.staging/mm-rlhf' ;;
    llava_ov_image) printf '%s' 'data/raw/.staging/llava-ov-image/train' ;;
  esac
}

update_inventory() {
  local log="$1" source="$2" status="$3" pid="$4"
  "$ROOT/.venv/bin/python" - "$log" "$source" "$status" "$pid" <<'PY'
import datetime
import fcntl
import pathlib
import sys

log, source, status, pid = sys.argv[1:]
inventory = pathlib.Path("logs/README.md")
inventory.parent.mkdir(parents=True, exist_ok=True)
entry = (
    f"- Path: {log}\n"
    f"  Created: {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}\n"
    f"  Status: {status}\n"
    f"  Source: {source} official Hugging Face training data\n"
    "  Base model: none; data download\n"
    "  GPU/precision/quantization/LoRA/context/batch: none\n"
    f"  Command: bash scripts/download-paper-sft-sources.sh {source}\n"
    f"  PID: {pid}\n"
)
with inventory.open("r+" if inventory.exists() else "w+", encoding="utf-8") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.seek(0)
    text = handle.read() or "# Download logs\n"
    start = text.find(f"- Path: {log}\n")
    if start >= 0:
        end = text.find("\n- Path: ", start + 1)
        text = text[:start] + entry.rstrip() + "\n" + (text[end:] if end >= 0 else "")
    else:
        text = text.rstrip() + "\n\n" + entry
    handle.seek(0)
    handle.write(text)
    handle.truncate()
PY
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing virtualenv Python: $ROOT/.venv/bin/python" >&2
  exit 1
fi
if ! command -v script >/dev/null 2>&1; then
  echo "Missing 'script' utility (util-linux), needed to show download progress and save a log" >&2
  exit 1
fi

if (( $# > 0 )); then
  SOURCES=("$@")
fi

mkdir -p logs
for source_name in "${SOURCES[@]}"; do
  if ! valid_source "$source_name"; then
    echo "Unknown source: $source_name" >&2
    usage >&2
    exit 2
  fi
  destination="$(source_dir "$source_name")"
  if [[ -f "$destination/AGENT0_DOWNLOAD.json" ]]; then
    echo "$source_name: already downloaded ($destination)"
    continue
  fi
  if pgrep -af "tools.download_paper_sft_sources $source_name" | grep -q '[p]ython'; then
    echo "$source_name: another download is running; stop it before starting this foreground run" >&2
    exit 1
  fi
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_path="logs/paper_sft_download_${source_name}_${stamp}.log"
  : > "$log_path"
  update_inventory "$log_path" "$source_name" running "$$"
  echo "$source_name: downloading in foreground; log: $log_path"
  unset HF_HUB_DISABLE_PROGRESS_BARS
  if script -q -e -f -c "$ROOT/.venv/bin/python -u -m tools.download_paper_sft_sources $source_name" "$log_path"; then
    update_inventory "$log_path" "$source_name" success "$$"
  else
    result=$?
    update_inventory "$log_path" "$source_name" "stopped or failed (exit=$result)" "$$"
    exit "$result"
  fi
done
