#!/usr/bin/env bash
set -Eeuo pipefail

# Select the hardware profile while reusing the common Agent0 launcher.
export CONFIG_NAME="${CONFIG_NAME:-agent0_trainer_2x4090}"
export N_GPUS=2

exec bash scripts/rl-agent0.sh "$@"
