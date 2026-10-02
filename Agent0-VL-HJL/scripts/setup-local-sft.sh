#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Install in an isolated Python 3.12 environment; existing Torch/vLLM is read-only.
.venv/bin/python -m venv .venv-sft
MAIN_SITE=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
SFT_SITE=$(.venv-sft/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n' "$MAIN_SITE" > "$SFT_SITE/hjl_main_readonly.pth"
printf '%s\n' 'torch==2.13.0+cu132' 'vllm==0.30.0' > .venv-sft/local_constraints.txt
.venv-sft/bin/python -m pip install -c .venv-sft/local_constraints.txt setuptools wheel ninja pybind11 cmake
.venv-sft/bin/python -m pip install --no-build-isolation -c .venv-sft/local_constraints.txt 'ms-swift==4.5.3' 'megatron-core==0.16.1' 'transformer-engine[core-cu13]==2.11.0'
# Compile extension on CPU using the main environment's CUDA 13 toolkit.
# Fixed hardware architecture prevents auto-detecting or querying a GPU.
export CUDA_VISIBLE_DEVICES=""
export CUDA_HOME="$MAIN_SITE/nvidia/cu13"
export PATH="$PWD/.venv-sft/bin:$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPLUS_INCLUDE_PATH="$MAIN_SITE/nvidia/cudnn/include:$CUDA_HOME/include:$CUDA_HOME/cccl/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CUDNN_HOME="$MAIN_SITE/nvidia/cudnn"
export LD_LIBRARY_PATH="$MAIN_SITE/nvidia/cudnn/lib:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$MAIN_SITE/nvidia/cudnn/lib:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export MAX_JOBS=2 NVTE_BUILD_THREADS=2 TORCH_CUDA_ARCH_LIST=8.9 NVTE_CUDA_ARCHS=89
.venv-sft/bin/python -m pip install --no-build-isolation -c .venv-sft/local_constraints.txt 'transformer-engine[pytorch]==2.11.0'
.venv-sft/bin/python -c 'import torch, swift, megatron.core, transformer_engine; assert torch.__version__ == "2.13.0+cu132"; print("SFT CPU imports passed; GPU execution remains unvalidated")'
