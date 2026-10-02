"""Resolve shared CUDA libraries without initializing any GPU."""
import json
import os
from pathlib import Path
import subprocess


def environment(root, existing=None):
    env = dict(existing or os.environ)
    result = subprocess.run([str(Path(root) / '.venv/bin/python'), '-c',
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
        capture_output=True, text=True, check=True, env=dict(env, CUDA_VISIBLE_DEVICES=''))
    site = Path(result.stdout.strip())
    env['CUDA_HOME'] = str(site / 'nvidia/cu13')
    env['CUDNN_HOME'] = str(site / 'nvidia/cudnn')
    libraries = [site / 'nvidia/cudnn/lib', site / 'nvidia/cu13/lib', site / 'nvidia/nccl/lib', site / 'torch/lib']
    env['LD_LIBRARY_PATH'] = os.pathsep.join([*(str(p) for p in libraries), env.get('LD_LIBRARY_PATH', '')])
    return env


def readiness(root):
    python = Path(root) / '.venv-sft/bin/python'
    if not python.exists():
        return {'ready': False, 'reason': 'missing .venv-sft'}
    result = subprocess.run([str(python), '-c',
        'import importlib.metadata as m; import torch, swift, megatron.core, transformer_engine; '
        'print({p:m.version(p) for p in ["torch","ms-swift","megatron-core","transformer-engine"]})'],
        capture_output=True, text=True, timeout=60, env=dict(environment(root), CUDA_VISIBLE_DEVICES=''))
    return {'ready': result.returncode == 0, 'versions': result.stdout.strip(),
            'reason': result.stderr.splitlines()[-1] if result.returncode else None}
