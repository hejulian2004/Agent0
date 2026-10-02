"""Atomic three-adapter checkpoints with independent optimizers and RNG state.

For FSDP the worker must first provide consolidated adapter/optimizer state.
This module handles plain PEFT policies and never reconstructs missing roles.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch

from agent0_protocol.checkpointed import MODES, PROTOCOL
from tools.checkpointed_bundle import register


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def save(path, policies, base_path, fingerprint, schedulers=None):
    if policies.rollout_versions is not None:
        raise RuntimeError('Cannot checkpoint during active rollout')
    path = Path(path).resolve()
    if path.exists():
        raise ValueError('Refusing to replace a role checkpoint')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        policies.model.save_pretrained(temporary / 'adapters', selected_adapters=list(MODES))
        register(base_path, {mode: temporary / 'adapters' / mode for mode in MODES}, temporary / 'bundle')
        manifest_path = temporary / 'bundle' / 'bundle.json'
        manifest = json.loads(manifest_path.read_text())
        # Paths must resolve after the atomic directory rename.
        manifest['adapters'] = {mode: str(path / 'adapters' / mode) for mode in MODES}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        state = cpu_tree(policies.state_dict())
        state['fingerprint'] = fingerprint
        state['schedulers'] = {mode: schedulers[mode].state_dict() if schedulers else None for mode in MODES}
        state['rng'] = {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state()}
        if torch.cuda.is_initialized():
            state['rng']['cuda'] = torch.cuda.get_rng_state_all()
        torch.save(state, temporary / 'training_state.pt')
        info = {'protocol_version': PROTOCOL, 'fingerprint': fingerprint,
            'adapter_versions': policies.versions,
            'training_state_sha256': hashlib.sha256((temporary / 'training_state.pt').read_bytes()).hexdigest()}
        (temporary / 'checkpoint.json').write_text(json.dumps(info, indent=2))
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def load_state(path, policies, fingerprint, schedulers=None):
    path = Path(path)
    info = json.loads((path / 'checkpoint.json').read_text())
    if info['protocol_version'] != PROTOCOL or info['fingerprint'] != fingerprint:
        raise ValueError('Incompatible role checkpoint protocol/config/data/model')
    state_path = path / 'training_state.pt'
    if hashlib.sha256(state_path.read_bytes()).hexdigest() != info['training_state_sha256']:
        raise ValueError('Role optimizer checkpoint hash mismatch')
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    policies.load_state_dict(state)
    if schedulers:
        for mode in MODES:
            if state['schedulers'][mode] is None:
                raise ValueError('Missing role scheduler state: ' + mode)
            schedulers[mode].load_state_dict(state['schedulers'][mode])
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if 'cuda' in state['rng']:
        if not torch.cuda.is_initialized():
            raise ValueError('CUDA RNG restore requires an already initialized training worker')
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state
