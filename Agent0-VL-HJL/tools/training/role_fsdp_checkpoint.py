"""Role optimizer sidecars for VERL's existing FSDP model checkpoints."""
import hashlib
import json
import os
from pathlib import Path

import torch

from agent0_protocol.checkpointed import ADAPTERS, ADAPTER_FOR_MODE, ADAPTER_LAYOUT, PROTOCOL
from tools.training.role_checkpoint import cpu_tree


def validate(path, fingerprint):
    path = Path(path)
    manifest = json.loads((path / 'role_checkpoint.json').read_text())
    if (manifest.get('protocol_version') != PROTOCOL or manifest.get('fingerprint') != fingerprint or
            manifest.get('adapter_layout_version') != ADAPTER_LAYOUT or
            manifest.get('mode_to_adapter') != ADAPTER_FOR_MODE):
        raise ValueError('Incompatible role FSDP checkpoint')
    if set(manifest.get('versions', {})) != set(ADAPTERS):
        raise ValueError('Missing role policy versions')
    if manifest['world_size'] != torch.distributed.get_world_size():
        raise ValueError('Role checkpoint world size changed')
    for name, expected in manifest['state_hashes'].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Role checkpoint state hash mismatch: ' + name)
    return manifest


def save(path, model, policies, schedulers, fingerprint):
    if policies.rollout_versions is not None:
        raise RuntimeError('Cannot save during rollout')
    path = Path(path)
    rank = torch.distributed.get_rank()
    state = {'protocol_version': PROTOCOL, 'adapter_layout_version': ADAPTER_LAYOUT,
        'mode_to_adapter': dict(ADAPTER_FOR_MODE), 'fingerprint': fingerprint,
        'versions': dict(policies.versions), 'consumed_phases': sorted(policies.consumed_phases),
        # Keep rank-local original-parameter optimizer shards, as VERL's
        # checkpoint manager does. FSDP's consolidated optimizer API assumes
        # one optimizer owns every trainable parameter in each flat shard.
        'optimizers': {mode: cpu_tree(policies.optimizers[mode].state_dict())
                       for mode in ADAPTERS},
        'parameter_shapes': {mode: [list(p.shape) for p in policies.parameters[mode]] for mode in ADAPTERS},
        'schedulers': {mode: schedulers[mode].state_dict() for mode in ADAPTERS}}
    name = f'role_state_rank_{rank}.pt'
    temporary = path / (name + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path / name)
    hashes = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(hashes, (name, hashlib.sha256((path / name).read_bytes()).hexdigest()))
    if rank == 0:
        manifest = {'protocol_version': PROTOCOL, 'adapter_layout_version': ADAPTER_LAYOUT,
                    'mode_to_adapter': dict(ADAPTER_FOR_MODE), 'fingerprint': fingerprint,
                    'world_size': len(hashes), 'versions': dict(policies.versions),
                    'state_hashes': dict(hashes)}
        temporary = path / 'role_checkpoint.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2))
        os.replace(temporary, path / 'role_checkpoint.json')
    torch.distributed.barrier()


def load(path, model, policies, schedulers, fingerprint):
    manifest = validate(path, fingerprint)
    state = torch.load(Path(path) / f'role_state_rank_{torch.distributed.get_rank()}.pt',
                       map_location='cpu', weights_only=False)
    if (state.get('protocol_version') != PROTOCOL or state.get('fingerprint') != fingerprint or
            state.get('adapter_layout_version') != ADAPTER_LAYOUT or
            state.get('mode_to_adapter') != ADAPTER_FOR_MODE):
        raise ValueError('Incompatible role optimizer sidecar')
    if set(state['optimizers']) != set(ADAPTERS) or set(state['schedulers']) != set(ADAPTERS):
        raise ValueError('Missing role optimizer or scheduler state')
    if state['versions'] != manifest['versions']:
        raise ValueError('Inconsistent role policy versions')
    for mode in ADAPTERS:
        optimizer = policies.optimizers[mode]
        if state['parameter_shapes'][mode] != [list(p.shape) for p in policies.parameters[mode]]:
            raise ValueError('Role optimizer shard layout changed: ' + mode)
        optimizer.load_state_dict(state['optimizers'][mode])
        schedulers[mode].load_state_dict(state['schedulers'][mode])
    policies.versions = dict(state['versions'])
    policies.consumed_phases = set(state.get('consumed_phases', []))
