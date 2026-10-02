"""One shared PEFT policy for three logical modes; no weight merging."""
from pathlib import Path
import hashlib
import json

from agent0_protocol.checkpointed import MODES, ADAPTERS, ADAPTER_FOR_MODE, ADAPTER_LAYOUT, PROTOCOL


def validate_bundle(path, base_model):
    path = Path(path).resolve()
    manifest = json.loads((path / 'bundle.json').read_text())
    if (manifest.get('protocol_version') != PROTOCOL or
            manifest.get('adapter_layout_version') != ADAPTER_LAYOUT or
            manifest.get('mode_to_adapter') != ADAPTER_FOR_MODE or
            manifest.get('strategy') != 'shared' or
            set(manifest.get('adapters', {})) != set(ADAPTERS) or
            set(manifest.get('adapter_hashes', {})) != set(ADAPTERS)):
        raise ValueError('Old checkpoints cannot resume checkpointed training')
    if Path(manifest['base_model']).resolve() != Path(base_model).resolve():
        raise ValueError('Adapter bundle base model mismatch')
    base = Path(base_model).resolve()
    if manifest.get('base_config_sha256') != hashlib.sha256((base / 'config.json').read_bytes()).hexdigest():
        raise ValueError('Adapter bundle base configuration changed')
    actual_weights = {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                      for p in sorted(base.glob('*.safetensors'))}
    if manifest.get('base_weight_files') != actual_weights:
        raise ValueError('Adapter bundle base weights changed')
    adapters = {}
    for mode in ADAPTERS:
        adapter = (path / manifest['adapters'][mode]).resolve()
        config = json.loads((adapter / 'adapter_config.json').read_text())
        if not (adapter / 'adapter_model.safetensors').is_file():
            raise ValueError('Missing adapter weights: ' + mode)
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            if manifest.get('adapter_hashes', {}).get(mode, {}).get(name) != hashlib.sha256((adapter / name).read_bytes()).hexdigest():
                raise ValueError('Adapter bundle content changed: ' + mode)
        if config.get('bias', 'none') != 'none':
            raise ValueError('Adapter policies must leave base biases frozen')
        if config.get('modules_to_save') or config.get('use_dora'):
            raise ValueError('Native role policies require plain LoRA and a frozen base')
        adapters[mode] = str(adapter)
    return manifest, adapters


class RoleAdapters:
    """CPU-testable policy ownership, to be wrapped by the distributed worker."""
    def __init__(self, base_model, config, learning_rate):
        from peft import LoraConfig, get_peft_model
        self.model = get_peft_model(base_model, LoraConfig(r=config['rank'],
            lora_alpha=config['alpha'], target_modules=config['target_modules'],
            bias='none'), adapter_name='shared')
        self._setup_optimizers(learning_rate)

    @classmethod
    def from_bundle(cls, base_model, base_path, bundle_path, learning_rate):
        from peft import PeftModel
        _, adapters = validate_bundle(bundle_path, base_path)
        instance = cls.__new__(cls)
        instance.model = PeftModel.from_pretrained(base_model, adapters['shared'],
                                                   adapter_name='shared', is_trainable=True)
        instance._setup_optimizers(learning_rate)
        return instance

    def _setup_optimizers(self, learning_rate, **optimizer_kwargs):
        from torch.optim import AdamW
        self.parameters = {mode: [p for name, p in self.model.named_parameters()
            if f'.{mode}.' in name and 'lora_' in name and 'visual' not in name] for mode in ADAPTERS}
        if any(not values for values in self.parameters.values()):
            raise ValueError('A role adapter has no trainable parameters')
        self.optimizers = {mode: AdamW(self.parameters[mode], lr=learning_rate,
                                      **optimizer_kwargs) for mode in ADAPTERS}
        self.versions = dict.fromkeys(ADAPTERS, 0)
        self.rollout_versions = None
        self.consumed_phases = set()

    def claim_update(self, phase, policy_version):
        if self.rollout_versions is not None:
            raise RuntimeError('Cannot update a policy during rollout')
        if str(policy_version) != str(self.versions['shared']):
            raise RuntimeError('Stale shared rollout batch')
        if phase in self.consumed_phases:
            raise RuntimeError('Shared rollout batch already consumed')
        self.consumed_phases.add(phase)

    def select(self, mode):
        if mode not in MODES:
            raise ValueError('Unknown adapter mode: ' + str(mode))
        self.model.set_adapter(ADAPTER_FOR_MODE[mode])
        for name, parameter in self.model.named_parameters():
            if 'visual' in name:
                parameter.requires_grad_(False)
        return self.model

    def select_fsdp(self, mode, trainable=True):
        """Keep original-parameter gradient flags stable across FSDP forwards.

        All modes select the same LoRA. Base and vision parameters stay frozen;
        the reference copy disables every gradient.
        """
        self.select(mode)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(trainable and 'lora_' in name and 'visual' not in name)
        return self.model

    def begin_rollout(self):
        if self.rollout_versions is not None:
            raise RuntimeError('Rollout already active')
        self.rollout_versions = dict(self.versions)
        return dict(self.rollout_versions)

    def end_rollout(self):
        if self.rollout_versions != self.versions:
            raise RuntimeError('Policy changed during rollout')
        self.rollout_versions = None

    def step(self, mode):
        if self.rollout_versions is not None:
            raise RuntimeError('Adapter updates are forbidden during rollout')
        mode = ADAPTER_FOR_MODE[mode]
        self.optimizers[mode].step()
        self.optimizers[mode].zero_grad(set_to_none=True)
        self.versions[mode] += 1

    def state_dict(self):
        return {'protocol_version': PROTOCOL, 'adapter_layout_version': ADAPTER_LAYOUT,
                'mode_to_adapter': dict(ADAPTER_FOR_MODE), 'versions': dict(self.versions),
                'consumed_phases': sorted(self.consumed_phases),
                'optimizers': {mode: optimizer.state_dict()
                               for mode, optimizer in self.optimizers.items()}}

    def load_state_dict(self, state):
        if (state.get('protocol_version') != PROTOCOL or
                state.get('adapter_layout_version') != ADAPTER_LAYOUT or
                state.get('mode_to_adapter') != ADAPTER_FOR_MODE or
                set(state.get('versions', {})) != set(ADAPTERS) or
                set(state.get('optimizers', {})) != set(ADAPTERS)):
            raise ValueError('Incompatible role optimizer checkpoint')
        for mode in ADAPTERS:
            self.optimizers[mode].load_state_dict(state['optimizers'][mode])
        self.versions = dict(state['versions'])
        self.consumed_phases = set(state.get('consumed_phases', []))
