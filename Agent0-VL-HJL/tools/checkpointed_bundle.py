"""Register one shared SFT adapter against a frozen base model."""
import argparse
import hashlib
import json
from pathlib import Path

from agent0_protocol.checkpointed import ADAPTERS, ADAPTER_FOR_MODE, ADAPTER_LAYOUT, PROTOCOL


def register(base_model, adapters, output):
    base = Path(base_model).resolve()
    if not (base / 'config.json').is_file() or set(adapters) != set(ADAPTERS):
        raise ValueError('A base HF model and one shared adapter export are required')
    if not list(base.glob('*.safetensors')):
        raise ValueError('Frozen base model safetensors are missing')
    entries, fingerprints, configs = {}, {}, {}
    for mode in ADAPTERS:
        path = Path(adapters[mode]).resolve()
        weights = path / 'adapter_model.safetensors'
        config = path / 'adapter_config.json'
        if not weights.is_file() or not config.is_file():
            raise ValueError('Missing independent PEFT export: ' + mode)
        value = json.loads(config.read_text())
        if value.get('bias', 'none') != 'none' or value.get('peft_type') != 'LORA':
            raise ValueError('Adapters must be LoRA with frozen base biases')
        if value.get('modules_to_save') or value.get('use_dora'):
            raise ValueError('Native role policies require plain LoRA and a frozen base')
        entries[mode] = str(path)
        configs[mode] = value
        fingerprints[mode] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (weights, config)}
    if len({(v['r'], v['lora_alpha'], tuple(sorted(v['target_modules']))) for v in configs.values()}) != 1:
        raise ValueError('All role exports must use compatible LoRA shapes')
    manifest = {'protocol_version': PROTOCOL, 'adapter_layout_version': ADAPTER_LAYOUT,
        'mode_to_adapter': dict(ADAPTER_FOR_MODE), 'base_model': str(base), 'adapters': entries,
        'adapter_hashes': fingerprints,
        'base_config_sha256': hashlib.sha256((base / 'config.json').read_bytes()).hexdigest(),
        'base_weight_files': {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                              for p in sorted(base.glob('*.safetensors'))},
        'strategy': 'shared', 'weights_summed': False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'bundle.json').exists():
        raise ValueError('Refusing to overwrite an existing adapter bundle')
    (output / 'bundle.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True)
    for mode in ADAPTERS:
        parser.add_argument('--' + mode, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    register(args.base_model, {mode: getattr(args, mode) for mode in ADAPTERS}, args.output)
    print('Registered one shared adapter for solve/repair/verify; no model weights were merged.')


if __name__ == '__main__':
    main()
