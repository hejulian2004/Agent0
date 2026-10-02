"""Separate frozen base weights and role adapters for native vLLM synchronization."""
import os
from pathlib import Path
import shutil
import uuid

from agent0_protocol.checkpointed import MODES


def split_weights(state, model):
    from peft import get_peft_model_state_dict

    if set(model.peft_config) != set(MODES):
        raise ValueError('Expected exactly three role adapters')
    adapters = {mode: get_peft_model_state_dict(model, state_dict=state,
        adapter_name=mode, save_embedding_layers=False) for mode in MODES}
    for weights in adapters.values():
        for name in list(weights):
            if 'visual' in name:
                if '.lora_B.' in name and weights[name].count_nonzero().item():
                    raise ValueError('SFT adapter modifies frozen vision weights')
                del weights[name]
    if any(not weights for weights in adapters.values()):
        raise ValueError('Missing consolidated role adapter weights')
    base = {}
    for name, value in state.items():
        if '.lora_' in name:
            continue
        if name.startswith('base_model.model.'):
            name = name[len('base_model.model.'):]
        name = name.replace('.base_layer.', '.')
        if name in base:
            raise ValueError('Ambiguous frozen base weight: ' + name)
        base[name] = value
    return base, adapters


def save_snapshots(path, weights, peft_configs):
    from safetensors.torch import save_file

    path = Path(path).resolve()
    if path.exists():
        raise ValueError('Refusing to replace rollout adapter snapshots')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        for mode in MODES:
            directory = temporary / mode
            directory.mkdir()
            peft_configs[mode].save_pretrained(directory)
            save_file({name: value.detach().cpu().contiguous().clone()
                       for name, value in weights[mode].items()},
                      directory / 'adapter_model.safetensors')
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return {mode: str(path / mode) for mode in MODES}
