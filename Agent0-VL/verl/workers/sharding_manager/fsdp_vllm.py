# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import torch
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.protocol import all_gather_data_proto
from verl.utils.debug import log_gpu_memory_usage
from verl.third_party.vllm import vllm_version

from .base import BaseShardingManager
from .patch import patched_ds_v3_load_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def _normalize_vllm_weight_name(name: str, model) -> str:
    """Normalize converted Transformers Qwen2.5-VL names for vLLM.

    Recent Transformers versions expose the converted model as
    model.visual and model.language_model. vLLM 0.8.x consumes the
    original Hugging Face layout (visual and model) and applies its
    own mapper afterwards.
    """
    # PEFT wraps the base model below ``base_model.model``. vLLM is
    # initialized from the unwrapped checkpoint, so remove that wrapper.
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]

    if getattr(getattr(model, 'config', None), 'model_type', None) in ('qwen2_5_vl', 'qwen2_5_vl_text'):
        if name.startswith('model.visual.'):
            return name[len('model.'):]
        if name.startswith('model.language_model.'):
            return 'language_model.model.' + name[len('model.language_model.'):]
    return name


def _dequantize_bnb_state_dict(params, shape_overrides=None):
    """Convert bitsandbytes 4-bit entries to dense CPU tensors for vLLM.

    The training actor keeps NF4 weights packed in ``Params4bit`` objects, but
    vLLM's HF weight loader expects the dense dtype used by the base model.
    bitsandbytes stores the quantization metadata beside each packed weight in
    the state dict, so reconstruct the ``QuantState`` before loading vLLM.
    """
    quantized_weights = [
        name for name in params
        if (name == 'weight' or name.endswith('.weight'))
        and any(key.startswith(name + '.quant_state.') for key in params)
    ]
    if not quantized_weights:
        return params

    from bitsandbytes.functional import QuantState, dequantize_4bit

    shape_overrides = shape_overrides or {}

    for name in quantized_weights:
        prefix = name + '.'
        state_dict = {
            key[len(prefix):]: value
            for key, value in params.items()
            if key.startswith(prefix)
        }
        weight = params[name]
        try:
            quant_state = QuantState.from_dict(state_dict, device=weight.device)
            # FSDP can replace the original Params4bit shape with the packed
            # storage shape. Recover the dense shape from the LoRA factors
            # before asking bitsandbytes to dequantize the tensor.
            if name in shape_overrides:
                quant_state.shape = tuple(shape_overrides[name])
            dense = dequantize_4bit(weight, quant_state=quant_state)
        except Exception as exc:
            raise RuntimeError(f'Failed to dequantize QLoRA weight {name}') from exc

        # QuantState.dtype describes the statistics tensor in some bnb
        # versions, not the model compute dtype. The Agent0-VL QLoRA profile
        # uses BF16 for both FSDP storage and vLLM weights.
        compute_dtype = getattr(quant_state, 'dtype', torch.bfloat16)
        if compute_dtype not in (torch.float16, torch.bfloat16):
            compute_dtype = torch.bfloat16
        params[name] = dense.to(dtype=compute_dtype).contiguous()
        for key in list(params):
            if key.startswith(prefix):
                del params[key]

    return params


def _merge_peft_weights_for_vllm(params, module):
    """Fold in-memory LoRA weights into dense tensors for vLLM.

    The rollout engine is initialized from the base checkpoint and this
    path does not send a LoRARequest per generation. Fold the adapter into
    a temporary state dict so vLLM follows the current actor while the
    training module remains unmerged and trainable.
    """
    if not any(".lora_A." in name for name in params):
        return params

    wrapped = getattr(module, "_fsdp_wrapped_module", module)
    peft_configs = getattr(wrapped, "peft_config", {})
    if not isinstance(peft_configs, dict):
        peft_configs = {}

    for a_name in list(params):
        if ".lora_A." not in a_name or not a_name.endswith(".weight"):
            continue
        prefix, adapter_suffix = a_name.split(".lora_A.", 1)
        adapter_name = adapter_suffix[:-len(".weight")]
        b_name = f"{prefix}.lora_B.{adapter_name}.weight"
        base_name = f"{prefix}.base_layer.weight"
        if b_name not in params or base_name not in params:
            continue

        config = peft_configs.get(adapter_name) or peft_configs.get("default")
        if config is None:
            continue
        scaling = float(config.lora_alpha) / float(config.r)
        base = params[base_name]
        a = params[a_name].to(dtype=base.dtype)
        b = params[b_name].to(dtype=base.dtype)
        delta = torch.matmul(b, a) * scaling
        if getattr(config, "fan_in_fan_out", False):
            delta = delta.transpose(0, 1)
        # FSDP-QLoRA may expose a Params4bit base weight as a flattened
        # storage tensor even when the LoRA factors retain the original 2-D
        # shape. Restore the shape before folding the adapter into vLLM.
        if base.ndim == 1 and base.numel() == delta.numel():
            base = base.reshape(delta.shape)
        if base.shape != delta.shape:
            raise RuntimeError(
                f"QLoRA/vLLM shape mismatch for {prefix}: "
                f"base={tuple(base.shape)}, delta={tuple(delta.shape)}")
        params[f"{prefix}.weight"] = base + delta
        del params[base_name]
        del params[a_name]
        del params[b_name]

    # PEFT also wraps untouched bias parameters as ``base_layer.bias``.
    # vLLM expects the original dense name for those parameters.
    for name in list(params):
        if ".base_layer." in name:
            params[name.replace(".base_layer.", ".")] = params.pop(name)

    return params


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None,
                 offload_actor: bool = False):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_actor = offload_actor

        # Full params
        self.full_params = full_params
        # vLLMRollout sleeps the engine immediately after construction.
        # Mark that initial state so the first rollout context wakes it before
        # loading FSDP weights into vLLM.
        self._vllm_is_sleeping = True
        if full_params:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=False))
        else:
            # Keep only sharded DTensors in the state dict. The iterator below
            # materializes one full parameter at a time on CPU.
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig())

        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()

        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        params = self.module.state_dict()
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'

        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
        else:
            # Move FSDP state tensors off GPU before vLLM remaps its full model.
            world_size = torch.distributed.get_world_size()
            for name in list(params.keys()):
                param = params[name]
                if world_size != 1 and hasattr(param, "full_tensor"):
                    param = param.full_tensor()
                if isinstance(param, torch.Tensor):
                    param = param.detach().to(device="cpu").contiguous()
                params[name] = param
                del param
            shape_overrides = {}
            for a_name in params:
                if ".lora_A." not in a_name or not a_name.endswith(".weight"):
                    continue
                prefix, adapter_suffix = a_name.split(".lora_A.", 1)
                adapter_name = adapter_suffix[:-len(".weight")]
                b_name = f"{prefix}.lora_B.{adapter_name}.weight"
                base_name = f"{prefix}.base_layer.weight"
                if b_name in params and base_name in params:
                    shape_overrides[base_name] = (
                        params[b_name].shape[0], params[a_name].shape[1])
            params = _dequantize_bnb_state_dict(params, shape_overrides)
            params = _merge_peft_weights_for_vllm(params, self.module)
            torch.cuda.empty_cache()

            # With manual FSDP param offload, keep the training actor on CPU
            # while vLLM is resident. This is safe after state_dict() has
            # materialized the tensors above and avoids a second full model
            # competing for GPU memory during wake_up(). The worker owns this
            # policy; do not change the actor placement for profiles that do
            # not request manual parameter offload.
            if self.offload_actor:
                from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu
                offload_fsdp_model_to_cpu(self.module)

            if self._vllm_is_sleeping:
                self.inference_engine.wake_up()
                self._vllm_is_sleeping = False
            world_size = torch.distributed.get_world_size()
            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            def iter_vllm_weights():
                for name, param in params.items():
                    name = _normalize_vllm_weight_name(name, model)
                    if world_size != 1 and hasattr(param, 'full_tensor'):
                        param = param.full_tensor()
                    if isinstance(param, torch.Tensor):
                        param = param.detach().to(device='cpu').contiguous()
                    yield name, param

            if model.config.architectures[0] in ['DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM']:
                loaded_params = patched_ds_v3_load_weights(
                    model, iter_vllm_weights())
            else:
                loaded_params = model.load_weights(iter_vllm_weights())
            logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")

        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        # TODO(ZSL): check this
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)
            self._vllm_is_sleeping = True
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
