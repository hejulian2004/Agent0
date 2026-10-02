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

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def _normalize_vllm_weight_name(name: str, model) -> str:
    """Normalize converted Transformers Qwen2.5-VL names for vLLM.

    Recent Transformers versions expose the converted model as
    model.visual and model.language_model. Current vLLM releases consume
    the corresponding visual/language_model layout through their own
    ``WeightsMapper``; normalize the FSDP state before calling
    ``model.load_weights`` so both checkpoint layouts remain supported.
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
                 device_mesh: DeviceMesh = None, local_profile=False,
                 role_policies=None, checkpointed=None):
        self.local_profile = local_profile
        self.role_policies = role_policies
        self.checkpointed = checkpointed
        self.native_role_lora = None
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig(offload_to_cpu=local_profile, rank0_only=False))
        else:
            FSDP.set_state_dict_type(self.module,
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
        world_size = torch.distributed.get_world_size()
        for name in list(params):
            parameter = params[name]
            if world_size != 1 and hasattr(parameter, 'full_tensor'):
                parameter = parameter.full_tensor()
            params[name] = parameter.detach().cpu().contiguous() if self.local_profile else parameter
        if self.local_profile:
            if self.checkpointed:
                from pathlib import Path
                import uuid
                from tools.training.role_weight_sync import split_weights, save_snapshots
                from tools.training.native_role_lora import NativeRoleLoRA
                self.role_policies.begin_rollout()
                params, adapters = split_weights(params, self.role_policies.model)
                identity = [uuid.uuid4().hex if torch.distributed.get_rank() == 0 else None]
                torch.distributed.broadcast_object_list(identity, src=0)
                root = Path(self.checkpointed.output_root) / 'adapter_snapshots' / identity[0]
                paths = {mode: str((root / mode).resolve()) for mode in adapters}
                if torch.distributed.get_rank() == 0:
                    save_snapshots(root, adapters, self.role_policies.model.peft_config)
                    (root / 'README.md').write_text(
                        'Native role adapter snapshot\n\nProtocol: ' + self.checkpointed.protocol_version +
                        '\nPolicy versions: ' + str(self.role_policies.versions) +
                        '\nStatus: immutable rollout input; frozen base synchronized separately.\n')
                torch.distributed.barrier()
                del adapters
                self.native_role_lora = NativeRoleLoRA(self.inference_engine.llm_engine,
                    paths, self.role_policies.versions)
            else:
                params = _merge_peft_weights_for_vllm(params, self.module)
            if any('.lora_' in name for name in params):
                raise RuntimeError('Unmerged LoRA weights in rollout synchronization')
            from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu
            offload_fsdp_model_to_cpu(self.module)
        torch.cuda.empty_cache()
        self.inference_engine.wake_up()
        weights = [(_normalize_vllm_weight_name(name, self.module) if self.local_profile else name, value)
                   for name, value in params.items()]
        self.inference_engine.collective_rpc('reload_weights', args=(weights,))
        del weights
        if self.native_role_lora is not None:
            self.native_role_lora.begin_rollout()

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
        if self.native_role_lora is not None:
            self.native_role_lora.end_rollout()
            self.role_policies.end_rollout()
            self.native_role_lora = None
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        # TODO(ZSL): check this
        self.inference_engine.sleep(level=2 if self.local_profile else 1)
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
        group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
