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
"""
Apply monkey-patch function to models
"""
import sys
from typing import Optional

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    get_ulysses_sequence_parallel_group,
)


def _patch_qwen2_5_vl_model_forward():
    """Make Qwen2.5-VL image replacement compatible with Ulysses slicing.

    The actor slices the unpadded token sequence before entering the model.
    Qwen2.5-VL, however, validates image placeholders against the complete
    image feature tensor. Reconstruct the complete token sequence inside the
    model, inject image features once, then return the local slice to the
    language stack.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    if hasattr(Qwen2_5_VLModel, "_verl_ulysses_original_forward"):
        return

    original_forward = Qwen2_5_VLModel.forward

    def forward_with_sharded_multimodal(self, *args, **kwargs):
        is_sharded_multimodal = kwargs.pop("_verl_ulysses_sharded_multimodal", False)
        if not is_sharded_multimodal:
            return original_forward(self, *args, **kwargs)

        input_ids = kwargs.get("input_ids")
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        position_ids = kwargs.get("position_ids")
        sp_group = get_ulysses_sequence_parallel_group()
        sp_size = get_ulysses_sequence_parallel_world_size(sp_group)

        if (input_ids is None or pixel_values is None or image_grid_thw is None or position_ids is None
                or sp_size <= 1 or input_ids.dim() != 2 or input_ids.size(0) != 1):
            return original_forward(self, *args, **kwargs)

        local_input_ids = input_ids
        local_seq_len = local_input_ids.size(-1)
        input_id_chunks = [torch.empty_like(local_input_ids) for _ in range(sp_size)]
        torch.distributed.all_gather(input_id_chunks, local_input_ids, group=sp_group)
        full_input_ids = torch.cat(input_id_chunks, dim=-1)

        full_inputs_embeds = self.get_input_embeddings()(full_input_ids)
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(full_inputs_embeds.device, full_inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            full_input_ids,
            inputs_embeds=full_inputs_embeds,
            image_features=image_embeds,
        )
        full_inputs_embeds = full_inputs_embeds.masked_scatter(image_mask, image_embeds)

        rank = torch.distributed.get_rank(group=sp_group)
        start = rank * local_seq_len
        end = start + local_seq_len
        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = full_inputs_embeds[:, start:end].contiguous()
        kwargs["pixel_values"] = None
        kwargs["image_grid_thw"] = None
        return original_forward(self, *args, **kwargs)

    Qwen2_5_VLModel._verl_ulysses_original_forward = original_forward
    Qwen2_5_VLModel.forward = forward_with_sharded_multimodal
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # Qwen2.5-VL uses the same Transformers flash-attention helper for
        # both language and vision blocks. Vision attention does not receive
        # position_ids, and its inputs are replicated on every SP rank, so it
        # must use the original attention path instead of the text all-to-all.
        if position_ids is None:
            return _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                *args,
                position_ids=None,
                **kwargs,
            )

        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
        # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
        # https://github.com/huggingface/transformers/pull/33932

        # (bsz, seq_len/n) -> (bsz, seq_len)
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = _flash_attention_forward(query_states,
                                           key_states,
                                           value_states,
                                           *args,
                                           position_ids=position_ids,
                                           **kwargs)

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def apply_monkey_patch(model: PreTrainedModel, ulysses_sp_size: int):
    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    module = sys.modules[model.__module__]

    num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    assert num_attention_heads % ulysses_sp_size == 0, \
        f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
    assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
        f"or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,"
        f"kv heads are repeated to ensure correctness.")
    # TODO: VLM models only, unify monkey patch to LLM models.
    if model.config.model_type in ("qwen2_vl", "qwen2_5_vl", "qwen2_5_vl_text"):  # patch remove padding for qwen2vl mrope
        if model.config.model_type in ("qwen2_5_vl", "qwen2_5_vl_text"):
            _patch_qwen2_5_vl_model_forward()
        from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward

        # Transformers 4.57 removed the model-local FlashAttention2 classes
        # used by older VERL/Qwen releases. Use them when available; otherwise
        # fall through to the integration-level patch below.
        try:
            if model.config.model_type == "qwen2_vl":
                from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2
                Qwen2VLFlashAttention2.forward = ulysses_flash_attn_forward
            else:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
                Qwen2_5_VLFlashAttention2.forward = ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in Qwen2VL")
            return
        except ImportError:
            pass

    # transformers<=4.47.1
    if hasattr(module, "_flash_attention_forward"):
        module._flash_attention_forward = _ulysses_flash_attention_forward
        print(f"Monkey patch _flash_attention_forward in {model.__module__}")
    else:
        # transformers>=4.48.0
        from transformers.integrations import flash_attention
        flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
        print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")


from functools import lru_cache
from packaging import version
import importlib.metadata


@lru_cache()
def is_transformers_version_in_range(min_version: str, max_version: str) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        raise ModuleNotFoundError("The `transformers` package is not installed.")

    # Check if the version is within the specified range
    return version.parse(min_version) <= version.parse(transformers_version) <= version.parse(max_version)
