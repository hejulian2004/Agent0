"""Qwen2.5-VL Ulysses using native SDPA and isolated packed samples."""
from functools import wraps
from types import MethodType

import torch
import torch.nn.functional as functional

from verl.utils.ulysses import (get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size, gather_seq_scatter_heads,
    gather_heads_scatter_seq)


def gather_sequence(value, dim=-1):
    size = get_ulysses_sequence_parallel_world_size()
    if size == 1:
        return value
    values = [torch.empty_like(value) for _ in range(size)]
    torch.distributed.all_gather(values, value.contiguous(), group=get_ulysses_sequence_parallel_group())
    return torch.cat(values, dim=dim)


def packed_sdpa(query, key, value, positions, dropout=0.0, scale=None):
    """No dense N*N mask: invoke causal SDPA independently per sample."""
    if positions is None:
        return functional.scaled_dot_product_attention(query, key, value, is_causal=True,
                                                        dropout_p=dropout, scale=scale, enable_gqa=True)
    positions = positions.reshape(-1)
    starts = torch.nonzero(positions[1:] < positions[:-1], as_tuple=False).flatten().add(1).tolist()
    bounds = [0, *starts, query.shape[-2]]
    return torch.cat([functional.scaled_dot_product_attention(
        query[..., left:right, :], key[..., left:right, :], value[..., left:right, :],
        is_causal=True, dropout_p=dropout, scale=scale, enable_gqa=True)
        for left, right in zip(bounds[:-1], bounds[1:]) if right > left], dim=-2)


def attention(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
    size = get_ulysses_sequence_parallel_world_size()
    positions = kwargs.get("_hjl_full_positions")
    if size > 1:
        repeats = max(size // key.shape[1], 1)
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        query = gather_seq_scatter_heads(query.transpose(1, 2), seq_dim=1, head_dim=2).transpose(1, 2)
        key = gather_seq_scatter_heads(key.transpose(1, 2), seq_dim=1, head_dim=2).transpose(1, 2)
        value = gather_seq_scatter_heads(value.transpose(1, 2), seq_dim=1, head_dim=2).transpose(1, 2)
    output = packed_sdpa(query, key, value, positions, dropout, scaling).transpose(1, 2)
    if size > 1:
        output = gather_heads_scatter_seq(output, seq_dim=1, head_dim=2)
    return output, None


def install(model, size):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register("hjl_ulysses_sdpa", attention)
    core = model.model
    language = core.language_model
    if getattr(core, "_hjl_sp_installed", False):
        return
    heads = language.config.num_attention_heads
    if heads % size:
        raise ValueError("Qwen attention heads must divide Ulysses size")
    # Only text attention uses the collective; visual attention stays SDPA.
    language.config._attn_implementation = "hjl_ulysses_sdpa"
    original_core = core.forward
    original_language = language.forward

    @wraps(original_core)
    def core_forward(self, *args, **kwargs):
        sharded = kwargs.pop("_verl_ulysses_sharded_multimodal", False)
        if sharded and size > 1:
            kwargs["input_ids"] = gather_sequence(kwargs["input_ids"])
            kwargs["position_ids"] = gather_sequence(kwargs["position_ids"])
            kwargs["_hjl_slice_embeddings"] = True
        return original_core(*args, **kwargs)

    @wraps(original_language)
    def language_forward(self, *args, **kwargs):
        slicing = kwargs.pop("_hjl_slice_embeddings", False)
        positions = kwargs["position_ids"]
        full = positions if slicing else gather_sequence(positions)
        # mRoPE's temporal channel increases through images but never resets
        # inside one sample; zeros/decreases delimit packed samples/padding.
        kwargs["_hjl_full_positions"] = full[0, 0] if full.ndim == 3 else full[0]
        if slicing:
            rank = torch.distributed.get_rank(get_ulysses_sequence_parallel_group())
            kwargs["inputs_embeds"] = kwargs["inputs_embeds"].chunk(size, dim=1)[rank].contiguous()
            kwargs["position_ids"] = positions.chunk(size, dim=-1)[rank].contiguous()
        kwargs["attention_mask"] = {"full_attention": None, "sliding_attention": None}
        kwargs["use_cache"] = False
        return original_language(*args, **kwargs)

    core.forward = MethodType(core_forward, core)
    language.forward = MethodType(language_forward, language)
    core._hjl_sp_installed = True
