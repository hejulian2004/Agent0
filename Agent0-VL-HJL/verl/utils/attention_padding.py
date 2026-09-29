"""PyTorch padding helpers used by the FSDP paths on current CUDA wheels."""

from __future__ import annotations

import torch
from einops import rearrange


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, indices.long())


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("attention mask must match batch and sequence dimensions")
    batch, seqlen = attention_mask.shape
    lengths = attention_mask.long().sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).reshape(-1)
    flat = hidden_states.reshape(batch * seqlen, *hidden_states.shape[2:])
    unpadded = index_first_axis(flat, indices)
    cu_seqlens = torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(lengths.max().item()) if batch else 0
    return unpadded, indices, cu_seqlens, max_seqlen


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    result = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    result.index_copy_(0, indices.long(), hidden_states)
    return result.reshape(batch, seqlen, *hidden_states.shape[1:])


__all__ = ["index_first_axis", "unpad_input", "pad_input", "rearrange"]
