"""Opt-in Swift SFT saved-activation offload, including checkpoint backward."""
import functools
import os
from contextlib import nullcontext

import torch
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer


def install():
    original = Seq2SeqTrainer.training_step
    if getattr(original, '_agent0_cpu_offload', False):
        return

    @functools.wraps(original)
    def training_step(self, model, inputs, *args, **kwargs):
        mode = os.environ.get('SFT_ACTIVATION_CPU_OFFLOAD', 'auto')
        token_limit = int(os.environ.get('SFT_OFFLOAD_TOKEN_THRESHOLD', '8192'))
        free_limit = float(os.environ.get('SFT_OFFLOAD_MIN_FREE_GB', '8'))
        ids = inputs.get('input_ids')
        # Include expanded image tokens and padded batch volume. Larger batches
        # can require offload even when individual sequences are short.
        token_count = ids.numel() if isinstance(ids, torch.Tensor) else None
        free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        free_gb = free_bytes / 1024**3
        reasons = []
        if token_count is None:
            reasons.append('unknown_length')
        elif token_count >= token_limit:
            reasons.append('token_budget')
        if free_gb < free_limit:
            reasons.append('low_free_memory')
        offload = mode == 'true' or (mode == 'auto' and bool(reasons))
        if mode == 'true':
            reasons = ['forced']
        elif mode == 'false':
            reasons = ['disabled']
        print(
            f'[sft-activation-offload pid={os.getpid()}] '
            f'step={self.state.global_step} mode={mode} enabled={offload} '
            f'batch_tokens={token_count} threshold={token_limit} '
            f'free_gb={free_gb:.2f} min_free_gb={free_limit:g} '
            f'reason={",".join(reasons) or "short_with_headroom"}', flush=True)
        context = torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else nullcontext()
        # Include backward: reentrant checkpointing recomputes inside this scope.
        with context:
            return original(self, model, inputs, *args, **kwargs)

    training_step._agent0_cpu_offload = True
    Seq2SeqTrainer.training_step = training_step


install()
