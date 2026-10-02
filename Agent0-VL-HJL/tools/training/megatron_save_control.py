"""Local Megatron compatibility and explicit checkpoint save control."""

import torch.distributed as distributed

from megatron.core.tensor_parallel import layers, mappings


def _use_current_collectives():
    """Replace cached legacy functions only when the replacement is available.

    Installed PyTorch exposes the same signatures under the new names. Older
    versions keep their original functions; no collective semantics change.
    """
    for module in (layers, mappings):
        for cached_name, old_name, new_name in (
            ('dist_all_gather_func', 'all_gather_into_tensor', 'all_gather_single'),
            ('dist_reduce_scatter_func', 'reduce_scatter_tensor', 'reduce_scatter_single'),
        ):
            current = getattr(module, cached_name, None)
            replacement = getattr(distributed, new_name, None)
            if replacement is not None and current is getattr(distributed, old_name, None):
                setattr(module, cached_name, replacement)


_use_current_collectives()

from swift.megatron.callbacks.default_flow import DefaultFlowCallback


if not getattr(DefaultFlowCallback, "_agent0_save_control_installed", False):
    _original_on_step_end = DefaultFlowCallback.on_step_end

    def _on_step_end_with_save_control(self):
        _original_on_step_end(self)
        if int(getattr(self.args, "save_steps", 0) or 0) <= 0:
            # DefaultFlowCallback requests a checkpoint on the final iteration even when
            # save_steps is zero. Clear that request only for the explicit no-save mode.
            self.state.should_save = False

    DefaultFlowCallback.on_step_end = _on_step_end_with_save_control
    DefaultFlowCallback._agent0_save_control_installed = True
