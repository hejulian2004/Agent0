"""Honor save_steps=0 by disabling Megatron-SWIFT's unconditional final save."""

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
