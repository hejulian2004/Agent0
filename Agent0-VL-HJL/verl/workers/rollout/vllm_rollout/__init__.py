"""SPMD rollout integrations for the pinned vLLM release."""

from .vllm_rollout_spmd import vLLMRollout
from .vllm_agent0_rollout_spmd import vLLMAgent0Rollout

vllm_mode = "spmd"
__all__ = ["vLLMRollout", "vLLMAgent0Rollout", "vllm_mode"]
