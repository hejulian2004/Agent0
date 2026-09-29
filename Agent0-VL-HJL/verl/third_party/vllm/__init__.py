"""vLLM 0.30.0 integration; unsupported versions fail at import."""

from importlib.metadata import version

vllm_version = version("vllm")
if vllm_version != "0.30.0":
    raise RuntimeError(f"Agent0-Vl-HJL requires vllm==0.30.0, got {vllm_version}")

from vllm import LLM  # noqa: E402
from vllm.distributed import parallel_state  # noqa: E402

__all__ = ["LLM", "parallel_state", "vllm_version"]
