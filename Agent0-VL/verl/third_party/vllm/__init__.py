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

import inspect
from importlib.metadata import version, PackageNotFoundError
from packaging import version as vs
from verl.utils.import_utils import is_sglang_available


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)
vllm_version = None


def vllm_mm_cache_kwargs():
    """Return the multimodal-cache option supported by the installed vLLM.

    vLLM 0.8 exposed ``disable_mm_preprocessor_cache``.  Newer releases
    replaced it with the size-based ``mm_processor_cache_gb`` setting.  Keep
    the cache disabled for the memory-constrained RL rollout without passing
    a removed keyword to the newer EngineArgs API.
    """
    try:
        from vllm.engine.arg_utils import EngineArgs

        parameters = inspect.signature(EngineArgs).parameters
    except (ImportError, TypeError, ValueError):
        return {'disable_mm_preprocessor_cache': True}

    if 'disable_mm_preprocessor_cache' in parameters:
        return {'disable_mm_preprocessor_cache': True}
    if 'mm_processor_cache_gb' in parameters:
        return {'mm_processor_cache_gb': 0}
    return {}

if package_version == '0.3.1':
    vllm_version = '0.3.1'
    from .vllm_v_0_3_1.llm import LLM
    from .vllm_v_0_3_1.llm import LLMEngine
    from .vllm_v_0_3_1 import parallel_state
elif package_version == '0.4.2':
    vllm_version = '0.4.2'
    from .vllm_v_0_4_2.llm import LLM
    from .vllm_v_0_4_2.llm import LLMEngine
    from .vllm_v_0_4_2 import parallel_state
elif package_version == '0.5.4':
    vllm_version = '0.5.4'
    from .vllm_v_0_5_4.llm import LLM
    from .vllm_v_0_5_4.llm import LLMEngine
    from .vllm_v_0_5_4 import parallel_state
elif package_version == '0.6.3':
    vllm_version = '0.6.3'
    from .vllm_v_0_6_3.llm import LLM
    from .vllm_v_0_6_3.llm import LLMEngine
    from .vllm_v_0_6_3 import parallel_state
elif package_version == '0.6.3+rocm624':
    vllm_version = '0.6.3'
    from .vllm_v_0_6_3.llm import LLM
    from .vllm_v_0_6_3.llm import LLMEngine
    from .vllm_v_0_6_3 import parallel_state
elif vs.parse(package_version) >= vs.parse('0.7.0'):
    # From 0.6.6.post2 on, vllm supports SPMD inference
    # See https://github.com/vllm-project/vllm/pull/12071

    # Keep the concrete version available to callers.  The old code left this
    # as None for every modern vLLM release, which made version-specific
    # compatibility branches ambiguous after upgrading vLLM.
    vllm_version = package_version
    from vllm import LLM
    from vllm.distributed import parallel_state
else:
    if not is_sglang_available():
        raise ValueError(
            f'vllm version {package_version} not supported and SGLang also not Found. Currently supported vllm versions are 0.3.1, 0.4.2, 0.5.4, 0.6.3 and 0.7.0+'
        )
