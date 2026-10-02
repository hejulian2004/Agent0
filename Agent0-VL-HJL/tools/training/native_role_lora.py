"""Native vLLM adapter routing and immutable rollout policy versions."""
from pathlib import Path

from agent0_protocol.checkpointed import ADAPTERS, ADAPTER_FOR_MODE


class NativeRoleLoRA:
    def __init__(self, engine, paths, versions=None, request_factory=None):
        if set(paths) != set(ADAPTERS):
            raise ValueError('Native rollout requires one shared adapter')
        if request_factory is None:
            from vllm.lora.request import LoRARequest
            request_factory = LoRARequest
        self.engine, self.factory = engine, request_factory
        self.paths = {mode: str(Path(path).resolve()) for mode, path in paths.items()}
        self.versions = dict(versions or dict.fromkeys(ADAPTERS, 0))
        if set(self.versions) != set(ADAPTERS):
            raise ValueError('Incomplete adapter policy versions')
        self.rollout_versions = None
        self.requests, self.active, self.next_id = {}, False, 1

    def begin_rollout(self):
        if self.active:
            raise RuntimeError('Adapter rollout already active')
        requests = {}
        try:
            for mode in ADAPTERS:
                request = self.factory(f'{mode}-v{self.versions[mode]}', self.next_id, self.paths[mode])
                self.next_id += 1
                self.engine.add_lora(request)
                requests[mode] = request
        except Exception:
            for request in requests.values():
                self.engine.remove_lora(request.lora_int_id)
            raise
        self.requests = requests
        self.rollout_versions = dict(self.versions)
        self.active = True

    def request(self, mode):
        if not self.active:
            raise RuntimeError('Native adapter policies must be frozen before rollout')
        if self.versions != self.rollout_versions:
            raise RuntimeError('Policy changed during rollout')
        return self.requests[ADAPTER_FOR_MODE[mode]]

    def end_rollout(self):
        if self.versions != self.rollout_versions:
            raise RuntimeError('Policy changed during rollout')
        self.active = False
        self.rollout_versions = None
        for request in self.requests.values():
            self.engine.remove_lora(request.lora_int_id)
        self.requests = {}

    def replace(self, paths, versions):
        if self.active:
            raise RuntimeError('Cannot update adapters during rollout')
        if set(paths) != set(ADAPTERS) or set(versions) != set(ADAPTERS):
            raise ValueError('Incomplete adapter policy versions')
        self.paths = {mode: str(Path(path).resolve()) for mode, path in paths.items()}
        self.versions = dict(versions)
