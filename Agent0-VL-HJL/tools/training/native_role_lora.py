"""Native vLLM adapter routing and immutable rollout policy versions."""
from pathlib import Path

from agent0_protocol.checkpointed import MODES


class NativeRoleLoRA:
    def __init__(self, engine, paths, versions=None, request_factory=None):
        if set(paths) != set(MODES):
            raise ValueError('Native rollout requires solve/repair/verify adapters')
        if request_factory is None:
            from vllm.lora.request import LoRARequest
            request_factory = LoRARequest
        self.engine, self.factory = engine, request_factory
        self.paths = {mode: str(Path(path).resolve()) for mode, path in paths.items()}
        self.versions = dict(versions or dict.fromkeys(MODES, 0))
        self.requests, self.active, self.next_id = {}, False, 1

    def begin_rollout(self):
        if self.active:
            raise RuntimeError('Adapter rollout already active')
        requests = {}
        try:
            for mode in MODES:
                request = self.factory(f'{mode}-v{self.versions[mode]}', self.next_id, self.paths[mode])
                self.next_id += 1
                self.engine.add_lora(request)
                requests[mode] = request
        except Exception:
            for request in requests.values():
                self.engine.remove_lora(request.lora_int_id)
            raise
        self.requests = requests
        self.active = True

    def request(self, mode):
        if not self.active:
            raise RuntimeError('Native adapter policies must be frozen before rollout')
        return self.requests[mode]

    def end_rollout(self):
        self.active = False
        for request in self.requests.values():
            self.engine.remove_lora(request.lora_int_id)
        self.requests = {}

    def replace(self, paths, versions):
        if self.active:
            raise RuntimeError('Cannot update adapters during rollout')
        if set(paths) != set(MODES) or set(versions) != set(MODES):
            raise ValueError('Incomplete adapter policy versions')
        self.paths = {mode: str(Path(path).resolve()) for mode, path in paths.items()}
        self.versions = dict(versions)
