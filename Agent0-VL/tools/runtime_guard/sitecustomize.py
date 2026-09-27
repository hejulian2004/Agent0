"""Opt-in DataLoader memory backpressure, without importing torch at startup."""
import importlib.abc
import importlib.machinery
import os
import sys
import time
from pathlib import Path


def memory_pressure():
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        info[name] = int(value.split()[0])
    ratios = [1 - info['MemAvailable'] / info['MemTotal']]
    # Respect a container's memory budget as well as the host's. Reclaimable
    # inactive file cache is excluded, like MemAvailable on the host.
    roots = {Path('/sys/fs/cgroup')}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            directory = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
            if directory.is_dir():
                roots.update([directory, *directory.parents])
    for directory in roots:
        try:
            limit = (directory / 'memory.max').read_text().strip()
            if limit == 'max':
                continue
            used = int((directory / 'memory.current').read_text())
            stats = dict(line.split() for line in (directory / 'memory.stat').read_text().splitlines())
            used -= int(stats.get('inactive_file', 0))
            ratios.append(max(0, used) / int(limit))
        except (OSError, ValueError, ZeroDivisionError):
            continue
    return max(ratios)


def wait_for_memory():
    limit = float(os.environ.get('AGENT0_DATA_MEMORY_PERCENT', '90')) / 100
    timeout = float(os.environ.get('AGENT0_DATA_MEMORY_WAIT_SECONDS', '180'))
    start = time.monotonic()
    pressure = memory_pressure()
    if pressure < limit:
        return
    print(f'[data-memory pid={os.getpid()}] pause loading: memory={pressure:.1%}, limit={limit:.1%}', flush=True)
    last_log = start
    while pressure >= max(0, limit - 0.02):
        if time.monotonic() - start >= timeout:
            raise RuntimeError('Data loading paused too long under memory pressure; reduce workers/prefetch/batch or CPU model offload. This guard is a loading threshold, not a total-process memory cap.')
        time.sleep(1)
        pressure = memory_pressure()
        if time.monotonic() - last_log >= 30:
            print(f'[data-memory pid={os.getpid()}] waiting: memory={pressure:.1%}', flush=True)
            last_log = time.monotonic()
    print(f'[data-memory pid={os.getpid()}] resume loading: memory={pressure:.1%}', flush=True)


class GuardLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        for name in ('_MapDatasetFetcher', '_IterableDatasetFetcher'):
            cls = getattr(module, name)
            original = cls.fetch
            def guarded(self, index, original=original):
                wait_for_memory()
                return original(self, index)
            cls.fetch = guarded


class GuardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'torch.utils.data._utils.fetch':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = GuardLoader(spec.loader)
        return spec


if os.environ.get('AGENT0_DATA_MEMORY_GUARD') == '1':
    sys.meta_path.insert(0, GuardFinder())
