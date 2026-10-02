"""Keep repurposed releases out of future independent benchmarks."""
import json
from pathlib import Path


def assert_benchmark_allowed(name, rows, default_split, manifest):
    manifest = Path(manifest)
    if not manifest.is_file():
        return
    policy = json.loads(manifest.read_text())
    if name.casefold() not in policy.get('future_eval_excluded_sources', []):
        return
    splits = []
    if rows is None:
        splits = [default_split]
    else:
        for row in rows:
            extra = row.get('extra_info') or row.get('metadata') or {}
            splits.append(row.get('original_split') or row.get('official_split') or row.get('split') or
                          extra.get('original_split') or extra.get('official_split') or extra.get('split'))
    if any(not split or str(split).casefold() == 'testmini' for split in splits):
        raise ValueError(f'{name}: testmini was authorized for local training; independent benchmarks require a different explicit split')
