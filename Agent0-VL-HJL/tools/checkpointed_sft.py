"""Export mode-specific positive SFT rows from accepted checkpointed audits."""
import argparse
import hashlib
import json
from pathlib import Path

from agent0_protocol.checkpointed import PROTOCOL, MODES, ADAPTER_LAYOUT, ADAPTER_FOR_MODE
from agent0_protocol.checkpointed_training import sft_rows
from tools.data_builder.checkpointed_quality import audit_flow
from tools.data_builder.sft_stream import _write_state


def export(data, output):
    data, output = Path(data), Path(output)
    manifest = json.loads(data.with_suffix('.manifest.json').read_text())
    if manifest.get('protocol') != PROTOCOL or manifest['sha256'] != hashlib.sha256(data.read_bytes()).hexdigest():
        raise ValueError('Input is not a fingerprinted checkpointed dataset')
    mode_rows = {mode: [] for mode in MODES}
    for line in data.read_text().splitlines():
        row = json.loads(line)
        flow = row['trajectory']['metadata']['checkpointed_flow']
        audit_flow(flow)
        for mode, trajectories in sft_rows(flow).items():
            mode_rows[mode].extend({'trajectory': trajectory, 'data_source': row['data_source'],
                'problem_id': flow['problem_id']} for trajectory in trajectories)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for mode, rows in mode_rows.items():
        path = output / (mode + '.jsonl')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        temporary.replace(path)
        counts[mode] = len(rows)
        _write_state(path.with_suffix('.manifest.json'), {'protocol': PROTOCOL,
            'mode': mode, 'rows': len(rows), 'accepted_problems': manifest['rows'],
            'source_sha256': manifest['sha256'],
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    shared = [row for mode in MODES for row in mode_rows[mode]]
    shared_path = output / 'shared.jsonl'
    temporary = shared_path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in shared))
    temporary.replace(shared_path)
    _write_state(shared_path.with_suffix('.manifest.json'), {
        'protocol': PROTOCOL, 'adapter_layout_version': ADAPTER_LAYOUT,
        'adapter': 'shared', 'mode_to_adapter': ADAPTER_FOR_MODE, 'modes': list(MODES),
        'rows': len(shared), 'rows_by_mode': counts, 'source_sha256': manifest['sha256'],
        'mode_sha256': {mode: hashlib.sha256((output / (mode + '.jsonl')).read_bytes()).hexdigest()
                        for mode in MODES},
        'sha256': hashlib.sha256(shared_path.read_bytes()).hexdigest()})
    _write_state(output / 'roles.json', {'protocol': PROTOCOL, 'rows_by_mode': counts,
        'source': str(data), 'source_sha256': manifest['sha256']})
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.data, args.output), indent=2))


if __name__ == '__main__':
    main()
