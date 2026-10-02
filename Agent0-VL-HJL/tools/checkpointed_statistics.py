"""Read-only teacher audit statistics. Suggestions never modify configuration."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tools.local_profile import load_config


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(flows, settings):
    samples, failures, transitions = defaultdict(list), Counter(), Counter()
    counts = Counter()
    for flow in flows:
        if flow.get('protocol_version') != settings['protocol_version']:
            raise ValueError('incompatible audit protocol')
        counts['problems'] += 1
        counts['accepted'] += bool(flow['accepted'])
        samples['repair_rounds'].append(max(0, len(flow['attempts']) - 1))
        for session in flow['sessions']:
            mode = session['spec']['mode']
            counts[mode] += 1
            if session['failure_kind']:
                failures[mode + ':' + session['failure_reason']] += 1
            for key, value in session['metrics'].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    samples[mode + ':' + key].append(value)
        for transition in flow['transitions']:
            if transition['before'] is not None and transition['after'] is not None:
                transitions[f"{transition['before']}->{transition['after']}"] += 1
    stats = settings['statistics']
    distributions = {key: {'count': len(values), 'max': max(values),
        'quantiles': {str(q): quantile(values, q) for q in stats['quantiles']}}
        for key, values in samples.items() if values}
    suggestions = {}
    for key, values in samples.items():
        if key.endswith(':output_tokens'):
            value = quantile(values, stats['coverage_target']) * (1 + stats['token_margin_ratio'])
            alignment = stats['token_alignment']
            suggestions[key.split(':')[0] + '_response_tokens'] = math.ceil(value / alignment) * alignment
    return {'counts': dict(counts), 'transitions': dict(transitions), 'failures': dict(failures),
            'distributions': distributions, 'suggested_limits': suggestions,
            'censoring_warning': 'Budget-exhausted samples are lower bounds; do not infer uncensored maxima.',
            'config_modified': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--profile', default='local_4090_checkpointed')
    args = parser.parse_args()
    settings = load_config(Path(__file__).resolve().parents[1], args.profile)['checkpointed']
    if args.audit.is_dir():
        flows = [json.loads(path.read_text()) for path in sorted(args.audit.rglob('*.json'))]
    else:
        with args.audit.open(encoding='utf-8') as stream:
            flows = [json.loads(line) for line in stream if line.strip()]
        flows = [row.get('trajectory', {}).get('metadata', {}).get('checkpointed_flow', row)
                 for row in flows]
    print(json.dumps(summarize(flows, settings), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
