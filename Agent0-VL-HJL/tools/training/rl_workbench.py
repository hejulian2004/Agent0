"""Console summaries and full Solver outputs, written by the trainer driver."""

import json
from pathlib import Path


class RLWorkbench:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / 'workbench.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        inventory = self.path.parent / 'README.md'
        note = '\n- Workbench: workbench.jsonl; per-step and cumulative trajectory correctness, real tool-call success including repairs, last5 versus prior5 trend; full last Solver responses retained. No-call rates are N/A. Console previews are truncated.\n'
        if inventory.is_file() and 'Workbench: workbench.jsonl' not in inventory.read_text():
            with inventory.open('a') as stream:
                stream.write(note)
        self.previous = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                try:
                    record = json.loads(line)
                    self.previous[int(record['step'])] = record['summary']
                except (ValueError, KeyError, TypeError):
                    continue

    def log(self, step, batch, extras, phase):
        values = batch.non_tensor_batch
        count = len(batch.batch['responses'])
        accuracy = [float(x) for x in extras.get('acc', [])]
        calls = sum(int(x) for x in values.get('tool_call_count', []))
        successes = sum(int(x) for x in values.get('successful_tool_call_count', []))
        per_tool = {}
        for trajectory in values.get('canonical_trajectory', []):
            executed = {item['call_id'] for item in trajectory['items'] if item['type'] == 'function_call_output' and not item['output'].get('not_executed', False)}
            names = {item['call_id']: item['name'] for item in trajectory['items'] if item['type'] == 'function_call' and item['call_id'] in executed}
            for name in names.values():
                per_tool.setdefault(name, {'calls': 0, 'successful': 0})['calls'] += 1
            for item in trajectory['items']:
                if item['type'] == 'function_call_output' and item['output'].get('success') is True:
                    per_tool[names[item['call_id']]]['successful'] += 1
        summary = {'per_tool': per_tool, 'phase': phase, 'trajectories': count,
                   'answer_count': len(accuracy), 'correct': sum(accuracy),
                   'tool_calls': calls, 'successful_tool_calls': successes,
                   'failed': sum(bool(x) for x in values.get('trajectory_failed', []))}
        # A resumed step replaces its old contribution instead of double counting.
        self.previous[step] = summary
        totals = {k: sum(r.get(k, 0) for s, r in self.previous.items() if s <= step)
                  for k in ('answer_count', 'correct', 'tool_calls', 'successful_tool_calls')}
        rate = lambda n, d: None if not d else n / d
        summary.update(answer_accuracy=rate(sum(accuracy), len(accuracy)),
                       tool_success_rate=rate(successes, calls),
                       overall_answer_accuracy=rate(totals['correct'], totals['answer_count']),
                       overall_tool_success_rate=rate(totals['successful_tool_calls'], totals['tool_calls']))
        history = [r for s, r in sorted(self.previous.items()) if s <= step and r['phase'] == phase]
        def window_rate(records, numerator, denominator):
            return rate(sum(r.get(numerator, 0) for r in records),
                        sum(r.get(denominator, 0) for r in records))
        for label, numerator, denominator in (
            ('answer_accuracy', 'correct', 'answer_count'),
            ('tool_success_rate', 'successful_tool_calls', 'tool_calls'),
        ):
            recent = window_rate(history[-5:], numerator, denominator)
            prior = window_rate(history[-10:-5], numerator, denominator)
            summary['last5_' + label] = recent
            summary['trend_' + label] = (recent - prior if len(history) >= 10
                                        and recent is not None and prior is not None else None)
        summary['per_tool_cumulative'] = {}
        for record_step, record in self.previous.items():
            if record_step <= step:
                for name, counts in record.get('per_tool', {}).items():
                    total = summary['per_tool_cumulative'].setdefault(name, {'calls': 0, 'successful': 0})
                    for key in total:
                        total[key] += counts[key]
        samples = []
        for i in range(count):
            def value(key, default=''):
                entries = values.get(key)
                return default if entries is None else entries[i]
            reward = value('reward_model', {})
            samples.append({'index': i, 'source': str(value('data_source')),
                            'reference': str(reward.get('ground_truth', '')),
                            'final_answer': str(value('final_answers')),
                            'accuracy': accuracy[i] if i < len(accuracy) else None,
                            'failure_reason': str(value('failure_reason')),
                            'last_solver_response': str(value('last_solver_response'))})
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'step': step, 'summary': summary, 'samples': samples}, ensure_ascii=False) + '\n')
        fmt = lambda x: 'N/A' if x is None else f'{x:.2%}'
        print(f"[RL workbench] phase={phase} step={step} trajectories={count} "
              f"answer_accuracy={fmt(summary['answer_accuracy'])} "
              f"tool_success={fmt(summary['tool_success_rate'])} ({successes}/{calls}) "
              f"overall_answer_accuracy={fmt(summary['overall_answer_accuracy'])} "
              f"overall_tool_success={fmt(summary['overall_tool_success_rate'])} "
              f"failed={summary['failed']}", flush=True)
        trend = lambda x: '待满10步' if x is None else f'{x * 100:+.2f}个百分点'
        print(f"[RL trend] last{min(5, len(history))} steps: "
              f"answer_accuracy={fmt(summary['last5_answer_accuracy'])} "
              f"({trend(summary['trend_answer_accuracy'])}); "
              f"tool_success={fmt(summary['last5_tool_success_rate'])} "
              f"({trend(summary['trend_tool_success_rate'])})", flush=True)
        for sample in samples[:1]:
            print(f"[RL Solver] source={sample['source']} final={sample['final_answer']} "
                  f"reference={sample['reference']} accuracy={sample['accuracy']}\n"
                  f"{sample['last_solver_response'][:500]}", flush=True)
