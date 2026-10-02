"""Independent Responses sessions for checkpointed verification and repair.

The outcome scorer is a separate callback, never part of the model request.
Token counting must use the deployed model processor, including image tokens.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError

from .checkpointed import Episode, SessionResult, digest, text_of, tool_metrics
from .schema import CanonicalTrajectory, ProtocolError
from .tools import ToolExecutionContext, execute_call_batch, input_image_from_items


def response_history(items, adapter):
    history = []
    for item in items:
        value = copy.deepcopy(item)
        raw = value.pop('_responses_item', None)
        if value['type'] == 'function_call_output':
            value = adapter.function_result(value)
        elif raw is not None:
            value = raw
        elif value['type'] == 'function_call':
            value['arguments'] = json.dumps(value['arguments'], ensure_ascii=False)
        elif value['type'] == 'reasoning' and not value.get('id'):
            raise ProtocolError('responses_reasoning_id_missing')
        history.append(value)
    return history


class CheckpointedResponsesRunner:
    def __init__(self, runtime, settings, output_root, count_tokens):
        self.runtime, self.settings = runtime, settings
        self.output_root = Path(output_root)
        self.count_tokens = count_tokens
        if count_tokens is None:
            raise ValueError('A multimodal token counter is required; truncation is forbidden')
        if runtime.config.timeout_seconds != settings['limits']['request_timeout_seconds']:
            raise ValueError('Responses request timeout must match checkpointed config')

    def run_session(self, spec):
        limits = self.settings['limits']
        root = self.output_root / digest(spec.session_id)
        root.mkdir(parents=True, exist_ok=False)
        trajectory = CanonicalTrajectory(spec.session_id, spec.tools,
            items=copy.deepcopy(spec.initial_items), metadata={
                'protocol_version': self.settings['protocol_version'],
                'role': 'Verifier-Repair' if spec.mode == 'verify' else 'Solver',
                'mode': spec.mode, 'session_id': spec.session_id,
                'group_id': spec.group_id, 'parent_id': spec.parent_id,
                'loss_start_item_index': len(spec.initial_items)})
        result = SessionResult(spec, trajectory.to_dict())
        context = ToolExecutionContext({'sft_output_root': str(root),
            'sandbox_timeout': limits['tool_timeout_seconds'], 'checkpointed_protocol': True})
        image = spec.initial_image_path or input_image_from_items(spec.initial_items)
        # Materialize a private copy, including Repair's restored image state.
        if image is not None:
            from tools.canonical_multimodal import load_image
            private_image = root / 'input.png'
            load_image(image).save(private_image)
            context.set_current_image(str(private_image))
        started = time.monotonic()
        tool_rounds = 0
        try:
            initial_tokens = self.count_tokens(trajectory)
            result.metrics['input_tokens'] = initial_tokens
            segments = (limits['max_verifier_tool_rounds'] + 1 if spec.mode == 'verify'
                        else limits['max_solver_segments'])
            for segment in range(segments):
                if time.monotonic() - started > limits['session_timeout_seconds']:
                    raise TimeoutError('session_timeout')
                current_tokens = self.count_tokens(trajectory)
                available = limits['max_model_tokens'] - current_tokens - limits['context_margin_tokens']
                if available <= 0:
                    raise ProtocolError('context_exhausted')
                if current_tokens - initial_tokens >= limits[f'{spec.mode}_response_tokens']:
                    raise ProtocolError('response_budget_exhausted')
                state = context.checkpoint()
                state['owned_paths'] = sorted(state['owned_paths'])
                begin = len(trajectory.items)
                at_tool_limit = spec.mode == 'verify' and tool_rounds >= limits['max_verifier_tool_rounds']
                response = self.runtime._create(
                    input=response_history(trajectory.items, self.runtime.adapter), tools=spec.tools,
                    request_budget_seconds=min(limits['request_timeout_seconds'],
                        limits['session_timeout_seconds'] - (time.monotonic() - started)),
                    tool_choice='none' if at_tool_limit else 'auto',
                    max_output_tokens=min(available, limits['teacher_reply_tokens'])
                        if limits['teacher_reply_tokens'] else 0)
                raw_output = [
                    value.model_dump(mode='json') if hasattr(value, 'model_dump') else copy.deepcopy(value)
                    for value in response.output]
                trajectory.metadata.setdefault('raw_response_outputs', []).append(raw_output)
                items = self.runtime.adapter.output_items(response)
                for item, raw in zip(items, raw_output):
                    item['_responses_item'] = copy.deepcopy(raw)
                if not items:
                    raise ProtocolError('empty_response')
                for item in items:
                    trajectory.append(item)
                calls = [item for item in items if item['type'] == 'function_call']
                if calls:
                    if at_tool_limit:
                        raise ProtocolError('verifier_tool_round_limit')
                    tool_rounds += 1
                    observations = execute_call_batch(self.runtime.registry, calls, context)
                    for observation in observations:
                        trajectory.append(observation)
                    if any(o['output'].get('failure_kind') == 'infrastructure' for o in observations):
                        result.failure_kind, result.failure_reason = 'infrastructure', 'tool_infrastructure_failure'
                        return result
                if spec.mode != 'verify':
                    result.steps.append({'step_id': f'S{spec.first_step_number + segment}',
                        'item_start': begin, 'item_end': len(trajectory.items),
                        'state_before': state})
                final_tokens = self.count_tokens(trajectory)
                result.metrics.update(output_tokens=final_tokens-initial_tokens,
                                      tool_rounds=tool_rounds)
                if time.monotonic() - started > limits['session_timeout_seconds']:
                    raise TimeoutError('session_timeout')
                if final_tokens + limits['context_margin_tokens'] > limits['max_model_tokens']:
                    raise ProtocolError('context_exhausted')
                if final_tokens-initial_tokens > limits[f'{spec.mode}_response_tokens']:
                    raise ProtocolError('response_budget_exhausted')
                # Tool execution happens even if the same response has an answer.
                texts = '\n'.join(text_of(i) for i in items if i.get('role') == 'assistant')
                if not calls and (spec.mode == 'verify' or 'FINAL_ANSWER:' in texts or '\\boxed{' in texts):
                    trajectory.validate()
                    return result
                if segment >= limits['max_solver_segments'] - 1 and spec.mode != 'verify':
                    raise ProtocolError('solver_segment_limit')
            raise ProtocolError('session_segment_limit')
        except (TimeoutError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            result.failure_kind, result.failure_reason = 'infrastructure', type(exc).__name__
        except ProtocolError as exc:
            result.failure_kind, result.failure_reason = 'model', str(exc)
        finally:
            state = context.checkpoint()
            state['owned_paths'] = sorted(state['owned_paths'])
            trajectory.metadata['image_state'] = state
            result.trajectory = trajectory.to_dict()
            result.metrics.update(tool_metrics(result))
            result.metrics['latency_seconds'] = time.monotonic() - started
            context.close()
        return result

    def run_episode(self, problem_id, initial_items, scorer, teacher=True):
        episode = Episode(problem_id, initial_items, self.runtime.registry.definitions(),
                          self.settings, scorer, teacher=teacher)
        flow = episode.run()
        try:
            specs = next(flow)
            while True:
                specs = flow.send([self.run_session(spec) for spec in specs])
        except StopIteration:
            return episode
