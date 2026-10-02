"""Independent, native-LoRA session rollout using the shared vLLM engine.

Every role retains its actual sampled token IDs and log probabilities. Episode
branches are chosen by fixed sampling indices, never by reference answers.
"""
import copy
import os
import time
from pathlib import Path

from agent0_protocol.checkpointed import SessionResult, digest, text_of, tool_metrics
from agent0_protocol.schema import CanonicalTrajectory, RawRollout, ProtocolError
from agent0_protocol.tools import ToolExecutionContext, input_image_from_items
from tools.canonical_multimodal import item_images, load_image
from tools.training.canonical_rollout import TrajectoryState, ContextExhausted, schedule


class CheckpointedSessionState(TrajectoryState):
    def __init__(self, rollout, spec, settings, output_root, adapter_request, policy_version):
        self.rollout, self.spec, self.settings = rollout, spec, settings
        self.adapter_request, self.policy_version = adapter_request, policy_version
        self.trajectory = CanonicalTrajectory(spec.session_id, spec.tools,
            items=copy.deepcopy(spec.initial_items), metadata={
                'protocol_version': settings['protocol_version'], 'mode': spec.mode,
                'session_id': spec.session_id, 'loss_start_item_index': len(spec.initial_items)})
        self.tokens, self.mask, self.logprobs = [], [], []
        self.images = [load_image(v) for item in spec.initial_items for v in item_images(item)]
        self.initial_images = len(self.images)
        self.steps, self.repairs = [], 0
        self.final, self.last_solver, self.failure = None, '', ''
        self.cpu_tool_workers = settings['sampling']['cpu_tool_workers']
        self.synchronize_tool_context = True
        root = Path(output_root) / digest(spec.session_id)
        root.mkdir(parents=True, exist_ok=True)
        self.context = ToolExecutionContext({'sft_output_root': str(root),
            'sandbox_timeout': settings['limits']['tool_timeout_seconds'], 'checkpointed_protocol': True})
        image = spec.initial_image_path or input_image_from_items(spec.initial_items)
        if image is not None:
            path = root / 'input.png'
            temporary = root / f'.input.{os.getpid()}.png'
            load_image(image).save(temporary)
            os.replace(temporary, path)
            self.context.set_current_image(str(path))
        rendered = rollout.model_adapter.render(self.trajectory.items, spec.tools, generate=True)
        rendered = rendered.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        self.prompt = rollout.tokenizer.encode(rendered, add_special_tokens=False)
        self.expanded_prompt = self.expand(self.prompt, [False] * len(self.prompt), [None] * len(self.prompt))[0]
        self.result = SessionResult(spec, self.trajectory.to_dict())
        self.result.metrics['input_tokens'] = len(self.expanded_prompt)

    def capacity(self):
        limits = self.settings['limits']
        ids, _, _, _ = self.expand(self.prompt + self.tokens,
            [False] * len(self.prompt) + self.mask, [None] * len(self.prompt) + self.logprobs)
        return min(limits[f'{self.spec.mode}_response_tokens'] - len(ids) + len(self.expanded_prompt),
            limits['max_model_tokens'] - len(ids) - limits['context_margin_tokens'])

    def sample(self):
        endings = self.rollout.tokenizer.encode('<|im_end|>\n', add_special_tokens=False)
        remaining = self.capacity() - len(endings)
        if remaining <= 0:
            raise ContextExhausted('context_exhausted')
        output = yield ('generate', {'prompt_token_ids': self.prompt + self.tokens,
            **({'multi_modal_data': {'image': self.images}} if self.images else {})},
            min(remaining, self.settings['limits']['per_generation_tokens']), self.adapter_request)
        sample = output.outputs[0]
        ids = list(sample.token_ids)
        if not sample.logprobs or len(sample.logprobs) != len(ids):
            raise ProtocolError('vLLM sampled log-probabilities are missing')
        probabilities = []
        for token, entry in zip(ids, sample.logprobs):
            if token not in entry:
                raise ProtocolError('Selected vLLM token has no log-probability')
            probabilities.append(float(entry[token].logprob))
        self.append(ids, True, probabilities)
        items = self.rollout.model_adapter.decode_items(ids)
        for ordinal, item in enumerate(items):
            if item['type'] == 'function_call':
                item['call_id'] = f'call_{self.spec.session_id}_{len(self.tokens)}_{ordinal}'
            self.trajectory.append(item)
        end_id = self.rollout.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.append(endings if not ids or ids[-1] != end_id else
                    self.rollout.tokenizer.encode('\n', add_special_tokens=False))
        return items

    def run(self):
        limits = self.settings['limits']
        started, tool_rounds = time.monotonic(), 0
        segments = limits['max_verifier_tool_rounds'] + 1 if self.spec.mode == 'verify' else limits['max_solver_segments']
        try:
            prompt_limit = (limits['solve_prompt_tokens'] if self.spec.mode == 'solve' else
                limits['max_model_tokens'] - limits['context_margin_tokens'] -
                (limits['verify_response_tokens'] if self.spec.mode == 'verify' else 1))
            if len(self.expanded_prompt) > prompt_limit:
                raise ContextExhausted('input_context_exhausted')
            for segment in range(segments):
                begin, state = len(self.trajectory.items), self.context.checkpoint()
                state['owned_paths'] = sorted(state['owned_paths'])
                items = yield from self.sample()
                calls = [item for item in items if item['type'] == 'function_call']
                if calls:
                    if self.spec.mode == 'verify' and tool_rounds >= limits['max_verifier_tool_rounds']:
                        for call in calls:
                            self.trajectory.append({'type': 'function_call_output', 'call_id': call['call_id'],
                                'output': {'success': False, 'not_executed': True, 'error': 'verifier_tool_round_limit'}})
                        raise ProtocolError('verifier_tool_round_limit')
                    tool_rounds += 1
                    observations = yield ('tools', calls, self.context)
                    old_images = len(self.images)
                    self.images.extend(load_image(v) for item in observations for v in item_images(item))
                    try:
                        self.inject(observations)
                    except ContextExhausted:
                        del self.images[old_images:]
                        for observation in observations:
                            self.trajectory.append(observation)
                        raise
                    if any(o['output'].get('failure_kind') == 'infrastructure' for o in observations):
                        self.result.failure_kind, self.result.failure_reason = 'infrastructure', 'tool_infrastructure_failure'
                        return
                if self.spec.mode != 'verify':
                    self.steps.append({'step_id': f'S{self.spec.first_step_number + segment}',
                        'item_start': begin, 'item_end': len(self.trajectory.items), 'state_before': state})
                self.last_solver = '\n'.join(text_of(i) for i in items if i.get('role') == 'assistant')
                if not calls and (self.spec.mode == 'verify' or 'FINAL_ANSWER:' in self.last_solver or '\\boxed{' in self.last_solver):
                    self.trajectory.validate()
                    self.final = self.last_solver
                    return
                if time.monotonic() - started > limits['session_timeout_seconds']:
                    raise TimeoutError('session_timeout')
            raise ProtocolError('session_segment_limit')
        except TimeoutError:
            self.result.failure_kind, self.result.failure_reason = 'infrastructure', 'session_timeout'
        except ProtocolError as exc:
            self.result.failure_kind, self.result.failure_reason = 'model', str(exc)
        finally:
            ids, masks, probabilities, pixels = self.expand(self.prompt + self.tokens,
                [False] * len(self.prompt) + self.mask, [None] * len(self.prompt) + self.logprobs)
            start = len(self.expanded_prompt)
            raw = RawRollout(self.prompt, self.expanded_prompt, ids[start:], probabilities[start:],
                [True] * (len(ids) - start), [True] * len(ids), masks[start:],
                [t for t, selected in zip(ids[start:], masks[start:]) if selected],
                policy_version=str(self.policy_version), model_version=self.rollout.model_path)
            self.trajectory.rollout = raw.to_dict()
            state = self.context.checkpoint()
            state['owned_paths'] = sorted(state['owned_paths'])
            self.trajectory.metadata['image_state'] = state
            self.result.raw_rollout = raw.to_dict()
            self.result.steps = self.steps
            self.result.trajectory = self.trajectory.to_dict()
            self.result.metrics.update(tool_metrics(self.result))
            self.result.metrics.update(output_tokens=len(ids)-start, tool_rounds=tool_rounds,
                                      latency_seconds=time.monotonic()-started)
            # Pixel tensors stay out of JSON audit; they are passed to Actor/ref.
            self.pixel_inputs = pixels
            self.context.close()


def run_episodes(episodes, rollout, settings, adapters, output_root, group=None, rank=0):
    flows, pending, states_by_id, owners, results = {}, {}, {}, {}, {}
    versions = dict(adapters.versions)

    def create(index, specs):
        if dict(adapters.versions) != versions:
            raise RuntimeError('adapter_policy_changed_during_rollout')
        pending[index] = specs
        states = []
        for spec in specs:
            state = CheckpointedSessionState(rollout, spec, settings, output_root,
                adapters.request(spec.mode), versions[spec.mode])
            owners[spec.session_id] = index
            states_by_id[spec.session_id] = state
            states.append(state)
        return states

    def completed(state):
        session_id = state.spec.session_id
        results[session_id] = state.result
        index = owners[session_id]
        specs = pending[index]
        if not all(spec.session_id in results for spec in specs):
            return []
        try:
            following = flows[index].send([results[spec.session_id] for spec in specs])
        except StopIteration:
            del pending[index]
            return []
        return create(index, following)

    states = []
    for index, episode in enumerate(episodes):
        flows[index] = episode.run()
        states.extend(create(index, next(flows[index])))
    schedule(states, rollout.inference_engine.llm_engine, rollout.sampling_params,
        settings['sampling']['concurrency'], group, rank, on_complete=completed)
    if pending or dict(adapters.versions) != versions:
        raise RuntimeError('incomplete_or_changed_checkpointed_rollout')
    return states_by_id
