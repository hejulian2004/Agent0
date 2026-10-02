"""Streaming local SERC with genuine vLLM tokens and multimodal observations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import uuid
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tensordict import TensorDict
from vllm import SamplingParams
from vllm.distributed import parallel_state

from verl import DataProto
from agent0_protocol.adapters import ResponsesAdapter
from agent0_protocol.schema import CanonicalTrajectory, RawRollout, ProtocolError
from agent0_protocol.tools import ToolExecutionContext, execute_call_batch
from agent0_protocol.local_prompts import render_verifier_request, render_repair_request
from tools.data_builder.sft_quality import StrictAnswerJudge, parse_verifier, item_text, _segment_snapshot
from tools.canonical_multimodal import item_images, load_image


class ContextExhausted(ProtocolError):
    pass


def object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


class TrajectoryState:
    def __init__(self, rollout, trajectory, prompt_ids, images, output_root, input_paths=None):
        self.rollout = rollout
        self.trajectory = trajectory
        self.prompt = list(prompt_ids)
        self.tokens, self.mask, self.logprobs = [], [], []
        self.images = [load_image(image) for image in images]
        self.initial_images = len(self.images)
        self.steps, self.repairs = [], 0
        self.final, self.last_solver, self.failure = None, "", ""
        self.context = ToolExecutionContext({"sandbox_timeout": rollout.sandbox_timeout,
                                             "sft_output_root": str(output_root)})
        output_root.mkdir(parents=True, exist_ok=True)
        self.input_paths = input_paths or []
        if self.images:
            self.context.set_current_image(self.input_paths[0] if self.input_paths else self.images[0])
        guidance = "\nInput image paths: " + json.dumps(self.input_paths) + "\nProject output root: " + str(output_root.resolve()) + ". Python receives image_path and output_dir; save generated images under output_dir for visual feedback."
        user = next(item for item in reversed(self.trajectory.items) if item.get('role') == 'user')
        if isinstance(user['content'], str):
            user['content'] += guidance
        else:
            user['content'].append({'type': 'input_text', 'text': guidance})
        rendered = rollout.model_adapter.render(self.trajectory.items, self.trajectory.tools, generate=True)
        rendered = rendered.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        self.prompt = rollout.tokenizer.encode(rendered, add_special_tokens=False)
        self.expanded_prompt = self.expand(self.prompt, [False] * len(self.prompt), [None] * len(self.prompt))[0]

    def expand(self, ids, mask, logprobs):
        if not self.images:
            return list(ids), list(mask), list(logprobs), {}
        pixels = self.rollout.processor.image_processor(images=self.images, return_tensors="pt")
        counts = [int(grid.prod()) // self.rollout.processor.image_processor.merge_size ** 2
                  for grid in pixels["image_grid_thw"]]
        image_token = self.rollout.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        output, selected, probs, cursor = [], [], [], 0
        for token, sampled, probability in zip(ids, mask, logprobs):
            count = 1
            if token == image_token and not sampled:
                if cursor >= len(counts):
                    raise ProtocolError("Too many image markers")
                count, cursor = counts[cursor], cursor + 1
            output.extend([token] * count)
            selected.extend([sampled] * count)
            probs.extend([probability] * count)
        if cursor != len(counts):
            raise ProtocolError("Image markers do not cover image history")
        return output, selected, probs, dict(pixels)

    def capacity(self):
        all_ids, _, _, _ = self.expand(self.prompt + self.tokens,
            [False] * len(self.prompt) + self.mask, [None] * len(self.prompt) + self.logprobs)
        used = len(all_ids) - len(self.expanded_prompt)
        return min(self.rollout.max_total_length - used,
                   int(self.rollout.config.max_model_len) - len(all_ids) - 8)

    def append(self, tokens, sampled=False, probabilities=None):
        self.tokens.extend(tokens)
        self.mask.extend([sampled] * len(tokens))
        self.logprobs.extend(probabilities if sampled else [None] * len(tokens))
        if self.capacity() < 0:
            # Never truncate generated actions. Keep a valid capacity-bounded
            # failed trajectory with this whole segment absent.
            del self.tokens[-len(tokens):]
            del self.mask[-len(tokens):]
            del self.logprobs[-len(tokens):]
            raise ContextExhausted("context_exhausted")

    def inject(self, items):
        text = self.rollout.model_adapter.render(items, [], generate=True)
        text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        self.append(self.rollout.tokenizer.encode(text, add_special_tokens=False))
        candidate = CanonicalTrajectory(self.trajectory.trajectory_id, self.trajectory.tools,
            items=[*self.trajectory.items, *items])
        try:
            candidate.validate(complete=False)
        except Exception as exc:
            from jsonschema import ValidationError
            if isinstance(exc, (ProtocolError, ValidationError)):
                raise ProtocolError('invalid_solver_items: ' + type(exc).__name__) from exc
            raise
        for item in items:
            self.trajectory.append(item)

    def sample(self):
        remaining = self.capacity() - len(self.rollout.tokenizer.encode("<|im_end|>\n", add_special_tokens=False))
        if remaining <= 0:
            raise ContextExhausted("context_exhausted")
        output = yield ("generate", {"prompt_token_ids": self.prompt + self.tokens,
            **({"multi_modal_data": {"image": self.images}} if self.images else {})},
            min(int(self.rollout.config.response_length), remaining))
        result = output.outputs[0]
        ids = list(result.token_ids)
        if not result.logprobs or len(result.logprobs) != len(ids):
            raise ProtocolError("vLLM sampled log-probabilities are missing")
        probabilities = []
        for token, entry in zip(ids, result.logprobs):
            if token not in entry:
                raise ProtocolError("Selected vLLM token has no log-probability")
            probabilities.append(float(entry[token].logprob))
        self.append(ids, True, probabilities)
        items = self.rollout.model_adapter.decode_items(ids)
        for ordinal, item in enumerate(items):
            if item['type'] == 'function_call':
                item['call_id'] = f'call_{self.trajectory.trajectory_id}_{len(self.tokens)}_{ordinal}'
        for item in items:
            self.trajectory.append(item)
        endings = self.rollout.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        end_id = self.rollout.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not ids or ids[-1] != end_id:
            self.append(endings)
        else:
            self.append(self.rollout.tokenizer.encode("\n", add_special_tokens=False))
        return items

    def role(self, prompt):
        self.inject([{"type": "message", "role": "user", "content": prompt}])
        items = yield from self.sample()
        if any(item["type"] == "function_call" for item in items):
            raise ProtocolError("Verifier/Repair emitted a function call")
        messages = [item for item in items if item['type'] == 'message']
        if len(messages) != 1 or not item_text(messages[0]).strip():
            raise ProtocolError('role_response_requires_one_message')
        return item_text(messages[0])

    def run(self):
        try:
            for step in range(self.rollout.max_reasoning_steps):
                checkpoint = self.context.checkpoint()
                for round_index in range(int(self.rollout.config.max_repairs_per_step) + 1):
                    start = len(self.trajectory.items)
                    items = yield from self.sample()
                    self.last_solver = "\n".join(item_text(item) for item in items if item["type"] == "message")
                    calls = [item for item in items if item["type"] == "function_call"]
                    failed = False
                    if calls:
                        outputs = yield ("tools", calls, self.context)
                        old_image_count = len(self.images)
                        for output in outputs:
                            self.images.extend(load_image(value) for value in item_images(output))
                        failed = any(output["output"].get("success") is not True for output in outputs)
                        try:
                            self.inject(outputs)
                        except ContextExhausted:
                            del self.images[old_image_count:]
                            for output in outputs:
                                self.trajectory.append(output)
                            raise
                    snapshot = _segment_snapshot(self.trajectory, start, len(self.trajectory.items), step)
                    verification_text = yield from self.role(render_verifier_request(snapshot))
                    verification = parse_verifier(verification_text)
                    if not failed and verification["confidence"] >= self.rollout.repair_threshold:
                        break
                    if round_index >= int(self.rollout.config.max_repairs_per_step):
                        raise ProtocolError("repair_exhausted")
                    repair_text = yield from self.role(render_repair_request(snapshot, verification) +
                        '\nReturn JSON {"action":"PATCH" or "NO_CHANGE", "patch":"instruction"}.')
                    from agent0_protocol.verifier import extract_json_dict
                    repair = extract_json_dict(repair_text)
                    if not isinstance(repair, dict) or repair.get("action") not in {"PATCH", "NO_CHANGE"} or not isinstance(repair.get("patch", ""), str):
                        raise ProtocolError("invalid_repair")
                    self.repairs += 1
                    if repair["action"] == "PATCH":
                        self.context.rollback(checkpoint)
                    self.inject([{"type": "message", "role": "user", "content":
                        "Regenerate the preceding Solver segment. " + repair.get("patch", "")}])
                self.steps.append({"step_index": step + 1, **verification, "verified": True,
                    "tool_used": bool(calls), "tool_success": bool(calls) and not failed,
                    "was_repaired": round_index > 0, "step_end_pos": len(self.tokens)})
                messages = [index for index in range(start, start + len(items))
                            if self.trajectory.items[index]["type"] == "message"]
                if messages and (not calls or StrictAnswerJudge.extract_answer(self.last_solver) is not None):
                    self.trajectory.metadata["final_solver_index"] = messages[-1]
                    self.final = self.last_solver
                    break
                self.inject([{"type": "message", "role": "user", "content": "Continue solving the task."}])
            if self.final is None:
                raise ProtocolError("solver_steps_exhausted")
        except ProtocolError as exc:
            self.failure = str(exc)
            self.final = None
            # Complete canonical tool pairs even for malformed role outputs.
            pending = {item["call_id"] for item in self.trajectory.items if item["type"] == "function_call"}
            pending -= {item["call_id"] for item in self.trajectory.items if item["type"] == "function_call_output"}
            for call_id in pending:
                self.trajectory.append({"type": "function_call_output", "call_id": call_id,
                    "output": {"success": False, "error": self.failure, "not_executed": True}})
        finally:
            self.context.close()
            for path in self.input_paths:
                Path(path).unlink(missing_ok=True)
        self.trajectory.metadata.update(failure_reason=self.failure, trajectory_failed=bool(self.failure),
                                       final_answer=self.final, num_repairs=self.repairs)


def schedule(states, engine, sampling_params, concurrency, group=None, rank=0, on_complete=None):
    waiting = deque(enumerate(states))
    active, model_jobs, tool_jobs = {}, {}, {}
    workers = getattr(states[0], 'cpu_tool_workers', concurrency) if states else concurrency
    pool = ThreadPoolExecutor(max_workers=workers) if rank == 0 else None
    counter = 0
    started = last_report = time.monotonic()

    def tools(calls, context):
        checkpoint = context.checkpoint()
        outputs = execute_call_batch(states[0].rollout.registry, calls, context=context)
        # Validate with the exact Actor processor before success accounting.
        for output in outputs:
            try:
                urls = item_images(output)
                images = [load_image(value) for value in urls]
                if images:
                    states[0].rollout.processor.image_processor(images=images, return_tensors="pt")
                    output['output']['image_urls'] = urls
            except (ValueError, OSError) as exc:
                context.rollback(checkpoint)
                result = output["output"]
                result.update(success=False, error="invalid_image_observation: " + type(exc).__name__)
                for name in ("image_url", "image_urls", "image_data", "images", "image_path", "output_path"):
                    result.pop(name, None)
        return outputs

    def advance(index, result=None, initial=False):
        nonlocal counter
        try:
            job = next(active[index]) if initial else active[index].send(result)
        except StopIteration:
            del active[index]
            if on_complete is not None:
                for state in on_complete(states[index]):
                    next_index = len(states)
                    states.append(state)
                    waiting.append((next_index, state))
            return
        if job[0] == "generate":
            key = f"hjl-{index}-{counter}"
            counter += 1
            params = copy.deepcopy(sampling_params)
            params.n, params.max_tokens, params.logprobs = 1, job[2], 1
            model_jobs[key] = index
            engine.add_request(key, job[1], params,
                **({'lora_request': job[3]} if len(job) > 3 else {}))
        else:
            tool_jobs[index] = pool.submit(tools, *job[1:]) if pool else None

    def refill():
        while waiting and len(active) < concurrency:
            index, state = waiting.popleft()
            active[index] = state.run()
            advance(index, initial=True)

    try:
        refill()
        while active:
            events = []
            if model_jobs:
                results = engine.step()
                if rank == 0:
                    events.extend(("generate", result.request_id, result) for result in results if result.finished)
            if rank == 0:
                for index, future in tool_jobs.items():
                    if future.done():
                        value = future.result()
                        if getattr(states[index], 'synchronize_tool_context', False):
                            value = {'outputs': value, 'context_state': states[index].context.checkpoint()}
                        events.append(('tools', index, value))
            if group is not None:
                payload = [events if rank == 0 else None]
                torch.distributed.broadcast_object_list(payload, src=torch.distributed.get_global_rank(group, 0), group=group)
                events = payload[0]
            for kind, key, result in events:
                if kind == "generate":
                    index = model_jobs.pop(key)
                    engine.abort_request([key])
                else:
                    index = key
                    del tool_jobs[key]
                    if isinstance(result, dict) and 'context_state' in result:
                        states[index].context.synchronize(result['context_state'])
                        result = result['outputs']
                advance(index, result)
                refill()
            now = time.monotonic()
            if rank == 0 and now - last_report >= 30:
                print(f'[RL streaming] elapsed={now-started:.0f}s completed={len(states)-len(active)-len(waiting)}/{len(states)} '
                      f'active={len(active)} generation={len(model_jobs)} tools={len(tool_jobs)} waiting={len(waiting)}', flush=True)
                last_report = now
            if not events and not model_jobs:
                time.sleep(.01)
        if rank == 0:
            print(f'[RL streaming] completed={len(states)}/{len(states)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    finally:
        if model_jobs:
            engine.abort_request(list(model_jobs))
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)


def generate_sequences(rollout, prompts, **kwargs):
    if rollout.processor is None:
        raise ValueError("Local multimodal rollout requires the real model processor")
    n = int(rollout.config.n) if prompts.meta_info.get("do_sample", True) and not prompts.meta_info.get("validate", False) else 1
    idx = prompts.batch["input_ids"].repeat_interleave(n, dim=0)
    values = {k: np.repeat(v, n, axis=0) for k, v in prompts.non_tensor_batch.items()}
    tp = parallel_state.get_tp_group()
    payload = [uuid.uuid4().hex if tp.rank_in_group == 0 else None]
    torch.distributed.broadcast_object_list(payload, src=torch.distributed.get_global_rank(tp.cpu_group, 0), group=tp.cpu_group)
    root = Path("outputs/rl_tool_images") / payload[0]
    paths = []
    if tp.rank_in_group == 0:
        for index in range(len(idx)):
            directory = root / str(index)
            directory.mkdir(parents=True, exist_ok=True)
            input_paths = []
            for image_index, image in enumerate(values.get('multi_modal_data', [{}] * len(idx))[index].get('image', [])):
                path = directory / f'input_scratch_{image_index}.png'
                load_image(image).save(path)
                input_paths.append(str(path.resolve()))
            paths.append(input_paths)
    payload_paths = [paths if tp.rank_in_group == 0 else None]
    torch.distributed.broadcast_object_list(payload_paths, src=torch.distributed.get_global_rank(tp.cpu_group, 0), group=tp.cpu_group)
    paths = payload_paths[0]
    states = []
    for i in range(len(idx)):
        trajectory = CanonicalTrajectory.from_dict(json.loads(values['canonical_trajectory_json'][i]))
        trajectory.trajectory_id = f'rl_{root.name}_{i}'
        states.append(TrajectoryState(rollout, trajectory, values['raw_prompt_ids'][i],
            values.get('multi_modal_data', [{}] * len(idx))[i].get('image', []), root / str(i), paths[i]))
    prompt_width = int(rollout.config.prompt_length)
    idx = torch.full((len(states), prompt_width), rollout.pad_token_id, dtype=idx.dtype, device=idx.device)
    prompt_attention = torch.zeros_like(idx)
    for index, state in enumerate(states):
        if len(state.expanded_prompt) > prompt_width:
            raise ProtocolError('Runtime-guided prompt exceeds max_prompt_length')
        idx[index, -len(state.expanded_prompt):] = torch.tensor(state.expanded_prompt, device=idx.device)
        prompt_attention[index, -len(state.expanded_prompt):] = 1
    params = copy.deepcopy(rollout.sampling_params)
    if not prompts.meta_info.get("do_sample", True):
        params.temperature = 0
    schedule(states, rollout.inference_engine.llm_engine, params, int(rollout.config.max_num_seqs),
             tp.cpu_group, tp.rank_in_group)
    expanded = [state.expand(state.prompt + state.tokens,
        [False] * len(state.prompt) + state.mask, [None] * len(state.prompt) + state.logprobs) for state in states]
    lengths = [len(ids) - len(state.expanded_prompt) for state, (ids, _, _, _) in zip(states, expanded)]
    width = max(max(lengths, default=1), 1)
    responses = torch.full((len(states), width), rollout.pad_token_id, dtype=idx.dtype, device=idx.device)
    sampled = torch.zeros_like(responses, dtype=torch.bool)
    response_attention = torch.zeros_like(responses, dtype=torch.bool)
    old_probs = torch.zeros_like(responses, dtype=torch.float32)
    position_rows, pixels, semantics, raw_rows = [], [], [], []
    attention = torch.cat([prompt_attention, response_attention], dim=-1)
    from verl.models.transformers.qwen2_vl import get_rope_index
    for i, (state, (ids, masks, probs, image_inputs)) in enumerate(zip(states, expanded)):
        prompt_length = len(state.expanded_prompt)
        size = lengths[i]
        responses[i, :size] = torch.tensor(ids[prompt_length:], device=idx.device)
        sampled[i, :size] = torch.tensor(masks[prompt_length:], device=idx.device)
        old_probs[i, :size] = torch.tensor([p or 0.0 for p in probs[prompt_length:]], device=idx.device)
        attention[i, idx.shape[1]:idx.shape[1] + size] = 1
        response_attention[i, :size] = True
        seq = torch.cat([idx[i], responses[i]])
        position_rows.append(get_rope_index(rollout.processor, seq, image_grid_thw=image_inputs.get("image_grid_thw"),
                                            attention_mask=attention[i]))
        pixels.append(image_inputs)
        image_pad = rollout.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        counts = [int(grid.prod()) // rollout.processor.image_processor.merge_size ** 2 for grid in image_inputs.get('image_grid_thw', [])]
        cursor, accumulated, offsets = state.initial_images, 0, [0]
        for token, selected in zip(state.tokens, state.mask):
            count = 1
            if token == image_pad and not selected:
                count = counts[cursor]
                cursor += 1
            accumulated += count
            offsets.append(accumulated)
        for step in state.steps:
            step['step_end_pos'] = offsets[step['step_end_pos']]
        raw = RawRollout(prompt_token_ids=state.prompt, expanded_prompt_token_ids=idx[i].tolist(),
            response_token_ids=responses[i].tolist(), old_logprobs=[float(p) if m else None for p, m in zip(old_probs[i], sampled[i])],
            response_mask=response_attention[i].tolist(), attention_mask=attention[i].bool().tolist(),
            sampling_mask=sampled[i].tolist(), sampled_token_ids=responses[i][sampled[i]].tolist(),
            sampling_metadata={"temperature": params.temperature, "top_p": params.top_p},
            policy_version=str(prompts.meta_info.get("policy_version", rollout.model_path)), model_version=rollout.model_path)
        state.trajectory.rollout = raw.to_dict()
        semantics.append(state.trajectory.to_dict())
        raw_rows.append(raw.to_dict())
    batch = TensorDict({"prompts": idx, "responses": responses, "input_ids": torch.cat([idx, responses], dim=-1),
        "attention_mask": attention, "position_ids": torch.stack(position_rows), "multiturn_mask": sampled,
        "old_log_probs": old_probs}, batch_size=len(states))
    output = {k: v for k, v in values.items() if k not in {"raw_prompt_ids", "multi_modal_data"}}
    for key, val in {"canonical_trajectory": semantics, "raw_rollout": raw_rows, "multi_modal_inputs": pixels,
        "step_data": [s.steps for s in states],
        "num_steps": [len(s.steps) for s in states], "num_repairs": [s.repairs for s in states],
        "final_answers": [s.final for s in states], "trajectory_failed": [bool(s.failure) for s in states],
        "failure_reason": [s.failure for s in states], "last_solver_response": [s.last_solver for s in states],
        "tool_call_count": [sum(x["type"] == "function_call_output" and not x["output"].get("not_executed", False) for x in s.trajectory.items) for s in states],
        "successful_tool_call_count": [sum(x["type"] == "function_call_output" and x["output"].get("success") is True for x in s.trajectory.items) for s in states]}.items():
        output[key] = object_array(val)
    verify_width = max(max((len(s.steps) for s in states), default=0), 1)
    output['verify_probs'] = np.zeros((len(states), verify_width), dtype=np.float32)
    for index, state in enumerate(states):
        output['verify_probs'][index, :len(state.steps)] = [step['confidence'] for step in state.steps]
    output['final_generation_step'] = np.array([max(len(s.steps)-1, 0) for s in states], dtype=np.int32)
    (root / "README.md").write_text("Local RL tool images; persisted per trajectory. Real image processor validation precedes observation attachment. Profile: local_4090.\n")
    return DataProto(batch=batch, non_tensor_batch=output)
