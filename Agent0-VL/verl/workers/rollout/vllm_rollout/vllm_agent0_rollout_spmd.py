# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Agent0-VL Rollout Worker implementing SERC (Self-Evolving Reasoning Cycle)

SERC inner loop (paper Algorithm 1):
1. Solver samples a reasoning step a_t (optionally containing a tool call)
2. The tool executes in a sandbox -> observation o_t
3. Verifier evaluates the step -> V_t = (score, confidence, critique, tool_check)
4. If confidence < tau_c: a repair instruction Delta_t is generated and the
   Solver re-samples a corrected segment a'_t

Token bookkeeping:
- ``multiturn_mask`` is True only on model-generated tokens (Solver steps,
  Verifier outputs, repair instructions, and repaired segments). Injected
  prompts and tool observations are False and therefore excluded from the
  policy loss.
- Per-step verification results and reward-relevant positions are exported in
  ``non_tensor_batch['step_data']`` for the Agent0 reward manager.
"""

import asyncio
import numpy as np
import re
import json
import os
import time
from typing import List, Dict, Any, Optional, Union

from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.reward_score.paper_eval import extract_boxed_answer
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    get_model_max_position_embeddings,
    vLLMRollout,
)
from verl.third_party.vllm import vllm_version

# Sandbox for tool execution. `sandbox.local_sandbox` talks to an HTTP
# sandbox service (SANDBOX_ENDPOINT); `sandbox.internal_sandbox` runs code in
# local subprocesses and requires no infrastructure.
try:
    if os.getenv("SANDBOX_ENDPOINT", None) is not None:
        from sandbox.local_sandbox import parallel_sandbox
    else:
        from sandbox.internal_sandbox import parallel_sandbox
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    parallel_sandbox = None


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove left padding from input token sequence"""
    non_pad = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if non_pad.numel() == 0:
        return []
    return prompt_token_ids[non_pad[0][0]:].tolist()


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def truncate_content(content: str, max_length: int = 2000) -> str:
    """Truncate content to max length keeping head and tail"""
    if len(content) <= max_length:
        return content
    return (
        content[: max_length // 2]
        + f"\n..._Truncated to {max_length} chars_...\n"
        + content[-max_length // 2:]
    )


class vLLMAgent0Rollout(vLLMRollout):
    """
    Agent0-VL rollout worker implementing the Self-Evolving Reasoning Cycle.

    Extends vLLMRollout with:
    - Multi-step reasoning with sandboxed tool execution
    - Step-by-step tool-grounded verification
    - Confidence-gated self-repair with segment re-sampling
    - Vision-language (multi-image) input handling
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        # Agent0-VL specific configuration (read before super().__init__ so we
        # can extend max_model_len for the multi-turn budget).
        self.max_reasoning_steps = int(config.get('max_reasoning_steps', config.get('num_turns', 8)))
        self.repair_threshold = float(config.get('repair_threshold', 0.7))
        self.enable_tool_execution = bool(config.get('enable_tool_execution', True))
        self.enable_step_verification = bool(config.get('enable_verification',
                                                        config.get('enable_step_verification', True)))
        self.enable_self_repair = bool(config.get('enable_self_repair', True))
        self.max_obs_length = int(config.get('max_obs_length', 512))
        self.sandbox_timeout = int(config.get('sandbox_timeout', 60))
        self.max_repairs_per_trajectory = int(config.get('max_repairs_per_trajectory', 2))
        self.verbose_logging = os.getenv('AGENT0_ROLLOUT_VERBOSE', '0').lower() in {
            '1', 'true', 'yes', 'on'
        }
        # Emit a small, bounded sample of generated text per local rollout
        # batch when verbose rollout logging is enabled. The launcher sets
        # these values so training logs show actual model behavior, not only
        # step timings.
        self.log_sample_count = max(0, int(os.getenv('AGENT0_ROLLOUT_LOG_SAMPLES', '0')))
        self.log_sample_max_chars = max(100, int(os.getenv('AGENT0_ROLLOUT_LOG_MAX_CHARS', '1200')))

        # The whole trajectory (all steps' Solver / tool-observation / Verifier /
        # repair tokens) is written into a single response buffer bounded by
        # ``max_total_response_length`` (or max_reasoning_steps * response_length).
        # The model context window therefore only needs to fit prompt + that
        # buffer, plus a small slack for the last in-flight generation. Sizing it
        # from an inflated per-step worst case would push max_model_len far above
        # what is ever used, blow up KV-cache memory, and trip vLLM's
        # ``max_num_batched_tokens < max_model_len`` chunked-prefill check.
        buffer_budget = int(config.get('max_total_response_length',
                                       self.max_reasoning_steps * config.response_length))
        buffer_budget = max(buffer_budget, config.response_length)
        extended_response_length = buffer_budget + self.max_obs_length + config.response_length

        original_max_model_len = config.get('max_model_len', None)
        if original_max_model_len is None:
            desired = config.prompt_length + extended_response_length
        else:
            desired = max(int(original_max_model_len), config.prompt_length + config.response_length)
        model_max_position_embeddings = get_model_max_position_embeddings(model_hf_config)
        config.max_model_len = min(desired, model_max_position_embeddings)

        # vLLM (and verl's base rollout) require max_num_batched_tokens >=
        # max_model_len when chunked prefill is enabled. The multi-turn budget
        # can push max_model_len above the configured token budget for large
        # ``max_total_response_length`` settings, so raise the budget to match
        # (a no-op for the default config where it is already large enough).
        if config.get('enable_chunked_prefill', True):
            mnbt = int(config.get('max_num_batched_tokens', 8192))
            if mnbt < int(config.max_model_len):
                config.max_num_batched_tokens = int(config.max_model_len)

        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        # Qwen checkpoints pad the language-model head to an aligned size
        # (152064 here), while the tokenizer only defines ids through 151664.
        # Without an allow-list, vLLM can sample one of the padded rows during
        # free-running RL generation. The invalid id is only noticed when the
        # next SERC turn feeds that response back into vLLM, producing
        # ``Token id ... is out of vocabulary``. Keep the padded weights for
        # checkpoint compatibility, but never sample those ids.
        tokenizer_vocab = tokenizer.get_vocab()
        valid_vocab_size = max(tokenizer_vocab.values(), default=-1) + 1
        text_config = getattr(model_hf_config, 'text_config', None)
        vocab_config = text_config if text_config is not None else model_hf_config
        model_vocab_size = int(getattr(vocab_config, 'vocab_size', valid_vocab_size))
        if valid_vocab_size < model_vocab_size:
            invalid_token_ids = list(range(valid_vocab_size, model_vocab_size))
            # vLLM V1 limits both allowed_token_ids and logit_bias to 1024
            # entries. The valid vocabulary is much larger than that, while
            # the padded tail in Qwen checkpoints is small (399 ids here).
            # Bias only the padded tail instead of constructing a huge
            # allow-list. -100 is vLLM's strongest supported negative bias and
            # is effectively zero probability for normal model logits.
            if hasattr(self.sampling_params, 'logit_bias'):
                logit_bias = dict(self.sampling_params.logit_bias or {})
                if len(logit_bias) + len(invalid_token_ids) > 1024:
                    raise RuntimeError(
                        'The padded vocabulary tail is too large for vLLM '
                        'V1 logit_bias (maximum 1024 entries).'
                    )
                logit_bias.update({token_id: -100.0 for token_id in invalid_token_ids})
                self.sampling_params.logit_bias = logit_bias
                mask_mode = 'logit_bias'
            elif hasattr(self.sampling_params, 'allowed_token_ids') and valid_vocab_size <= 1024:
                # Compatibility fallback for old vLLM releases where the
                # valid vocabulary itself fits the allow-list limit.
                self.sampling_params.allowed_token_ids = list(range(valid_vocab_size))
                mask_mode = 'allowed_token_ids'
            else:
                raise RuntimeError(
                    'This vLLM version cannot safely mask the padded Qwen '
                    'vocabulary within its token-ID limit.'
                )
            print(
                'Agent0-VL rollout: masked padded vocabulary ids '
                f'{valid_vocab_size}:{model_vocab_size - 1} '
                f'(tokenizer_vocab={valid_vocab_size}, model_vocab={model_vocab_size}, '
                f'mode={mask_mode})'
            )

        self.tokenizer = tokenizer
        self._load_prompt_templates()

        # Response buffer capacity. Bounded separately from max_model_len so a
        # large position-embedding budget doesn't blow up the response tensor.
        default_total = min(
            int(self.config.max_model_len) - int(self.config.prompt_length),
            self.max_reasoning_steps * int(self.config.response_length),
        )
        self.max_total_length = max(
            int(self.config.response_length),
            int(config.get('max_total_response_length', default_total)),
        )

        print(f"Agent0-VL Rollout initialized: max_steps={self.max_reasoning_steps}, "
              f"repair_threshold={self.repair_threshold}, tools={self.enable_tool_execution}, "
              f"verification={self.enable_step_verification}, self_repair={self.enable_self_repair}, "
              f"sandbox_available={SANDBOX_AVAILABLE}, max_model_len={self.config.max_model_len}, "
              f"verbose_logging={self.verbose_logging}")

    def _progress_log(self, message: str) -> None:
        """Emit rollout progress without changing the rollout algorithm."""
        if self.verbose_logging:
            print(f"[Agent0-VL rollout pid={os.getpid()}] {message}", flush=True)

    # ------------------------------------------------------------------
    # Prompt templates (aligned with the paper's Appendix C system prompts)
    # ------------------------------------------------------------------
    def _load_prompt_templates(self):
        """Load prompt templates for the Verifier and Self-Repair roles.

        JSON braces are escaped as ``{{ }}`` for use with ``str.format``.
        """
        self.tool_observation_prompt_template = "{observation}"
        self.solver_continue_prompt_template = "Continue solving the problem using the conversation so far."
        self.verifier_prompt_template = """Now switch to the Verifier role. Verify the reasoning step above using the available evidence.

Step to evaluate:
{step_content}

Tool outputs (if any):
{tool_outputs}

Principles:
- Ground verification on objective tool evidence.
- Penalize unsupported or inconsistent reasoning.
- High confidence requires agreement between tool and text.

Output exactly one JSON line:
{{"step_index": {step_index}, "score": <-1 to 1>, "confidence": <0 to 1>, "critique": "<at most 2 sentences>", "tool_check": <true|false>}}"""

        self.repair_prompt_template = """Now switch to the Self-Repair role. The Verifier flagged the reasoning step with low confidence.

Verification result:
- Score: {score}
- Confidence: {confidence}
- Critique: {critique}

Original step:
{original_step}

Propose a minimal local patch that fixes the specific error WITHOUT rewriting validated context.
Output exactly one JSON line, either:
{{"action": "PATCH", "target_step": {step_index}, "patch_type": "<text|code|tool_call|parameter>", "new_content": "<minimal replacement>", "justification": "<at most 2 sentences>"}}
or:
{{"action": "NO_CHANGE", "target_step": {step_index}, "reason": "<why repair is not warranted>"}}"""

        self.regenerate_prompt_template = """A repair instruction has been issued for the previous step:
{repair_instruction}

Switch back to the Solver role. Re-derive the corrected reasoning step applying this patch, then continue toward the final answer."""

    def _get_template_tokens(self, template_key: str,
                             add_generation_prompt: bool = True, **kwargs) -> List[int]:
        """Convert prompt template to token ids with variable substitution"""
        template = getattr(self, f"{template_key}_prompt_template", "")
        try:
            formatted = template.format(**kwargs)
        except (KeyError, IndexError) as e:
            print(f"Warning: prompt template formatting failed ({e}); using raw template")
            formatted = template

        messages = [{"role": "user", "content": formatted}]
        chat_template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

        # Remove the duplicated system prompt that apply_chat_template inserts
        # for Qwen models (the conversation already has one).
        if chat_template.startswith("<|im_start|>system"):
            system_end = chat_template.find("<|im_end|>") + len("<|im_end|>")
            chat_template = chat_template[system_end:].lstrip()
        if chat_template.startswith("<|begin_of_sentence|>"):
            chat_template = chat_template[len("<|begin_of_sentence|>"):]

        return self.tokenizer.encode(chat_template, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    def _execute_code_in_sandbox(self, code_blocks: List[str]) -> List[Dict[str, Any]]:
        """Execute code blocks in the sandbox and return results"""
        if not code_blocks:
            return []
        if not SANDBOX_AVAILABLE:
            return [{"success": False, "stdout": "", "stderr": "Sandbox not available"}] * len(code_blocks)

        try:
            success_list, stdout_list, stderr_list = asyncio.run(
                parallel_sandbox(code_blocks, num_processes=min(256, len(code_blocks)),
                                 run_timeout=self.sandbox_timeout)
            )
            results = []
            for success, stdout, stderr in zip(success_list, stdout_list, stderr_list):
                results.append({
                    "success": bool(success),
                    "stdout": truncate_content(str(stdout), 512),
                    "stderr": truncate_content(str(stderr), 512) if stderr else ""
                })
            return results
        except Exception as e:
            return [{"success": False, "stdout": "", "stderr": str(e)}] * len(code_blocks)

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from text.

        Tolerant of an optional ``python``/``py`` language tag and of code
        fences without a trailing newline before the closing ``` (so single-line
        snippets are captured too)."""
        pattern = r"```(?:python|py)?[ \t]*\r?\n?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer from solver output"""
        answers = []
        for match in re.finditer(r"\\boxed\{", text):
            start = match.end()
            depth = 1
            for end in range(start, len(text)):
                if text[end] == '{':
                    depth += 1
                elif text[end] == '}':
                    depth -= 1
                    if depth == 0:
                        answers.append(text[start:end].strip())
                        break
        if answers:
            return answers[-1]

        final_pattern = r"FINAL_ANSWER:\s*(.+?)(?:\n|$)"
        matches = re.findall(final_pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return None

    # ------------------------------------------------------------------
    # Verification / repair parsing
    # ------------------------------------------------------------------
    def _parse_verification_output(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse verification JSON from verifier output"""
        decoder = json.JSONDecoder()
        values = []
        for index, char in enumerate(text):
            if char != '{':
                continue
            try:
                value, _ = decoder.raw_decode(text, index)
                if isinstance(value, dict) and "step_index" in value:
                    values.append(value)
            except json.JSONDecodeError:
                continue
        if values:
            try:
                verification = values[-1]
                required = ['step_index', 'score', 'confidence']
                if all(k in verification for k in required):
                    verification['score'] = max(-1.0, min(1.0, float(verification.get('score', 0))))
                    verification['confidence'] = max(0.0, min(1.0, float(verification.get('confidence', 0.5))))
                    return verification
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        return None

    def _parse_repair_instruction(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse repair instruction JSON from repair output"""
        decoder = json.JSONDecoder()
        values = []
        for index, char in enumerate(text):
            if char != '{':
                continue
            try:
                value, _ = decoder.raw_decode(text, index)
                if isinstance(value, dict) and "action" in value:
                    values.append(value)
            except json.JSONDecodeError:
                continue
        if values:
            try:
                repair = values[-1]
                if repair.get('action') in ['PATCH', 'NO_CHANGE']:
                    return repair
            except json.JSONDecodeError:
                pass
        return None

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate multi-step SERC trajectories.

        Returns a DataProto with responses, attention_mask, position_ids,
        multiturn_mask (model-generated tokens only) and per-step metadata in
        ``non_tensor_batch['step_data']``.
        """
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        # generation_config.eos_token_id may be a list of stopping tokens
        # (Qwen2.5-VL uses [<|im_end|>, <|endoftext|>]). A chat turn needs
        # the single <|im_end|> token, not that list.
        chat_end_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if not isinstance(chat_end_token_id, int) or chat_end_token_id < 0:
            raise ValueError('Agent0 rollout requires a tokenizer with <|im_end|>')

        batch_size = idx.size(0)
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        rollout_started_at = time.monotonic()
        original_batch_size = int(batch_size)

        non_tensor_batch = dict(prompts.non_tensor_batch)

        # Sampling parameter overrides for this call
        if not do_sample:
            sample_kwargs = {'best_of': 1, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'temperature': 0, 'n': 1}
        elif is_validate:
            sample_kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,
            }
        else:
            sample_kwargs = {'n': 1}

        # Handle GRPO group sampling (n > 1): repeat everything upfront and
        # generate one sample per repeated row.
        n_repeat = self.config.n if (do_sample and self.config.n > 1 and not is_validate) else 1
        if n_repeat > 1:
            idx = _repeat_interleave(idx, n_repeat)
            attention_mask = _repeat_interleave(attention_mask, n_repeat)
            position_ids = _repeat_interleave(position_ids, n_repeat)
            batch_size = batch_size * n_repeat
            for key in list(non_tensor_batch.keys()):
                non_tensor_batch[key] = _repeat_interleave(non_tensor_batch[key], n_repeat)

        self._progress_log(
            f"start prompts={original_batch_size} expanded_batch={batch_size} "
            f"n={n_repeat} validate={is_validate} max_steps={self.max_reasoning_steps}"
        )

        # Build vLLM prompt token ids. For multimodal models the raw
        # (unexpanded) prompt ids must be used; the expanded ``input_ids`` are
        # only for FSDP training.
        if 'raw_prompt_ids' in non_tensor_batch:
            current_inputs = []
            for raw in non_tensor_batch['raw_prompt_ids']:
                current_inputs.append(list(raw) if not isinstance(raw, list) else list(raw))
        else:
            current_inputs = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]

        multi_modal_data = non_tensor_batch.get('multi_modal_data', None)

        # Response buffer
        max_total_length = self.max_total_length
        combined_response = torch.full((batch_size, max_total_length), self.pad_token_id,
                                       dtype=idx.dtype, device=idx.device)
        multiturn_mask = torch.zeros_like(combined_response, dtype=torch.bool)
        response_attention_mask = torch.zeros_like(combined_response, dtype=torch.bool)

        current_positions = [0] * batch_size

        # Per-sample bookkeeping
        step_data: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        verify_probs: List[List[float]] = [[] for _ in range(batch_size)]
        final_answers: List[Optional[str]] = [None] * batch_size
        num_steps = [0] * batch_size
        num_repairs = [0] * batch_size
        active_samples = list(range(batch_size))
        pending_tool_result = [False] * batch_size
        assistant_start_tokens = self.tokenizer.encode(
            '<|im_start|>assistant\n', add_special_tokens=False)

        verify_kwargs = dict(sample_kwargs)

        def _append_tokens(global_idx: int, tokens: List[int], trainable: bool) -> int:
            """Append tokens to a sample's response buffer with bounds checks.

            Returns the number of tokens actually written."""
            if not tokens:
                return 0
            pos = current_positions[global_idx]
            room = max_total_length - pos
            if room <= 0:
                return 0
            tokens = tokens[:room]
            tensor = torch.tensor(tokens, dtype=combined_response.dtype, device=combined_response.device)
            combined_response[global_idx, pos:pos + len(tokens)] = tensor
            if trainable:
                multiturn_mask[global_idx, pos:pos + len(tokens)] = True
            response_attention_mask[global_idx, pos:pos + len(tokens)] = True
            current_positions[global_idx] = pos + len(tokens)
            return len(tokens)

        def _append_user_turn(global_idx: int, tokens: List[int]) -> int:
            """Close the previous assistant turn before adding a chat-formatted user turn."""
            if current_inputs[global_idx] and current_inputs[global_idx][-1] != chat_end_token_id:
                tokens = [chat_end_token_id] + tokens
            written = _append_tokens(global_idx, tokens, trainable=False)
            current_inputs[global_idx].extend(tokens[:written])
            return written

        def _has_room(global_idx: int, needed: int) -> bool:
            """Whether the sample still has response-buffer and model-window room."""
            if current_positions[global_idx] + needed > max_total_length:
                return False
            ctx_len = len(current_inputs[global_idx]) + needed
            return ctx_len < int(self.config.max_model_len) - 8

        def _batched_generate(vllm_inputs: List[dict], gen_kwargs: dict, label: str) -> List:
            """Run vLLM generation with a per-call max_tokens that fits the window."""
            max_input_len = max(len(v['prompt_token_ids']) for v in vllm_inputs)
            room = int(self.config.max_model_len) - max_input_len - 8
            call_kwargs = dict(gen_kwargs)
            call_kwargs['max_tokens'] = max(16, min(self.config.response_length, room))
            self._progress_log(
                f"generate_start label={label} requests={len(vllm_inputs)} "
                f"max_input_tokens={max_input_len} max_tokens={call_kwargs['max_tokens']}"
            )
            with self.update_sampling_params(**call_kwargs):
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False)
            self._progress_log(f"generate_done label={label} requests={len(vllm_inputs)}")
            return outputs

        def _build_vllm_inputs(sample_indices: List[int]) -> List[dict]:
            inputs = []
            for g in sample_indices:
                entry = {'prompt_token_ids': list(current_inputs[g])}
                if multi_modal_data is not None:
                    entry['multi_modal_data'] = multi_modal_data[g]
                inputs.append(entry)
            return inputs

        # === SERC main loop ===
        for step_idx in range(self.max_reasoning_steps):
            # Drop samples that ran out of window/buffer room
            active_samples = [g for g in active_samples
                              if _has_room(g, min(256, self.config.response_length))]
            if not active_samples:
                self._progress_log(
                    f"stop_no_active step={step_idx + 1} elapsed={time.monotonic() - rollout_started_at:.1f}s"
                )
                break
            if step_idx > 0:
                for g in active_samples:
                    if pending_tool_result[g]:
                        written = _append_tokens(g, assistant_start_tokens, trainable=False)
                        current_inputs[g].extend(assistant_start_tokens[:written])
                        pending_tool_result[g] = False
                    else:
                        _append_user_turn(g, self._get_template_tokens('solver_continue'))
            step_active_count = len(active_samples)
            self._progress_log(
                f"step_start step={step_idx + 1}/{self.max_reasoning_steps} "
                f"active={step_active_count}"
            )

            # --- Step 1: Solver generates a reasoning step ---
            solver_outputs = _batched_generate(
                _build_vllm_inputs(active_samples), sample_kwargs,
                label=f"solver step={step_idx + 1}")

            solver_text_by_g: Dict[int, str] = {}
            tool_output_by_g: Dict[int, str] = {}
            tool_success_by_g: Dict[int, bool] = {}
            tool_call_count_by_g: Dict[int, int] = {}
            successful_tool_call_count_by_g: Dict[int, int] = {}
            newly_completed = []

            solver_responses = [list(output.outputs[0].token_ids) for output in solver_outputs]

            for local_idx, global_idx in enumerate(active_samples):
                tokens = solver_responses[local_idx]
                # strip trailing eos for context growth, but keep it in buffer
                written = _append_tokens(global_idx, tokens, trainable=True)
                current_inputs[global_idx].extend(tokens[:written])
                num_steps[global_idx] += 1

                solver_text = self.tokenizer.decode(tokens[:written], skip_special_tokens=True)
                solver_text_by_g[global_idx] = solver_text
                pending_tool_result[global_idx] = False

                final_answer = self._extract_final_answer(solver_text)
                if final_answer is not None:
                    final_answers[global_idx] = final_answer
                    newly_completed.append(global_idx)
            self._progress_log(
                f"solver_done step={step_idx + 1} requests={step_active_count} "
                f"completed={len(newly_completed)}"
            )

            # --- Step 2: Execute tools for still-active samples ---
            exec_candidates = list(active_samples)
            if self.enable_tool_execution and exec_candidates:
                code_blocks_batch = []
                samples_with_code = []
                for g in exec_candidates:
                    code_blocks = self._extract_code_blocks(solver_text_by_g.get(g, ""))
                    if code_blocks:
                        code_blocks_batch.extend(code_blocks)
                        samples_with_code.append((g, len(code_blocks)))

                if code_blocks_batch:
                    self._progress_log(
                        f"tool_start step={step_idx + 1} samples={len(samples_with_code)} "
                        f"blocks={len(code_blocks_batch)}"
                    )
                    exec_results = self._execute_code_in_sandbox(code_blocks_batch)
                    result_idx = 0
                    for g, num_blocks in samples_with_code:
                        sample_results = exec_results[result_idx:result_idx + num_blocks]
                        result_idx += num_blocks

                        tool_output_text = "\n[Code Execution Result]\n"
                        successful_calls = sum(bool(result.get('success', False)) for result in sample_results)
                        any_success = successful_calls > 0
                        for result in sample_results:
                            if result['stderr']:
                                tool_output_text += f"Error: {result['stderr']}\n"
                            elif result['stdout']:
                                tool_output_text += f"Output: {result['stdout']}\n"
                            else:
                                tool_output_text += "No output\n"

                        tool_call_count_by_g[g] = len(sample_results)
                        successful_tool_call_count_by_g[g] = successful_calls
                        tool_success_by_g[g] = any_success
                        tool_output_by_g[g] = tool_output_text

                        observation_ids = self.tokenizer.encode(
                            tool_output_text, add_special_tokens=False)[:self.max_obs_length]
                        observation = self.tokenizer.decode(
                            observation_ids, skip_special_tokens=False)
                        tool_tokens = self._get_template_tokens(
                            'tool_observation', add_generation_prompt=False,
                            observation=observation)
                        _append_user_turn(g, tool_tokens)
                        pending_tool_result[g] = True

                        # A tool result may print the boxed final answer
                        boxed = self._extract_final_answer(tool_output_text)
                        if boxed is not None and final_answers[g] is None:
                            final_answers[g] = boxed
                    self._progress_log(
                        f"tool_done step={step_idx + 1} samples={len(samples_with_code)} "
                        f"successes={sum(bool(tool_success_by_g.get(g, False)) for g, _ in samples_with_code)}"
                    )

            # --- Step 3: Verifier evaluates the step ---
            verification_by_g: Dict[int, Optional[Dict[str, Any]]] = {}
            verify_candidates = [g for g in active_samples
                                 if self.enable_step_verification and _has_room(g, 256)]
            if verify_candidates:
                for g in verify_candidates:
                    verify_prompt_tokens = self._get_template_tokens(
                        "verifier",
                        step_content=truncate_content(solver_text_by_g.get(g, ""), 2000),
                        tool_outputs=truncate_content(tool_output_by_g.get(g, "None"), 1000),
                        step_index=step_idx + 1,
                    )
                    _append_user_turn(g, verify_prompt_tokens)
                    pending_tool_result[g] = False

                verify_outputs = _batched_generate(
                    _build_vllm_inputs(verify_candidates), verify_kwargs,
                    label=f"verifier step={step_idx + 1}")

                for local_idx, g in enumerate(verify_candidates):
                    tokens = list(verify_outputs[local_idx].outputs[0].token_ids)
                    written = _append_tokens(g, tokens, trainable=True)
                    current_inputs[g].extend(tokens[:written])

                    verify_text = self.tokenizer.decode(tokens[:written], skip_special_tokens=True)
                    verification = self._parse_verification_output(verify_text)
                    verification_by_g[g] = verification

                    verify_prob = verification.get('confidence', 0.5) if verification else 0.5
                    verify_probs[g].append(float(verify_prob))
                valid_verifications = sum(value is not None for value in verification_by_g.values())
                self._progress_log(
                    f"verifier_done step={step_idx + 1} requests={len(verify_candidates)} "
                    f"parsed={valid_verifications}"
                )

            # --- Step 4: Confidence-gated self-repair ---
            repair_candidates = []
            if self.enable_self_repair:
                for g in verify_candidates:
                    if g in newly_completed:
                        continue
                    verification = verification_by_g.get(g)
                    confidence = verification.get('confidence', 0.5) if verification else 0.5
                    if (confidence < self.repair_threshold
                            and num_repairs[g] < self.max_repairs_per_trajectory
                            and _has_room(g, 512)):
                        repair_candidates.append(g)

            initial_verification_by_g = dict(verification_by_g)
            repair_pre_verification_by_g = {
                g: verification_by_g.get(g) for g in repair_candidates
            }
            reverification_by_g: Dict[int, Optional[Dict[str, Any]]] = {}
            repair_success_by_g: Dict[int, bool] = {}
            repair_score_improved_by_g: Dict[int, bool] = {}
            repaired_by_g: Dict[int, bool] = {}
            if repair_candidates:
                self._progress_log(
                    f"repair_start step={step_idx + 1} candidates={len(repair_candidates)}"
                )
                # 4a: generate repair instructions
                for g in repair_candidates:
                    verification = verification_by_g.get(g)
                    repair_prompt_tokens = self._get_template_tokens(
                        "repair",
                        score=verification.get('score', 0) if verification else 0,
                        confidence=verification.get('confidence', 0.5) if verification else 0.5,
                        critique=(verification.get('critique', 'Low confidence')
                                  if verification else 'Verification failed'),
                        original_step=truncate_content(solver_text_by_g.get(g, ""), 1000),
                        step_index=step_idx + 1,
                    )
                    _append_user_turn(g, repair_prompt_tokens)

                repair_outputs = _batched_generate(
                    _build_vllm_inputs(repair_candidates), sample_kwargs,
                    label=f"repair step={step_idx + 1}")

                regen_candidates = []
                for local_idx, g in enumerate(repair_candidates):
                    tokens = list(repair_outputs[local_idx].outputs[0].token_ids)
                    written = _append_tokens(g, tokens, trainable=True)
                    current_inputs[g].extend(tokens[:written])

                    repair_text = self.tokenizer.decode(tokens[:written], skip_special_tokens=True)
                    repair = self._parse_repair_instruction(repair_text)
                    if repair is not None and repair.get('action') == 'PATCH' and _has_room(g, 512):
                        regen_candidates.append((g, repair))

                # 4b: Solver re-samples the corrected segment a'_t
                if regen_candidates:
                    for g, repair in regen_candidates:
                        regen_prompt_tokens = self._get_template_tokens(
                            "regenerate",
                            repair_instruction=truncate_content(json.dumps(repair, ensure_ascii=False), 800),
                        )
                        _append_user_turn(g, regen_prompt_tokens)

                    regen_indices = [g for g, _ in regen_candidates]
                    regen_outputs = _batched_generate(
                        _build_vllm_inputs(regen_indices), sample_kwargs,
                        label=f"regenerate step={step_idx + 1}")

                    for local_idx, g in enumerate(regen_indices):
                        tokens = list(regen_outputs[local_idx].outputs[0].token_ids)
                        written = _append_tokens(g, tokens, trainable=True)
                        current_inputs[g].extend(tokens[:written])

                        regen_text = self.tokenizer.decode(tokens[:written], skip_special_tokens=True)
                        solver_text_by_g[g] = regen_text  # corrected step replaces the old one
                        num_repairs[g] += 1
                        repaired_by_g[g] = True

                        # Reverification must inspect evidence from the corrected code.
                        tool_output_by_g.pop(g, None)
                        tool_success_by_g[g] = False
                        blocks = self._extract_code_blocks(regen_text) if self.enable_tool_execution else []
                        if blocks:
                            results = self._execute_code_in_sandbox(blocks)
                            tool_output = "\n[Code Execution Result]\n" + "".join(
                                f"Error: {r['stderr']}\n" if r['stderr'] else
                                f"Output: {r['stdout']}\n" if r['stdout'] else "No output\n"
                                for r in results)
                            successes = sum(bool(r.get('success')) for r in results)
                            tool_call_count_by_g[g] = tool_call_count_by_g.get(g, 0) + len(results)
                            successful_tool_call_count_by_g[g] = successful_tool_call_count_by_g.get(g, 0) + successes
                            tool_success_by_g[g] = successes > 0
                            tool_output_by_g[g] = tool_output
                            obs_ids = self.tokenizer.encode(tool_output, add_special_tokens=False)[:self.max_obs_length]
                            _append_user_turn(g, self._get_template_tokens(
                                'tool_observation', add_generation_prompt=False,
                                observation=self.tokenizer.decode(obs_ids, skip_special_tokens=False)))

                        final_answer = self._extract_final_answer(regen_text)
                        if final_answer is not None:
                            final_answers[g] = final_answer
                            if g not in newly_completed:
                                newly_completed.append(g)

                    # The repaired segment must be evaluated again.  The
                    # paper's repair loop uses the post-repair verifier result
                    # for the process reward; retaining the pre-repair score
                    # would reward an unverified correction.
                    if self.enable_step_verification and regen_indices:
                        for g, _ in regen_candidates:
                            verify_prompt_tokens = self._get_template_tokens(
                                "verifier",
                                step_content=truncate_content(solver_text_by_g.get(g, ""), 2000),
                                tool_outputs=truncate_content(tool_output_by_g.get(g, "None"), 1000),
                                step_index=step_idx + 1,
                            )
                            _append_user_turn(g, verify_prompt_tokens)
                        reverify_outputs = _batched_generate(
                            _build_vllm_inputs(regen_indices), verify_kwargs,
                            label=f"reverify step={step_idx + 1}")
                        for local_idx, g in enumerate(regen_indices):
                            tokens = list(reverify_outputs[local_idx].outputs[0].token_ids)
                            written = _append_tokens(g, tokens, trainable=True)
                            current_inputs[g].extend(tokens[:written])
                            reverified = self._parse_verification_output(
                                self.tokenizer.decode(tokens[:written], skip_special_tokens=True))
                            reverification_by_g[g] = reverified
                            verification_by_g[g] = reverified
                            if reverified is not None:
                                repair_success_by_g[g] = (
                                    float(reverified.get('confidence', 0.5)) >= self.repair_threshold
                                )
                                original_verification = repair_pre_verification_by_g.get(g)
                                original_score = (
                                    float(original_verification.get('score', 0.0))
                                    if original_verification else 0.0
                                )
                                repair_score_improved_by_g[g] = (
                                    float(reverified.get('score', 0.0)) > original_score
                                )
                self._progress_log(
                    f"repair_done step={step_idx + 1} candidates={len(repair_candidates)} "
                    f"regenerated={len(regen_candidates)}"
                )

            # --- Record per-step data for the reward manager ---
            for g in active_samples:
                verification = verification_by_g.get(g)
                step_data[g].append({
                    'step_index': step_idx + 1,
                    'score': float(verification.get('score', 0.0)) if verification else 0.0,
                    'confidence': float(verification.get('confidence', 0.5)) if verification else 0.5,
                    'verified': verification is not None,
                    'verifier_triggered': g in verify_candidates,
                    'verifier_calls': int(g in verify_candidates) + int(g in reverification_by_g),
                    'verifier_successful_calls': int(initial_verification_by_g.get(g) is not None)
                    + int(reverification_by_g.get(g) is not None),
                    'tool_used': g in tool_output_by_g,
                    'tool_success': bool(tool_success_by_g.get(g, False)),
                    'tool_call_count': int(tool_call_count_by_g.get(g, 0)),
                    'successful_tool_call_count': int(successful_tool_call_count_by_g.get(g, 0)),
                    'repair_triggered': g in repair_candidates,
                    'was_repaired': bool(repaired_by_g.get(g, False)),
                    'repair_success': bool(repair_success_by_g.get(g, False)),
                    'repair_score_improved': bool(repair_score_improved_by_g.get(g, False)),
                    'step_end_pos': current_positions[g],
                })

            # Remove completed samples
            completed_now = sum(final_answers[g] is not None for g in active_samples)
            active_samples = [g for g in active_samples if final_answers[g] is None]
            self._progress_log(
                f"step_done step={step_idx + 1} processed={step_active_count} "
                f"completed={completed_now} tools={len(tool_output_by_g)} "
                f"verified={len(verification_by_g)} repaired={sum(repaired_by_g.values())} "
                f"active_remaining={len(active_samples)} "
                f"elapsed={time.monotonic() - rollout_started_at:.1f}s"
            )

        # === Finalize outputs ===
        # Keep the output shape identical across DP workers. Local samples
        # can finish at different points, so trimming to the local maximum
        # would make DataProto.concat fail across GPUs.
        max_response_len = max_total_length
        combined_response = combined_response[:, :max_response_len]
        multiturn_mask = multiturn_mask[:, :max_response_len]
        response_attention_mask = response_attention_mask[:, :max_response_len]

        seq = torch.cat([idx, combined_response], dim=-1)

        delta_position_id = torch.arange(1, combined_response.size(1) + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask.to(attention_mask.dtype)), dim=-1)

        batch = TensorDict({
            'prompts': idx,
            'responses': combined_response,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'multiturn_mask': multiturn_mask,
        }, batch_size=batch_size)

        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        # Non-tensor outputs. Everything is already repeated to batch_size.
        # Verification can stop at different SERC turns per sample. Keep
        # each sample's probability history as one object so DP workers can
        # concatenate batches with different second-axis lengths.
        verify_probs_array = np.empty(batch_size, dtype=object)
        for i, vp in enumerate(verify_probs):
            verify_probs_array[i] = np.asarray(vp, dtype=np.float32)

        step_data_array = np.empty(batch_size, dtype=object)
        for i in range(batch_size):
            step_data_array[i] = step_data[i]
        final_answers_array = np.empty(batch_size, dtype=object)
        for i in range(batch_size):
            final_answers_array[i] = final_answers[i]

        if self.verbose_logging and self.log_sample_count:
            # GRPO samples are repeated contiguously per prompt. Log a bounded
            # number of actual trajectories so sampled format drift is visible.
            for sample_idx in range(min(batch_size, self.log_sample_count)):
                prompt_slot = sample_idx // n_repeat
                response_length = current_positions[sample_idx]
                response_text = self.tokenizer.decode(
                    combined_response[sample_idx, :response_length].tolist(),
                    skip_special_tokens=True,
                )
                response_text = ' '.join(response_text.split())
                if len(response_text) > self.log_sample_max_chars:
                    response_text = truncate_content(response_text, self.log_sample_max_chars)
                dataset_index = non_tensor_batch.get('index')
                if dataset_index is not None:
                    dataset_index = dataset_index[sample_idx]
                    if torch.is_tensor(dataset_index):
                        dataset_index = dataset_index.item()
                    elif isinstance(dataset_index, np.generic):
                        dataset_index = dataset_index.item()
                sample_summary = {
                    'prompt_slot': prompt_slot,
                    'dataset_index': dataset_index,
                    'trajectory': sample_idx % n_repeat + 1,
                    'trajectories_per_prompt': n_repeat,
                    'parsed_answer': final_answers[sample_idx],
                    'steps': num_steps[sample_idx],
                    'text': response_text,
                }
                self._progress_log('sample ' + json.dumps(sample_summary, ensure_ascii=False))

        output_non_tensor = {
            'step_data': step_data_array,
            'verify_probs': verify_probs_array,
            'num_steps': np.array(num_steps, dtype=np.int32),
            'num_repairs': np.array(num_repairs, dtype=np.int32),
            'final_answers': final_answers_array,
            'final_generation_step': np.array([max(n - 1, 0) for n in num_steps], dtype=np.int32),
        }
        # Preserve inputs the trainer needs back (e.g. multi_modal_inputs for
        # log-prob recomputation), already repeated to batch_size.
        for key, value in non_tensor_batch.items():
            if key in ('raw_prompt_ids', 'multi_modal_data'):
                continue
            if key not in output_non_tensor:
                output_non_tensor[key] = value

        self._progress_log(
            f"finished expanded_batch={batch_size} completed={sum(answer is not None for answer in final_answers)} "
            f"total_steps={sum(num_steps)} total_repairs={sum(num_repairs)} "
            f"elapsed={time.monotonic() - rollout_started_at:.1f}s"
        )

        return DataProto(batch=batch, non_tensor_batch=output_non_tensor)
