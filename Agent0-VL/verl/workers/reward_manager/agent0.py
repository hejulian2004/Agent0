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
Agent0-VL Reward Manager - SERC Process Rewards

Implements the reward formulation from the paper:
    r_proc^(t) = lambda_tool * r(tool_t) + score_t * conf_t - beta_div * D_KL   (Eq. 2)
    g_t        = sigmoid(kappa * (tau_c - conf_t))                              (Eq. 3)
    r_t        = r_proc^(t) - g_t * C_repair^(t)                                (Eq. 4)
    g(tau)     = alpha_out * r_out + sum_t gamma^(t-1) r_t                      (Eq. 5)

Token-level placement: the discounted step reward gamma^(t-1) * r_t is placed
at the end position of step t (as reported by the rollout worker in
``non_tensor_batch['step_data']``), and alpha_out * r_out is added at the last
generated token. Summing the reward tensor over the sequence therefore yields
exactly g(tau), which is what GRPO normalizes per prompt group.
"""

from verl import DataProto
from verl.utils.reward_score.math_verify import compute_score as get_policy_score
import torch
import numpy as np
from typing import Dict, List, Any, Union
from collections import defaultdict
import math


class Agent0RewardManager:
    """Agent0-VL reward manager implementing SERC process rewards."""

    def __init__(self, tokenizer, num_examine: int, config=None, compute_score=None, **kwargs):
        """
        Args:
            tokenizer: Tokenizer used to decode responses.
            num_examine: Number of samples per data source to print for debugging.
            config: Dict-like config with reward parameters (lambda_tool,
                beta_div, alpha_out, gamma, tau_c, eta, kappa, max_steps).
            compute_score: Optional custom outcome scoring function with the
                signature ``fn(solution_str, ground_truth) -> dict|float``.
            **kwargs: Reward parameters may also be passed directly as keyword
                arguments (e.g. ``lambda_tool=0.3``); they override ``config``.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score

        config = config or {}

        def _get(key, default):
            if key in kwargs and kwargs[key] is not None:
                return kwargs[key]
            try:
                value = config.get(key, default)
            except AttributeError:
                value = default
            return default if value is None else value

        # Reward weights (paper Eq. 2 / Eq. 5)
        self.lambda_tool = float(_get('lambda_tool', 0.3))
        self.beta_div = float(_get('beta_div', 0.01))
        self.alpha_out = float(_get('alpha_out', 1.0))
        self.gamma = float(_get('gamma', 0.99))

        # Repair parameters (paper Eq. 3 / Eq. 4)
        self.tau_c = float(_get('tau_c', _get('repair_threshold', 0.7)))
        self.eta = float(_get('eta', _get('repair_penalty', 0.05)))
        self.kappa = float(_get('kappa', 5.0))

        self.max_steps = int(_get('max_steps', _get('num_turns', 8)))

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """Compute SERC rewards.

        Returns the reward tensor (same shape as responses) by default, or a
        dict ``{"reward_tensor", "reward_extra_info", "metrics"}`` when
        ``return_dict=True``.
        """
        if 'rm_scores' in data.batch:
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores'], "reward_extra_info": {}, "metrics": {}}
            return data.batch['rm_scores']

        batch_size = data.batch['responses'].shape[0]
        device = data.batch['responses'].device

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        metrics_tensors = {
            'step_scores': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'verify_confidences': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'step_counts': torch.zeros(batch_size, dtype=torch.long, device=device),
            'repair_flags': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'tool_success': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'tool_calls': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'successful_tool_calls': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'verifier_triggered': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'verifier_calls': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'valid_verifier_calls': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'repair_triggered': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'repair_applied': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'repair_success': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'repair_score_improved': torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=device),
            'outcome_acc': torch.zeros(batch_size, dtype=torch.float32, device=device),
        }

        reward_extra_info = defaultdict(list)
        printed_sources: Dict[str, int] = {}

        for i in range(batch_size):
            data_item = data[i]
            response_ids = data_item.batch['responses']
            seq_len = response_ids.shape[-1]

            multiturn_mask = data_item.batch.get('multiturn_mask', None)
            ground_truth = self._extract_ground_truth(data_item)
            data_source = str(data_item.non_tensor_batch.get('data_source', 'unknown'))
            step_data = list(data_item.non_tensor_batch.get('step_data', []) or [])

            # Fallback: derive step boundaries from the multiturn mask when the
            # rollout did not provide step_data (e.g. other rollout types).
            if not step_data and multiturn_mask is not None:
                step_data = [
                    {'score': 0.0, 'confidence': 0.5, 'tool_success': False,
                     'was_repaired': False, 'verified': False, 'step_end_pos': int(pos)}
                    for pos in self._find_step_boundaries(multiturn_mask)
                ]

            num_steps = len(step_data)
            metrics_tensors['step_counts'][i] = num_steps

            # ---- Per-step process rewards (Eq. 2-4) ----
            effective_rewards = []
            for step_idx, sd in enumerate(step_data):
                score = float(sd.get('score', 0.0))
                confidence = float(sd.get('confidence', 0.5))
                tool_success = bool(sd.get('tool_success', False))
                was_repaired = bool(sd.get('was_repaired', False))
                # Cross-role KL (paper Eq. 2). Inactive unless the rollout
                # populates step_data['kl_divergence']; defaults to 0 otherwise.
                kl_div = float(sd.get('kl_divergence', 0.0))

                r_tool = 1.0 if tool_success else 0.0
                r_proc = self.lambda_tool * r_tool + score * confidence - self.beta_div * kl_div

                # Repair gate and cost
                g_t = self._sigmoid(self.kappa * (self.tau_c - confidence))
                c_repair = self.eta if was_repaired else 0.0
                r_t = r_proc - g_t * c_repair
                effective_rewards.append(r_t)

                # Place discounted reward at the step's final token
                step_end_pos = int(sd.get('step_end_pos', 0))
                pos = min(max(step_end_pos, 1), seq_len) - 1
                reward_tensor[i, pos] += (self.gamma ** step_idx) * r_t

                if step_idx < self.max_steps:
                    metrics_tensors['step_scores'][i, step_idx] = score
                    if sd.get('verifier_triggered', sd.get('verified', False)):
                        metrics_tensors['verify_confidences'][i, step_idx] = confidence
                    metrics_tensors['repair_flags'][i, step_idx] = 1.0 if was_repaired else 0.0
                    metrics_tensors['tool_success'][i, step_idx] = r_tool
                    metrics_tensors['tool_calls'][i, step_idx] = int(
                        sd.get('tool_call_count', int(sd.get('tool_used', False)))
                    )
                    metrics_tensors['successful_tool_calls'][i, step_idx] = int(
                        sd.get('successful_tool_call_count', int(tool_success))
                    )
                    metrics_tensors['verifier_triggered'][i, step_idx] = float(
                        sd.get('verifier_triggered', sd.get('verified', False))
                    )
                    metrics_tensors['verifier_calls'][i, step_idx] = int(
                        sd.get('verifier_calls', int(sd.get('verified', False)))
                    )
                    metrics_tensors['valid_verifier_calls'][i, step_idx] = int(
                        sd.get('verifier_successful_calls', int(sd.get('verified', False)))
                    )
                    metrics_tensors['repair_triggered'][i, step_idx] = float(
                        sd.get('repair_triggered', was_repaired)
                    )
                    metrics_tensors['repair_applied'][i, step_idx] = float(was_repaired)
                    metrics_tensors['repair_success'][i, step_idx] = float(
                        sd.get('repair_success', False)
                    )
                    metrics_tensors['repair_score_improved'][i, step_idx] = float(
                        sd.get('repair_score_improved', False)
                    )

            # ---- Outcome reward r_out (Eq. 5) ----
            # The rollout already extracts each trajectory's final answer
            # (\boxed{} / FINAL_ANSWER) into non_tensor_batch['final_answers'].
            # Score that directly: decoding the whole trainable span would end on
            # the Verifier's JSON, whose confidence number gets mistaken for the
            # prediction and collapses the outcome signal.
            final_answer = data_item.non_tensor_batch.get('final_answers', None)
            if final_answer is not None and str(final_answer).strip():
                solution_str = "\\boxed{" + str(final_answer).strip() + "}"
            else:
                # Fallback: no explicit final answer emitted — decode the model's
                # trainable tokens so a boxed answer inside the text is still scored.
                if multiturn_mask is not None and multiturn_mask.any():
                    decode_ids = response_ids[multiturn_mask.bool()]
                else:
                    decode_ids = response_ids
                solution_str = self.tokenizer.decode(decode_ids, skip_special_tokens=True)

            reward_model = data_item.non_tensor_batch.get('reward_model', {})
            if not isinstance(reward_model, dict):
                reward_model = {}
            if self.compute_score is not None:
                outcome_result = self.compute_score(solution_str=solution_str, ground_truth=ground_truth)
            else:
                outcome_result = get_policy_score(solution_str=solution_str, ground_truth=ground_truth)
            if isinstance(outcome_result, dict):
                r_out = float(outcome_result.get('acc', outcome_result.get('score', 0.0)))
                pred = outcome_result.get('pred', '')
            else:
                r_out = float(outcome_result)
                pred = ''
            metrics_tensors['outcome_acc'][i] = r_out

            final_pos = self._last_valid_position(data_item, seq_len)
            reward_tensor[i, final_pos] += self.alpha_out * r_out

            # ---- Extra info ----
            reward_extra_info['pred'].append(pred)
            reward_extra_info['acc'].append(r_out)
            reward_extra_info['ground_truth'].append(ground_truth)
            reward_extra_info['response'].append(
                self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            reward_extra_info['num_steps'].append(num_steps)
            reward_extra_info['num_repairs'].append(
                sum(1 for sd in step_data if sd.get('was_repaired', False)))
            reward_extra_info['tool_call_count'].append(
                sum(int(sd.get('tool_call_count', int(sd.get('tool_used', False)))) for sd in step_data))
            reward_extra_info['successful_tool_calls'].append(
                sum(int(sd.get('successful_tool_call_count', int(sd.get('tool_success', False))))
                    for sd in step_data))
            reward_extra_info['verifier_trigger_count'].append(
                sum(int(sd.get('verifier_triggered', sd.get('verified', False))) for sd in step_data))
            reward_extra_info['verifier_call_count'].append(
                sum(int(sd.get('verifier_calls', int(sd.get('verified', False)))) for sd in step_data))
            reward_extra_info['valid_verifier_calls'].append(
                sum(int(sd.get('verifier_successful_calls', int(sd.get('verified', False))))
                    for sd in step_data))
            reward_extra_info['repair_trigger_count'].append(
                sum(int(sd.get('repair_triggered', sd.get('was_repaired', False))) for sd in step_data))
            reward_extra_info['repair_applied_count'].append(
                sum(int(sd.get('was_repaired', False)) for sd in step_data))
            reward_extra_info['repair_success_count'].append(
                sum(int(sd.get('repair_success', False)) for sd in step_data))
            reward_extra_info['repair_score_improvement_count'].append(
                sum(int(sd.get('repair_score_improved', False)) for sd in step_data))
            reward_extra_info['total_return'].append(
                self.alpha_out * r_out + sum((self.gamma ** t) * r for t, r in enumerate(effective_rewards)))

            if self.num_examine > 0 and printed_sources.get(data_source, 0) < self.num_examine:
                printed_sources[data_source] = printed_sources.get(data_source, 0) + 1
                print(f"\n{'=' * 80}")
                print(f"[Agent0] Sample {i} ({data_source})")
                print(f"Ground Truth: {ground_truth}")
                print(f"Prediction: {pred}")
                print(f"Outcome Reward: {r_out:.3f} | Steps: {num_steps} | "
                      f"Effective step rewards: {[f'{r:.3f}' for r in effective_rewards]}")
                print(self.tokenizer.decode(response_ids, skip_special_tokens=True)[:2000])
                print(f"{'=' * 80}\n")

        data_sources = [str(data[i].non_tensor_batch.get('data_source', 'unknown')) for i in range(batch_size)]
        metrics = self._compute_metrics(metrics_tensors, data_sources)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
                "metrics": metrics,
            }
        return reward_tensor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_ground_truth(data_item) -> str:
        rm_info = data_item.non_tensor_batch.get('reward_model', None)
        if isinstance(rm_info, dict) and 'ground_truth' in rm_info:
            return str(rm_info['ground_truth'])
        gt = data_item.non_tensor_batch.get('ground_truth', '')
        return str(gt)

    @staticmethod
    def _last_valid_position(data_item, seq_len: int) -> int:
        """Index of the last attended response token (0-based)."""
        attn = data_item.batch.get('attention_mask', None)
        if attn is not None:
            response_attn = attn[-seq_len:]
            nz = torch.nonzero(response_attn, as_tuple=False)
            if nz.numel() > 0:
                return int(nz[-1][0])
        return seq_len - 1

    def _find_step_boundaries(self, multiturn_mask: torch.Tensor) -> List[int]:
        """Step end positions (1-indexed) from True->False transitions."""
        if multiturn_mask.numel() == 0:
            return []
        mask = multiturn_mask.bool()
        boundaries = []
        padded = torch.cat([mask[:1].new_zeros(1), mask])  # pad False at start
        diff = padded[1:].long() - padded[:-1].long()
        # end of a True-run is where diff == -1 (position of first False)
        ends = torch.where(diff == -1)[0].tolist()
        boundaries.extend(int(e) for e in ends)
        if mask[-1]:
            boundaries.append(int(mask.size(0)))
        return boundaries

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _compute_metrics(self, metrics_tensors: Dict[str, torch.Tensor], data_sources: List[str]) -> Dict[str, float]:
        metrics = self._compute_single_group_metrics(metrics_tensors)

        data_sources_arr = np.array(data_sources)
        for source in np.unique(data_sources_arr):
            indices = np.where(data_sources_arr == source)[0].tolist()
            if indices:
                group = {k: v[indices] for k, v in metrics_tensors.items()}
                metrics.update(self._compute_single_group_metrics(group, prefix=f'{source}/'))
        return metrics

    def _compute_single_group_metrics(self, t: Dict[str, torch.Tensor], prefix: str = "") -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        step_counts = t['step_counts']
        batch_size = step_counts.shape[0]
        if batch_size == 0:
            return metrics

        metrics[f'{prefix}outcome_acc'] = t['outcome_acc'].mean().item()
        metrics[f'{prefix}answer_accuracy'] = t['outcome_acc'].mean().item()
        metrics[f'{prefix}avg_num_steps'] = step_counts.float().mean().item()

        # Per-step verification statistics over samples that reached step i
        for i in range(self.max_steps):
            valid_mask = step_counts > i
            if valid_mask.any():
                metrics[f'{prefix}step_{i + 1}_score'] = t['step_scores'][valid_mask, i].mean().item()
                metrics[f'{prefix}step_{i + 1}_confidence'] = t['verify_confidences'][valid_mask, i].mean().item()

        total_steps = step_counts.sum().item()
        verifier_steps = t['verifier_triggered'].sum().item()
        verifier_calls = t['verifier_calls'].sum().item()
        valid_verifier_calls = t['valid_verifier_calls'].sum().item()
        repair_triggers = t['repair_triggered'].sum().item()
        repair_applied = t['repair_applied'].sum().item()
        repair_successes = t['repair_success'].sum().item()
        repair_score_improvements = t['repair_score_improved'].sum().item()
        tool_calls = t['tool_calls'].sum().item()
        successful_tool_calls = t['successful_tool_calls'].sum().item()

        metrics[f'{prefix}verifier_trigger_count'] = verifier_steps
        metrics[f'{prefix}verifier_trigger_rate'] = verifier_steps / max(total_steps, 1)
        metrics[f'{prefix}verifier_call_count'] = verifier_calls
        metrics[f'{prefix}valid_verifier_calls'] = valid_verifier_calls
        metrics[f'{prefix}verifier_call_success_rate'] = valid_verifier_calls / max(verifier_calls, 1)
        metrics[f'{prefix}repair_trigger_count'] = repair_triggers
        metrics[f'{prefix}repair_trigger_rate'] = repair_triggers / max(verifier_steps, 1)
        metrics[f'{prefix}repair_applied_count'] = repair_applied
        metrics[f'{prefix}repair_application_rate'] = repair_applied / max(repair_triggers, 1)
        metrics[f'{prefix}repair_success_count'] = repair_successes
        metrics[f'{prefix}repair_success_rate'] = repair_successes / max(repair_applied, 1)
        metrics[f'{prefix}repair_score_improvement_count'] = repair_score_improvements
        metrics[f'{prefix}repair_score_improvement_rate'] = repair_score_improvements / max(repair_applied, 1)
        metrics[f'{prefix}repair_rate'] = repair_applied / max(total_steps, 1)
        metrics[f'{prefix}total_repairs'] = repair_applied
        metrics[f'{prefix}tool_call_count'] = tool_calls
        metrics[f'{prefix}successful_tool_calls'] = successful_tool_calls
        metrics[f'{prefix}tool_call_success_rate'] = successful_tool_calls / max(tool_calls, 1)
        metrics[f'{prefix}tool_success_rate'] = metrics[f'{prefix}tool_call_success_rate']

        tool_step_success_sum, tool_step_count, conf_sum = 0.0, 0, 0.0
        for b in range(batch_size):
            n_steps = min(int(step_counts[b].item()), self.max_steps)
            if n_steps > 0:
                tool_step_success_sum += t['tool_success'][b, :n_steps].sum().item()
                conf_sum += t['verify_confidences'][b, :n_steps].sum().item()
                tool_step_count += n_steps
        metrics[f'{prefix}tool_step_success_rate'] = tool_step_success_sum / max(tool_step_count, 1)
        metrics[f'{prefix}avg_verify_confidence'] = conf_sum / max(verifier_steps, 1)

        return metrics
