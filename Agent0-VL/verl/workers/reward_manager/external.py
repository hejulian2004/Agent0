"""External correctness reward used for the paper's RL warm-up phase."""

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score.math_verify import compute_score


class ExternalCorrectnessRewardManager:
    """Assign only the final-answer correctness reward to each trajectory.

    This intentionally excludes SERC process rewards, verifier confidence, and
    repair costs.  It is used only for the short conventional-RL warm-up before
    the self-evolving reward manager is enabled.
    """

    def __init__(self, tokenizer, num_examine=0, config=None, compute_score_fn=None, **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score_fn = compute_score_fn

    @staticmethod
    def _last_response_position(item, response_length):
        attention = item.batch.get("attention_mask")
        if attention is not None:
            response_attention = attention[-response_length:]
            valid = torch.nonzero(response_attention, as_tuple=False)
            if valid.numel():
                return int(valid[-1, 0])
        return response_length - 1

    def __call__(self, data: DataProto, return_dict=False):
        batch_size, response_length = data.batch["responses"].shape
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extras = defaultdict(list)

        for i in range(batch_size):
            item = data[i]
            answer = item.non_tensor_batch.get("final_answers")
            if answer is not None and str(answer).strip():
                solution = "\\boxed{" + str(answer).strip() + "}"
            else:
                response = item.batch["responses"]
                mask = item.batch.get("multiturn_mask")
                if mask is not None and mask.any():
                    response = response[mask.bool()]
                solution = self.tokenizer.decode(response, skip_special_tokens=True)

            reward_info = item.non_tensor_batch.get("reward_model", {})
            if not isinstance(reward_info, dict):
                reward_info = {}
            ground_truth = str(reward_info.get("ground_truth", ""))
            if self.compute_score_fn is not None:
                result = self.compute_score_fn(solution_str=solution, ground_truth=ground_truth)
            else:
                result = compute_score(solution_str=solution, ground_truth=ground_truth)
            score = float(result.get("acc", result.get("score", 0.0))) if isinstance(result, dict) else float(result)
            pos = self._last_response_position(item, response_length)
            reward_tensor[i, pos] = score
            extras["acc"].append(score)
            extras["pred"].append(result.get("pred", "") if isinstance(result, dict) else "")

        metrics = {"external/accuracy": sum(extras["acc"]) / max(len(extras["acc"]), 1)}
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(extras), "metrics": metrics}
        return reward_tensor
