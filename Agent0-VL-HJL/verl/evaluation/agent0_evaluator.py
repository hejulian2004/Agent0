"""Agent0 evaluator driven by canonical Responses trajectories."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime
from agent0_protocol.tools import get_tool_registry
from agent0_protocol.verifier import verify_trajectory
from tools.data_builder.backends.base import image_data_url
from verl.evaluation.metrics import (
    EvaluationMetrics,
    aggregate_metrics,
    compute_exact_match,
    format_metrics_table,
    save_metrics_to_csv,
)
from verl.prompts.agent0_templates import assistant_text, render_solver_request, render_system_prompt


@dataclass
class EvaluatorConfig:
    output_dir: str = "./evaluation_results"
    batch_size: int = 16
    save_per_sample_results: bool = True
    save_intermediate: bool = True
    intermediate_save_interval: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Agent0Evaluator:
    def __init__(
        self,
        config: EvaluatorConfig | dict[str, Any] | None = None,
        *,
        model_path: str | None = None,
        responses_config: ResponsesConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = EvaluatorConfig(**config) if isinstance(config, dict) else (config or EvaluatorConfig())
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self.registry = get_tool_registry()
        cfg = responses_config or ResponsesConfig.from_env()
        target_model = model_path or kwargs.get("model") or cfg.model
        if target_model != cfg.model:
            cfg = ResponsesConfig(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                model=target_model,
                timeout_seconds=cfg.timeout_seconds,
                max_retries=cfg.max_retries,
                max_tool_rounds=cfg.max_tool_rounds,
                max_output_tokens=cfg.max_output_tokens,
            )
        # The constructor probes text, images, function tools, result handoff and
        # two successive function rounds before any benchmark is evaluated.
        self.runtime = ResponsesRuntime(cfg, self.registry)

    def evaluate_sample(self, sample: dict[str, Any], sample_idx: int = 0) -> dict[str, Any]:
        started = time.time()
        question = str(sample.get("question", sample.get("prompt", "")))
        ground_truth = str(sample.get("ground_truth", sample.get("answer", "")))
        image = sample.get("image_pil") or sample.get("image_path") or sample.get("image")
        content: str | list[dict[str, Any]] = render_solver_request(question)
        if image is not None:
            if isinstance(image, str) and not image.startswith("data:") and not os.path.isfile(image):
                raise FileNotFoundError(image)
            content = [
                {"type": "input_text", "text": render_solver_request(question)},
                {"type": "input_image", "image_url": image_data_url(image)},
            ]
        trajectory = self.runtime.run(
            [
                {"type": "message", "role": "system", "content": render_system_prompt()},
                {"type": "message", "role": "user", "content": content},
            ],
            metadata={"sample_id": str(sample.get("id", sample_idx))},
        )
        verification = verify_trajectory(trajectory, self.registry)
        prediction = assistant_text(trajectory.items) or ""
        result = {
            "sample_id": str(sample.get("id", sample.get("sample_id", sample_idx))),
            "data_source": sample.get("data_source", "unknown"),
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "is_correct": compute_exact_match(prediction, ground_truth),
            "verification": asdict(verification),
            "num_steps": sum(item["type"] == "message" and item.get("role") == "assistant"
                             for item in trajectory.items),
            "num_repairs": int(trajectory.metadata.get("num_repairs", 0)),
            "calls": [item for item in trajectory.items if item["type"] == "function_call"],
            "trajectory": trajectory.to_dict(),
            "elapsed_time": time.time() - started,
        }
        if not verification.valid:
            raise RuntimeError(f"invalid semantic trajectory: {verification.issues}")
        return result

    def evaluate_dataset(
        self,
        dataset: list[dict[str, Any]],
        output_path: str | None = None,
        resume_from: str | None = None,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        if resume_from:
            with open(resume_from, encoding="utf-8") as handle:
                results = list(json.load(handle)["results"])
        for index, sample in tqdm(enumerate(dataset[len(results):], start=len(results)), total=len(dataset)):
            try:
                results.append(self.evaluate_sample(sample, index))
            except Exception as exc:
                results.append({
                    "sample_id": str(sample.get("id", index)),
                    "data_source": sample.get("data_source", "unknown"),
                    "ground_truth": str(sample.get("ground_truth", sample.get("answer", ""))),
                    "prediction": "",
                    "is_correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if self.config.save_intermediate and (index + 1) % self.config.intermediate_save_interval == 0:
                self._write_json(Path(self.config.output_dir) / "intermediate.json", {"results": results})
        metrics = aggregate_metrics(results)
        output = {
            "config": self.config.to_dict(),
            "metrics": metrics.to_dict(),
            "results": results if self.config.save_per_sample_results else [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if output_path:
            target = Path(output_path)
            self._write_json(target, output)
            save_metrics_to_csv(metrics, str(target.with_name(target.stem + "_metrics.csv")))
        print(format_metrics_table(metrics))
        return output

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def evaluate_benchmarks(
        self,
        benchmark_names: list[str],
        data_dir: str,
        output_dir: str | None = None,
    ) -> dict[str, EvaluationMetrics]:
        root = Path(output_dir or self.config.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        all_metrics: dict[str, EvaluationMetrics] = {}
        for name in benchmark_names:
            from verl.evaluation.benchmarks import get_benchmark

            dataset = get_benchmark(name, data_dir).load_data()
            if not dataset:
                raise RuntimeError(f"benchmark {name!r} has no samples")
            output = self.evaluate_dataset(dataset, str(root / f"{name}_results.json"))
            all_metrics[name] = EvaluationMetrics(**output["metrics"])
        self._write_json(root / "combined_results.json", {
            "benchmarks": {name: metrics.to_dict() for name, metrics in all_metrics.items()}
        })
        return all_metrics
