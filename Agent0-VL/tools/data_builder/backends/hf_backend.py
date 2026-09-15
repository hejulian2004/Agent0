"""Local Hugging Face/VLM Teacher backend.

Imports for ``torch`` and ``transformers`` are lazy so Phase 0 schema/parser
tests and API-only data generation do not load a local model or require a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import (
    GenerationChunk,
    TeacherBackend,
    TeacherBackendError,
    TeacherConfig,
    TeacherRole,
    normalize_messages,
    pil_images,
)


class LocalHFBackend(TeacherBackend):
    """Generate one response with a local Transformers VLM checkpoint."""

    backend_name = "hf"

    def __init__(
        self,
        config: TeacherConfig,
        *,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(config)
        if config.normalized_backend != self.backend_name:
            raise ValueError("LocalHFBackend requires an hf config")
        config.validate()
        self.processor = processor
        self.model = model
        self._torch: Any | None = None

    def _load(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise TeacherBackendError(
                "Local HF Teacher requires torch and transformers; "
                "install requirements.txt or use the API backend"
            ) from exc

        assert self.config.checkpoint is not None
        model_class = self._resolve_model_class()
        model_kwargs: dict[str, Any] = {
            "revision": self.config.revision,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.revision is None:
            model_kwargs.pop("revision")
        torch_dtype = self._resolve_dtype(torch)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if self.config.device_map is not None:
            model_kwargs["device_map"] = self.config.device_map

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.checkpoint,
                revision=self.config.revision,
                trust_remote_code=self.config.trust_remote_code,
            )
            self.model = model_class.from_pretrained(
                self.config.checkpoint,
                **model_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - preserve provider detail
            raise TeacherBackendError(
                f"Failed to load local Teacher checkpoint "
                f"{self.config.checkpoint!r}: {exc}"
            ) from exc

        self._torch = torch
        if self.config.device_map is None:
            device = self._resolve_device(torch)
            self.model.to(device)

    @staticmethod
    def _resolve_model_class() -> Any:
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq

                return AutoModelForVision2Seq
            except ImportError as exc:  # pragma: no cover - version dependent
                raise TeacherBackendError(
                    "Installed transformers has no multimodal AutoModel class"
                ) from exc

    def _resolve_dtype(self, torch: Any) -> Any | None:
        dtype = self.config.dtype.strip().lower()
        if dtype in {"auto", "none"}:
            return None
        values = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype not in values:
            raise ValueError(
                f"Unsupported AGENT0_TEACHER_DTYPE {self.config.dtype!r}"
            )
        return values[dtype]

    def _resolve_device(self, torch: Any) -> Any:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def generate_next(
        self,
        context: Sequence[Mapping[str, Any]] | Any,
        role: TeacherRole,
        images: Sequence[Any] | None = None,
    ) -> GenerationChunk:
        self._load()
        assert self.processor is not None
        assert self.model is not None
        assert self._torch is not None

        messages = normalize_messages(context)
        image_inputs = list(images) if images is not None else []
        image_values = pil_images(image_inputs)
        prompt = self._render_prompt(messages)
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if image_values:
            processor_kwargs["images"] = image_values
        try:
            inputs = self.processor(**processor_kwargs)
            inputs = self._move_inputs(inputs)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.max_tokens,
                "do_sample": self.config.do_sample,
            }
            if self.config.do_sample:
                generation_kwargs.update(
                    {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                    }
                )
            with self._torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
            input_ids = inputs.get("input_ids")
            prompt_length = int(input_ids.shape[-1]) if input_ids is not None else 0
            generated_ids = output_ids[:, prompt_length:]
            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            text = str(decoded[0]).strip() if decoded else ""
            if not text:
                raise TeacherBackendError("Local Teacher generated empty text")
        except TeacherBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve model detail
            raise TeacherBackendError(f"Local Teacher generation failed: {exc}") from exc

        generated_token_count = int(generated_ids.shape[-1])
        model_name = self.config.checkpoint or "local"
        return GenerationChunk(
            text=text,
            role=role,
            backend=self.backend_name,
            model=model_name,
            finish_reason=None,
            usage={"completion_tokens": generated_token_count},
            raw_response=None,
            metadata={
                "backend": self.backend_name,
                "checkpoint": self.config.checkpoint,
                "revision": self.config.revision,
                "requested_role": role,
                "image_count": len(image_values),
                "generation_parameters": {
                    "max_new_tokens": self.config.max_tokens,
                    "do_sample": self.config.do_sample,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                },
                "config": self.config.public_dict(),
            },
        )

    def _render_prompt(self, messages: list[dict[str, Any]]) -> str:
        apply_chat_template = getattr(self.processor, "apply_chat_template", None)
        if callable(apply_chat_template):
            return str(
                apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        from verl.prompts.agent0_templates import render_chat_messages

        return render_chat_messages(messages, add_generation_prompt=True)

    def _move_inputs(self, inputs: Any) -> Any:
        if self.config.device_map is not None:
            # ``device_map=auto`` still requires input IDs and image tensors on
            # the device hosting the model's input embedding layer.  Do not
            # assume CPU just because the processor created CPU tensors.
            device = getattr(self.model, "device", None)
            if device is None or str(device) == "meta":
                try:
                    device = next(self.model.parameters()).device
                except (AttributeError, StopIteration):
                    return inputs
        else:
            device = self._resolve_device(self._torch)
        if hasattr(inputs, "to"):
            return inputs.to(device)
        if isinstance(inputs, Mapping):
            return {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def close(self) -> None:
        # Explicitly release references so a builder can switch backends in one
        # process without retaining a full local VLM in memory.
        self.model = None
        self.processor = None
        self._torch = None
