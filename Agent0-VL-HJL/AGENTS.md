# AGENTS.md

This file provides architectural, operational, and memory conventions for AI agents (Claude Code, Cursor, Codex, OpenCode, etc.) working in this repository. It coordinates shared context with `CLAUDE.md` and persistent project memory in `docs/memory/`.

---

## 1. Project Overview & Multi-Agent Architecture

`Agent0-VL` is a self-evolving vision-language reasoning framework built on RL (PPO/GRPO via VERL) and OpenAI Responses API (`client.responses.create(...)`).

The system is organized around three complementary agent roles (The SERC Cycle):
- **Solver**: Conducts multi-turn tool-integrated reasoning steps, generating thoughts, tool calls, and final answers.
- **Verifier**: Evaluates intermediate reasoning steps and final trajectories, outputting structured critique and confidence scores.
- **Repairer (Self-Repair)**: Triggers when Verifier confidence falls below threshold ($\tau_c < 0.7$), emitting `PATCH` instructions, executing image context rollback (`context.rollback()`), and re-sampling solver segments.

---

## 2. Tool Architecture & Current Status

### A. Canonical Agent0-VL Tools (Active)
Defined in `agent0_protocol/tools.py`. These 9 general-domain tools are equipped and fully functional for multimodal benchmarks (MathVista, ChartQA, ArxivQA, etc.):
1. `python_exec`: Python sandbox with preloaded `math`, `np`, `Image`, `cv2`, `sp`, `RapidOCR`.
2. `crop_image`: Active image cropping; automatically passes updated visual pixels to subsequent turns.
3. `zoom_image`: Multi-scale image magnification (0.1x to 8.0x).
4. `rotate_image`: Geometric rotation (-360° to 360°).
5. `ocr`: Text and polygon extraction via RapidOCR (PP-OCRv6).
6. `plot_parser`: Chart label extraction and location parsing.
7. `visual_analyzer`: Global color, dimension, and dark region detection.
8. `object_detector`: Open-vocabulary / COCO object detection via local `yolo26n.pt`.
9. `retrieve`: Keyword search over knowledge documents in `data/knowledge/`.

### B. HJL Industrial Anomaly Tools (Temporarily Dormant)
Defined in `hjl/tools_adapter.py`. Currently deregistered/commented out to keep the focus on general Agent0-VL improvements.
- **To restore HJL tools**: See the step-by-step restoration guide in [`docs/memory/hjl_tools_restoration.md`](docs/memory/hjl_tools_restoration.md).
- Reference commits: `5fd9f74` (last full working state) and `8843d40` (MVTec AD dataset setup).

---

## 3. Operational Rules & Constraints

1. **Single Agent Execution**:
   - Always execute directly as a single agent in the main session. Do not spawn background subagents unless explicitly instructed.
2. **Responses API Only**:
   - All remote model calls must use `/v1/responses`. Legacy Chat Completions endpoints are unsupported.
   - Credentials must come from environment variables (`AGENT0_RESPONSES_API_KEY`, etc.) or `.env`, never written to `config.yaml` or tracked files.
3. **Multimodal Tool Image Feedback**:
   - Whenever an image tool executes (`crop_image`, `zoom_image`, `rotate_image`), the generated image must be encoded as a data URL and passed as `input_image` in `function_call_output` so the model's vision encoder observes the transformed pixels.
4. **Environment Integrity**:
   - Python 3.12 managed via `uv` in `.venv/bin/python`.
   - Never downgrade vLLM (pinned to 0.30.0).

---

## 4. Key References & Data Intercommunication

- **General Guidance & Hardware Constraints**: [`CLAUDE.md`](CLAUDE.md)
- **HJL Industrial Architecture**: [`hjl/README.md`](hjl/README.md)
- **HJL Tool Restoration Checklist**: [`docs/memory/hjl_tools_restoration.md`](docs/memory/hjl_tools_restoration.md)
- **Global Configuration**: [`config.yaml`](config.yaml)
- **Environment Template**: [`.env.example`](.env.example)
