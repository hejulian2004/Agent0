# HJL Industrial Anomaly Detection Tools Restoration Guide

This document records the exact steps, code modifications, and commit baselines required to re-enable the **HJL (Hierarchical Judgment Loop)** industrial visual anomaly detection tools.

Currently, all HJL-specific tools are deregistered/commented out (in commit `a41ca2d` and `20913e5`) to focus engineering and research efforts on the **general Agent0-VL vision-language self-evolving reasoning** framework. All underlying implementation functions (`_locate_or_create_reference_image`, `_compare_images_simple`, etc.) are completely preserved in the codebase.

---

## 1. Commit Baselines

- **`5fd9f74`**: Full operational baseline where both canonical Agent0-VL tools and all HJL tools were 100% functional and passing unit tests.
- **`8843d40`**: Introduced `scripts/setup_mvtec.py` for automated MVTec AD dataset setup from `/mnt/d/Triad`.
- **`a41ca2d`**: Deregistered HJL-specific tools to focus purely on general Agent0-VL canonical tools.

---

## 2. Step-by-Step Restoration Checklist

### Step 1: Re-enable Tool Schemas in `hjl/tools_adapter.py`
In `hjl/tools_adapter.py`, uncomment the schemas inside `HJL_TOOL_DEFINITIONS`:
- `crop_region`
- `zoom_region`
- `rotate_image`
- `retrieve_normal_reference`
- `compare_with_reference`
- `localize_candidate`

In `execute_adapted_tool(name, arguments, context, registry)`:
- Remove the top early-return deregistration guard:
  ```python
  # Remove this guard:
  # if name not in tool_defs:
  #     return ToolResult(success=False, error=f"HJL tool '{name}' is currently deregistered...")
  ```
- Uncomment the tool execution branches for `retrieve_normal_reference` and `compare_with_reference`.

---

### Step 2: Re-enable Configuration in `config.yaml`
In `config.yaml` under the `hjl:` section:
```yaml
hjl:
  enabled: true
  max_steps: 8
  anomaly_threshold: 0.75
  normal_threshold: 0.20
  min_evidence_count: 1
  checkpoint_confidence_threshold: 0.80
  global_normal_confidence_threshold: 0.95
  global_checkpoint: true
  regional_checkpoint: true
  evidence_checkpoint: true
  failure_diagnosis: true
  adaptive_stop: true
  tool_failure_limit: 3
  enabled_tools:
    - crop_region
    - zoom_region
    - rotate_image
    - retrieve_normal_reference
    - compare_with_reference
    - localize_candidate
  trajectory_output_dir: outputs/hjl_trajectories
```

---

### Step 3: Re-enable Python Dataclass in `hjl/config.py`
In `hjl/config.py` inside `HJLConfig`:
```python
    enabled: bool = True
    ...
    enabled_tools: list[str] = field(
        default_factory=lambda: [
            "crop_region",
            "zoom_region",
            "rotate_image",
            "retrieve_normal_reference",
            "compare_with_reference",
            "localize_candidate",
        ]
    )
```

---

### Step 4: Reactivate Test Suite in `tests/`
Remove `@unittest.skip("Skipped while HJL tools are deregistered...")` from:
1. `tests/test_hjl_tools.py`: `class TestHJLTools`
2. `tests/test_hjl_graph.py`: `class TestHJLGraph`
3. `tests/test_hjl_baselines.py`: `class TestHJLBaselines`
4. `tests/test_hjl_trajectory.py`: `test_to_canonical_trajectory_schema_conformance`, `test_to_canonical_trajectory_faithful_failures`, `test_to_canonical_trajectory_agent_visible_schemas_and_isolation`
5. `tests/test_hjl_config.py`: restore `self.assertTrue(config.enabled)` and check `self.assertIn("crop_region", config.enabled_tools)`

Run pytest to verify full pass:
```bash
.venv/bin/python -m pytest tests
```

---

## 3. Data & Benchmark Artifacts
- **Dataset Setup**: Run `.venv/bin/python scripts/setup_mvtec.py` to recreate symbolic links for MVTec AD from `/mnt/d/Triad/evaluation/mvtec` into `data/mvtec/`.
- **Batch Evaluation**: Run `.venv/bin/python scripts/eval_mvtec_batch.py --mode hjl` to run the state machine over `data/mvtec/index.jsonl`.
