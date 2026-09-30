"""Unit and acceptance tests for the 4 baseline execution modes in HJL."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from hjl.engine import HJLEngine
from hjl.model_caller import (
    GenericVerificationResult,
    MockHJLModelCaller,
    ReactAction,
    ReactDecisionResult,
    ScriptedHJLModelCaller,
)
from hjl.state import HJLState


class TestHJLBaselines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img = Image.new("RGB", (80, 80), color=(150, 150, 150))
        img.save(self.tmp.name)
        self.image_path = self.tmp.name
        self.engine = HJLEngine(mock=True)

    def tearDown(self):
        Path(self.image_path).unlink(missing_ok=True)

    def test_mode_1_direct(self):
        result = self.engine.run_direct(self.image_path)
        self.assertEqual(result["mode"], "direct")
        self.assertIn("is_anomaly", result)
        self.assertIn("conclusion", result)
        self.assertEqual(result["total_steps"], 1)

    def test_mode_2_react(self):
        result = self.engine.run_react(self.image_path, max_steps=3)
        self.assertEqual(result["mode"], "react")
        self.assertIn("observations", result)
        self.assertGreaterEqual(result["tool_cost"], 1)

    def test_mode_3_react_verifier(self):
        result = self.engine.run_react_verifier(self.image_path, max_steps=3)
        self.assertEqual(result["mode"], "react_verifier")
        self.assertIn("verifier_passed", result)
        self.assertIn("verifier_feedback", result)

    def test_mode_4_hjl(self):
        state = self.engine.run_hjl(self.image_path, max_steps=4)
        self.assertIsInstance(state, HJLState)
        self.assertIsNotNone(state.final_prediction)
        self.assertIn("is_anomaly", state.final_prediction)
        self.assertIn("conclusion", state.final_prediction)
        self.assertGreaterEqual(state.current_step, 1)

    def test_acceptance_react_transformed_image_and_rollback(self):
        """ReAct agent must observe the transformed crop image on subsequent steps, and rollback on failure."""
        recorded_images: list[str] = []

        class TrackingCaller(MockHJLModelCaller):
            def react_step(self, history, image_path, enabled_tools, *args, **kwargs):
                recorded_images.append(image_path)
                if not history:
                    # Step 1: Crop [10, 10, 30, 30]
                    return ReactDecisionResult(
                        action=ReactAction.TOOL_CALL,
                        tool_name="crop_region",
                        tool_arguments={"bbox": [10, 10, 30, 30]},
                    )
                elif len(history) == 1:
                    # Step 2: Failed crop with invalid bbox -> triggers rollback
                    return ReactDecisionResult(
                        action=ReactAction.TOOL_CALL,
                        tool_name="crop_region",
                        tool_arguments={"bbox": [1000, 1000, 2000, 2000]},
                    )
                # Step 3: Finish
                return ReactDecisionResult(
                    action=ReactAction.FINISH,
                    is_anomaly=False,
                    final_answer="Normal part after verification.",
                )

        engine = HJLEngine(model_caller=TrackingCaller(), mock=True)
        res = engine.run_react(self.image_path, max_steps=4)

        # 1. Step 1 saw original image
        self.assertEqual(recorded_images[0], str(Path(self.image_path).resolve()))

        # 2. Step 2 saw transformed crop image (different path)
        self.assertNotEqual(recorded_images[1], recorded_images[0])

        # 3. Observation history recorded failed tool with retriable/error
        self.assertEqual(len(res["observations"]), 2)
        failed_obs = res["observations"][1]
        self.assertFalse(failed_obs["success"])
        self.assertIsNotNone(failed_obs["error"])

    def test_react_verifier_skips_on_max_steps(self):
        """When ReAct reaches max_steps without finishing, verifier_passed must be None without model verification."""
        class InfiniteCaller(MockHJLModelCaller):
            def react_step(self, history, image_path, enabled_tools, *args, **kwargs):
                return ReactDecisionResult(
                    action=ReactAction.TOOL_CALL,
                    tool_name="rotate_image",
                    tool_arguments={"angle": 90.0},
                )

        engine = HJLEngine(model_caller=InfiniteCaller(), mock=True)
        res = engine.run_react_verifier(self.image_path, max_steps=2)
        self.assertEqual(res["conclusion"], "UNRESOLVED")
        self.assertIsNone(res["is_anomaly"])
        self.assertIsNone(res["verifier_passed"])
        self.assertIn("Not applicable", res["verifier_feedback"])

    def test_live_engine_requires_api_key(self):
        """HJLEngine(mock=False) without AGENT0_RESPONSES_API_KEY must raise immediately."""
        old_val = os.environ.pop("AGENT0_RESPONSES_API_KEY", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                HJLEngine(mock=False)
            self.assertIn("AGENT0_RESPONSES_API_KEY", str(ctx.exception))
        finally:
            if old_val is not None:
                os.environ["AGENT0_RESPONSES_API_KEY"] = old_val

    @unittest.skip("Skipped while retrieve_normal_reference is commented out for general Agent0-VL")
    def test_react_runtime_injected_arguments_isolation(self):
        """ReAct agent-visible tool schemas must omit allow_synthetic, corpus_dir, and category, and runtime must inject them."""
        from hjl.tools_adapter import get_hjl_tool_definitions

        defs = get_hjl_tool_definitions(agent_visible=True)
        ret_def = next(d for d in defs if d["name"] == "retrieve_normal_reference")
        props = ret_def["parameters"]["properties"]
        self.assertNotIn("allow_synthetic", props)
        self.assertNotIn("corpus_dir", props)
        self.assertNotIn("category", props)
        self.assertEqual(props, {})
        self.assertEqual(ret_def["parameters"]["required"], [])

        # Verify Engine unconditionally overrides even if agent outputs a hallucinated category
        class HallucinatingRetrieveCaller(MockHJLModelCaller):
            def react_step(self, history, image_path, enabled_tools, *args, **kwargs):
                if not history:
                    return ReactDecisionResult(
                        action=ReactAction.TOOL_CALL,
                        tool_name="retrieve_normal_reference",
                        tool_arguments={"category": "wrong_hallucinated_category"},
                    )
                return ReactDecisionResult(action=ReactAction.FINISH, is_anomaly=False, final_answer="Done")

        engine = HJLEngine(model_caller=HallucinatingRetrieveCaller(), mock=True)
        res = engine.run_react(self.image_path, category="cable", max_steps=2)
        ret_obs = res["observations"][0]
        self.assertTrue(ret_obs["arguments"]["allow_synthetic"])
        self.assertEqual(ret_obs["arguments"]["category"], "cable")

        # Verify Engine properly injects when agent outputs empty arguments {}
        class CleanRetrieveCaller(MockHJLModelCaller):
            def react_step(self, history, image_path, enabled_tools, *args, **kwargs):
                if not history:
                    return ReactDecisionResult(
                        action=ReactAction.TOOL_CALL,
                        tool_name="retrieve_normal_reference",
                        tool_arguments={},
                    )
                return ReactDecisionResult(action=ReactAction.FINISH, is_anomaly=False, final_answer="Done")

        engine_clean = HJLEngine(model_caller=CleanRetrieveCaller(), mock=True)
        res_clean = engine_clean.run_react(self.image_path, category="metal_casting", max_steps=2)
        ret_obs_clean = res_clean["observations"][0]
        self.assertTrue(ret_obs_clean["arguments"]["allow_synthetic"])
        self.assertEqual(ret_obs_clean["arguments"]["category"], "metal_casting")


if __name__ == "__main__":
    unittest.main()
