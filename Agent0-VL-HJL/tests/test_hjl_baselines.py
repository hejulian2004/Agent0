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
            def react_step(self, history, image_path, enabled_tools):
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
            def react_step(self, history, image_path, enabled_tools):
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


if __name__ == "__main__":
    unittest.main()
