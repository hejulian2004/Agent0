"""Unit tests for the 4 baseline execution modes in HJL."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from hjl.engine import HJLEngine
from hjl.state import HJLState


class TestHJLBaselines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img = Image.new("RGB", (80, 80), color=(150, 150, 150))
        img.save(self.tmp.name)
        self.image_path = self.tmp.name
        self.engine = HJLEngine()

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


if __name__ == "__main__":
    unittest.main()
