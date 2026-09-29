"""Unit tests for HJL tool adapters, anti-leakage validation, and context isolation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from agent0_protocol.tools import ToolExecutionContext, get_tool_registry
from hjl.tools_adapter import (
    ToolResult,
    execute_adapted_tool,
    validate_reference_metadata,
)


class TestHJLTools(unittest.TestCase):
    def setUp(self):
        # Create a small dummy image for testing
        self.tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img = Image.new("RGB", (60, 40), color="blue")
        img.save(self.tmp.name)
        self.image_path = Path(self.tmp.name)
        self.context = ToolExecutionContext(image=self.image_path)

        # Create a distinct reference image
        self.ref_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        ref_img = Image.new("RGB", (60, 40), color="red")
        ref_img.save(self.ref_tmp.name)
        self.ref_path = Path(self.ref_tmp.name)

    def tearDown(self):
        self.context.close()
        self.image_path.unlink(missing_ok=True)
        self.ref_path.unlink(missing_ok=True)

    def test_crop_region_tool_result(self):
        result = execute_adapted_tool("crop_region", {"bbox": [0, 0, 20, 20]}, self.context)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output_path)
        self.assertEqual(result.metadata["image_size"], [20, 20])
        self.assertIsInstance(result, ToolResult)

    def test_zoom_region_tool_result(self):
        result = execute_adapted_tool("zoom_region", {"scale": 2.0}, self.context)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["image_size"], [120, 80])

    def test_rotate_image_tool_result(self):
        result = execute_adapted_tool("rotate_image", {"angle": 90.0}, self.context)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["image_size"], [40, 60])

    def test_retrieve_normal_reference_tool_result(self):
        result = execute_adapted_tool("retrieve_normal_reference", {"category": "metal_casting"}, self.context)
        self.assertTrue(result.success)
        self.assertIn("reference_path", result.metadata)
        self.assertEqual(result.metadata.get("split"), "train")
        self.assertTrue(result.metadata.get("is_normal"))

    def test_reference_anti_leakage_runtime_validation(self):
        """Reference metadata must strictly be train-split normal samples."""
        valid_meta = {"split": "train", "is_normal": True, "category": "metal"}
        validate_reference_metadata(valid_meta)  # Should pass

        # Test leakage rejection
        with self.assertRaises(ValueError) as ctx:
            validate_reference_metadata({"split": "test", "is_normal": True})
        self.assertIn("Reference leakage detected", str(ctx.exception))

        # Abnormal reference rejection
        with self.assertRaises(ValueError) as ctx:
            validate_reference_metadata({"split": "train", "is_normal": False})
        self.assertIn("confirmed normal", str(ctx.exception))

    def test_compare_with_reference_tool_result(self):
        result = execute_adapted_tool(
            "compare_with_reference",
            {"reference_path": str(self.ref_path)},
            self.context,
        )
        self.assertTrue(result.success)
        self.assertIn("similarity", result.metadata)
        self.assertGreaterEqual(result.metadata["similarity"], 0.0)

    def test_compare_with_reference_rejects_missing_path(self):
        result = execute_adapted_tool("compare_with_reference", {}, self.context)
        self.assertFalse(result.success)
        self.assertIn("Missing required argument 'reference_path'", result.error or "")

    def test_compare_with_reference_rejects_self_comparison(self):
        """Comparing the active image against itself must be strictly rejected."""
        curr_path = self.context["current_image_path"]
        result = execute_adapted_tool(
            "compare_with_reference",
            {"reference_path": curr_path},
            self.context,
        )
        self.assertFalse(result.success)
        self.assertIn("Self-comparison rejected", result.error or "")

    def test_unknown_tool_fails_gracefully(self):
        result = execute_adapted_tool("nonexistent_tool", {}, self.context)
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.error or "")

    def test_tool_context_rollback_on_failure(self):
        initial_path = self.context["current_image_path"]
        self.context.save_checkpoint()

        # Execute valid crop
        res = execute_adapted_tool("crop_region", {"bbox": [5, 5, 25, 25]}, self.context)
        self.assertTrue(res.success)
        self.assertNotEqual(self.context["current_image_path"], initial_path)

        # Rollback restores initial path
        self.context.rollback()
        self.assertEqual(self.context["current_image_path"], initial_path)


if __name__ == "__main__":
    unittest.main()
