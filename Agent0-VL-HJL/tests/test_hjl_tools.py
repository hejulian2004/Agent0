"""Unit tests for HJL tool adapters, anti-leakage validation, and context isolation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from agent0_protocol.tools import ToolExecutionContext, get_tool_registry
from hjl.tools_adapter import (
    ToolResult,
    _is_retriable_error,
    execute_adapted_tool,
    validate_reference_metadata,
)


class TestHJLTools(unittest.TestCase):
    def setUp(self):
        # Create a small dummy image for testing (60x40)
        self.tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img = Image.new("RGB", (60, 40), color="blue")
        img.save(self.tmp.name)
        self.image_path = Path(self.tmp.name)
        self.context = ToolExecutionContext(image=self.image_path)
        self.context["original_image_path"] = str(self.image_path)

        # Create a distinct reference image (120x80) with different dimensions to test normalized bbox
        self.ref_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        ref_img = Image.new("RGB", (120, 80), color="red")
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
        self.assertIn("retriable", result.to_dict())

    def test_crop_region_requires_bbox_and_valid_geometry(self):
        # Missing bbox
        res1 = execute_adapted_tool("crop_region", {}, self.context)
        self.assertFalse(res1.success)
        self.assertFalse(res1.retriable)
        self.assertIn("'bbox'", res1.error or "")

        # Invalid geometry: x1 >= x2
        res2 = execute_adapted_tool("crop_region", {"bbox": [20, 0, 10, 20]}, self.context)
        self.assertFalse(res2.success)
        self.assertFalse(res2.retriable)
        self.assertIn("Invalid bbox dimensions", res2.error or "")

    def test_zoom_region_tool_result(self):
        result = execute_adapted_tool("zoom_region", {"scale": 2.0}, self.context)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["image_size"], [120, 80])

    def test_zoom_region_requires_positive_scale(self):
        # Missing scale
        res1 = execute_adapted_tool("zoom_region", {}, self.context)
        self.assertFalse(res1.success)
        self.assertFalse(res1.retriable)

        # Zero or negative scale
        res2 = execute_adapted_tool("zoom_region", {"scale": -1.5}, self.context)
        self.assertFalse(res2.success)
        self.assertFalse(res2.retriable)

    def test_rotate_image_tool_result(self):
        result = execute_adapted_tool("rotate_image", {"angle": 90.0}, self.context)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["image_size"], [40, 60])

    def test_rotate_image_requires_angle(self):
        res = execute_adapted_tool("rotate_image", {}, self.context)
        self.assertFalse(res.success)
        self.assertFalse(res.retriable)

    def test_retrieve_normal_reference_tool_result(self):
        result = execute_adapted_tool(
            "retrieve_normal_reference",
            {"category": "metal_casting", "allow_synthetic": True},
            self.context,
        )
        self.assertTrue(result.success)
        self.assertIn("reference_path", result.metadata)
        self.assertEqual(result.metadata.get("split"), "train")
        self.assertTrue(result.metadata.get("is_normal"))

    def test_retrieve_normal_reference_rejects_missing_in_live_mode(self):
        # In live mode (allow_synthetic=False), if reference corpus does not have the file, fail cleanly
        result = execute_adapted_tool(
            "retrieve_normal_reference",
            {"category": "nonexistent_category_12345", "allow_synthetic": False},
            self.context,
        )
        self.assertFalse(result.success)
        self.assertFalse(result.retriable)
        self.assertIn("No train-normal reference available", result.error or "")

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
            {
                "reference_path": str(self.ref_path),
                "normalized_bbox": [0.1, 0.1, 0.5, 0.5],
                "rotation_deg": 90.0,
            },
            self.context,
        )
        self.assertTrue(result.success)
        self.assertIn("similarity", result.metadata)
        self.assertGreaterEqual(result.metadata["similarity"], 0.0)

    def test_compare_with_reference_rejects_missing_path(self):
        result = execute_adapted_tool("compare_with_reference", {}, self.context)
        self.assertFalse(result.success)
        self.assertFalse(result.retriable)
        self.assertIn("'reference_path'", result.error or "")

    def test_compare_with_reference_rejects_self_comparison(self):
        """Comparing the active image against itself must be strictly rejected."""
        curr_path = self.context["current_image_path"]
        result = execute_adapted_tool(
            "compare_with_reference",
            {"reference_path": curr_path},
            self.context,
        )
        self.assertFalse(result.success)
        self.assertFalse(result.retriable)
        self.assertIn("Self-comparison rejected", result.error or "")

    def test_localize_candidate_returns_empty_when_no_defect(self):
        """Clean solid image has no dark boxes; must return empty candidate_regions list without fake center fallback."""
        result = execute_adapted_tool("localize_candidate", {"use_original": True}, self.context)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata.get("candidate_regions"), [])

    def test_unknown_tool_fails_gracefully(self):
        result = execute_adapted_tool("nonexistent_tool", {}, self.context)
        self.assertFalse(result.success)
        self.assertFalse(result.retriable)
        self.assertIn("Unknown tool", result.error or "")

    def test_tool_context_rollback_on_failure(self):
        initial_path = self.context["current_image_path"]
        checkpoint = self.context.checkpoint()

        # Execute valid crop
        res = execute_adapted_tool("crop_region", {"bbox": [5, 5, 25, 25]}, self.context)
        self.assertTrue(res.success)
        self.assertNotEqual(self.context["current_image_path"], initial_path)

        # Rollback restores initial path
        self.context.rollback(checkpoint)
        self.assertEqual(self.context["current_image_path"], initial_path)

    def test_schema_validation_rejection_on_invalid_types(self):
        """JSON Schema validator must reject non-conforming primitive types without execution."""
        # bbox with string elements instead of integers
        res1 = execute_adapted_tool("crop_region", {"bbox": ["10", "10", "20", "20"]}, self.context)
        self.assertFalse(res1.success)
        self.assertFalse(res1.retriable)
        self.assertIn("Schema validation error", res1.error or "")

        # scale with string instead of number
        res2 = execute_adapted_tool("zoom_region", {"scale": "2.0"}, self.context)
        self.assertFalse(res2.success)
        self.assertFalse(res2.retriable)
        self.assertIn("Schema validation error", res2.error or "")

        # use_original with string "false" instead of boolean
        res3 = execute_adapted_tool("crop_region", {"bbox": [0, 0, 10, 10], "use_original": "false"}, self.context)
        self.assertFalse(res3.success)
        self.assertFalse(res3.retriable)
        self.assertIn("Schema validation error", res3.error or "")

    def test_retriable_error_classification(self):
        """Timeout and connection errors must be classified as retriable; logical errors as non-retriable."""
        self.assertTrue(_is_retriable_error("TimeoutError: tool call timed out after 30s"))
        self.assertTrue(_is_retriable_error("Connection reset by peer"))
        self.assertTrue(_is_retriable_error("Service temporarily unavailable"))

        self.assertFalse(_is_retriable_error("Missing required argument 'bbox'"))
        self.assertFalse(_is_retriable_error("Invalid bbox dimensions"))
        self.assertFalse(_is_retriable_error("Reference image not found at /path"))
        self.assertFalse(_is_retriable_error(None))


if __name__ == "__main__":
    unittest.main()
