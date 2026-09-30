"""Unit tests for HJL configuration loading and validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hjl.config import HJLConfig


class TestHJLConfig(unittest.TestCase):
    def test_default_config_values(self):
        config = HJLConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.max_steps, 8)
        self.assertEqual(config.anomaly_threshold, 0.75)
        self.assertEqual(config.normal_threshold, 0.20)
        self.assertEqual(config.checkpoint_confidence_threshold, 0.80)
        self.assertEqual(config.global_normal_confidence_threshold, 0.95)
        self.assertEqual(config.reference_similarity_threshold, 0.80)
        self.assertEqual(config.max_discovery_attempts, 2)
        self.assertEqual(config.enabled_tools, [])

    def test_config_from_yaml_file(self):
        yaml_content = """
hjl:
  enabled: true
  max_steps: 12
  anomaly_threshold: 0.82
  normal_threshold: 0.18
  reference_similarity_threshold: 0.78
  max_discovery_attempts: 3
  trajectory_output_dir: custom_outputs
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)

        try:
            config = HJLConfig.from_yaml(tmp_path)
            self.assertEqual(config.max_steps, 12)
            self.assertEqual(config.anomaly_threshold, 0.82)
            self.assertEqual(config.normal_threshold, 0.18)
            self.assertEqual(config.reference_similarity_threshold, 0.78)
            self.assertEqual(config.max_discovery_attempts, 3)
            self.assertEqual(config.trajectory_output_dir, "custom_outputs")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_config_from_nonexistent_yaml_falls_back(self):
        config = HJLConfig.from_yaml("nonexistent_config_file_123.yaml")
        self.assertEqual(config.max_steps, 8)
        self.assertEqual(config.anomaly_threshold, 0.75)


if __name__ == "__main__":
    unittest.main()
