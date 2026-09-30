"""Configuration dataclass and YAML parser for HJL."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

logger = logging.getLogger(__name__)


@dataclass
class HJLConfig:
    """Unified configuration for Hierarchical Judgment Loop."""

    enabled: bool = False
    max_steps: int = 8
    anomaly_threshold: float = 0.75
    normal_threshold: float = 0.20
    min_evidence_count: int = 1
    checkpoint_confidence_threshold: float = 0.80
    global_normal_confidence_threshold: float = 0.95
    reference_similarity_threshold: float = 0.80
    max_discovery_attempts: int = 2
    global_checkpoint: bool = True
    regional_checkpoint: bool = True
    evidence_checkpoint: bool = True
    failure_diagnosis: bool = True
    adaptive_stop: bool = True
    tool_failure_limit: int = 3
    enabled_tools: list[str] = field(
        default_factory=lambda: [
            # All HJL tools are currently deregistered in favor of canonical Agent0-VL tools
        ]
    )
    trajectory_output_dir: str = "outputs/hjl_trajectories"
    reference_corpus_dir: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HJLConfig":
        """Instantiate HJLConfig from a mapping/dict, ignoring extraneous keys."""
        hjl_dict = dict(data.get("hjl", data) if "hjl" in data else data)
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in hjl_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "HJLConfig":
        """Load configuration from a YAML file path or fallback to defaults."""
        config_path = Path(path) if path else Path("config.yaml")
        if not config_path.is_file():
            logger.debug(f"Config file not found at {config_path}, using default HJLConfig.")
            return cls()

        try:
            with config_path.open("r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or {}
            return cls.from_mapping(raw_data)
        except Exception as exc:
            logger.warning(f"Failed to parse config from {config_path}: {exc}. Using default HJLConfig.")
            return cls()
