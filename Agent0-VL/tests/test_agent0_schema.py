from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.data_builder.schema import (
    MAX_STEPS,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    ConversationState,
    ExecutionFailureKind,
    GenerationBoundary,
    ImageAsset,
    ObservationEvent,
    RawDatasetProbeResult,
    SolverTurnType,
    SupervisionRecord,
    TerminationReason,
    TrajectoryRecord,
    TrajectoryStep,
    canonical_relative_path,
    clone_for_rollout,
    content_fingerprint,
    has_images,
    probe_fingerprint,
    tool_success_flags,
    verification_id,
    acquisition_id,
)


class Agent0SchemaTests(unittest.TestCase):
    def test_schema_versions(self):
        self.assertEqual(SCHEMA_VERSION, "agent0vl.dataset.v1")
        self.assertEqual(PROTOCOL_VERSION, "agent0vl.protocol.v1")
        self.assertEqual(SCORER_VERSION, "agent0vl.scorer.v1")

    def test_step_indices_are_one_based_and_bounded(self):
        with self.assertRaises(ValueError):
            TrajectoryStep(step_index=0, solver_text="bad")
        with self.assertRaises(ValueError):
            GenerationBoundary(0, "solver", "s", "t")
        steps = [
            TrajectoryStep(
                step_index=index,
                solver_text="reasoning",
                turn_type=SolverTurnType.REASONING,
            )
            for index in range(1, MAX_STEPS + 1)
        ]
        trajectory = TrajectoryRecord(
            trajectory_id="trajectory-1",
            task_id="task-1",
            stage="stage2",
            generator="fixture",
            sandbox_backend="fixture",
            steps=steps,
            num_steps=MAX_STEPS,
            termination_reason=TerminationReason.MAX_STEPS_WITHOUT_FINAL_ANSWER,
        )
        trajectory.validate()

    def test_tool_success_has_any_and_all_semantics(self):
        self.assertEqual(tool_success_flags([]), (False, False))
        self.assertEqual(tool_success_flags([True, True]), (True, True))
        self.assertEqual(tool_success_flags([True, False]), (True, False))
        self.assertEqual(tool_success_flags([False, False]), (False, False))

        step = TrajectoryStep(
            step_index=1,
            solver_text="tool",
            turn_type=SolverTurnType.TOOL,
            call_execution_success=[True, False],
        )
        self.assertTrue(step.step_tool_success_any)
        self.assertFalse(step.step_tool_success_all)

    def test_content_fingerprint_is_inventory_only(self):
        inventory = [
            {"path": "images\\a.png", "size": 3, "sha256": "AAA"},
            {"path": "data.json", "size": 10, "sha256": "BBB"},
        ]
        same_inventory_different_order = list(reversed(inventory))
        self.assertEqual(
            content_fingerprint(inventory),
            content_fingerprint(same_inventory_different_order),
        )
        self.assertNotEqual(
            content_fingerprint(inventory),
            content_fingerprint(
                [
                    {"path": "images/a.png", "size": 4, "sha256": "AAA"},
                    {"path": "data.json", "size": 10, "sha256": "BBB"},
                ]
            ),
        )

        raw_fp = content_fingerprint(inventory)
        first = acquisition_id("demo", "source-a", "revision-a", raw_fp)
        second = acquisition_id("demo", "source-b", "revision-b", raw_fp)
        self.assertNotEqual(first, second)
        self.assertNotEqual(
            verification_id("acq", "probe-a", "probe.v1", "verification.v1", "code-a"),
            verification_id("acq", "probe-b", "probe.v2", "verification.v1", "code-b"),
        )

    def test_probe_fingerprint_contains_details_not_only_counts(self):
        first = RawDatasetProbeResult(
            sample_count=2,
            split_names=["train"],
            splits={"train": 2},
            image_count=2,
            missing_images=["images/a.png"],
            decode_failures=["images/b.png"],
            duplicate_groups=[["images/c.png", "images/d.png"]],
            metadata_fields=["question"],
        )
        second = RawDatasetProbeResult(
            sample_count=2,
            split_names=["train"],
            splits={"train": 2},
            image_count=2,
            missing_images=["images/other.png"],
            decode_failures=["images/b.png"],
            duplicate_groups=[["images/c.png", "images/d.png"]],
            metadata_fields=["question"],
        )
        self.assertNotEqual(probe_fingerprint(first), probe_fingerprint(second))
        self.assertEqual(first.decode_failure_count, 1)
        self.assertEqual(first.duplicate_file_count, 2)

    def test_canonical_relative_path_is_portable_and_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                canonical_relative_path(root / "images" / "a.png", root),
                "images/a.png",
            )
            self.assertEqual(
                canonical_relative_path("images\\sub\\a.png", root),
                "images/sub/a.png",
            )
            self.assertEqual(
                canonical_relative_path("e\u0301.png", root),
                "é.png",
            )
            with self.assertRaises(ValueError):
                canonical_relative_path("../outside.png", root)
            with self.assertRaises(ValueError):
                canonical_relative_path(root.parent / "outside.png", root)

            external = root.parent / "outside-target.txt"
            external.write_text("outside", encoding="utf-8")
            link = root / "external-link.txt"
            try:
                link.symlink_to(external)
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaises(ValueError):
                    canonical_relative_path(link, root)

    def test_supervision_record_has_single_assistant_target(self):
        record = SupervisionRecord(
            record_id="record-1",
            record_type="verifier",
            source_trajectory_id="trajectory-1",
            source_step_index=1,
            stage="stage2",
            messages=[
                {"role": "user", "content": "context"},
                {"role": "assistant", "content": "target"},
            ],
        )
        record.validate_single_assistant_target()

        invalid = SupervisionRecord(
            record_id="record-2",
            record_type="verifier",
            source_trajectory_id="trajectory-1",
            source_step_index=1,
            stage="stage2",
            messages=[
                {"role": "assistant", "content": "historical"},
                {"role": "user", "content": "request"},
                {"role": "assistant", "content": "target"},
            ],
        )
        with self.assertRaises(ValueError):
            invalid.validate_single_assistant_target()

    def test_conversation_state_is_deep_copied_for_rollouts(self):
        original = ConversationState(
            messages=[{"role": "user", "content": "question"}],
            original_images=[ImageAsset("input", "sha-input")],
            derived_images=[],
            observation_events=[],
            boundaries=[GenerationBoundary(1, "solver", "state-1", "text-1")],
        )
        group = [clone_for_rollout(original) for _ in range(8)]
        group[0].messages.append({"role": "assistant", "content": "changed"})
        group[0].observation_events.append(
            ObservationEvent(step_index=1, stdout="changed")
        )
        group[0].boundaries.append(
            GenerationBoundary(1, "verifier", "state-2", "text-2")
        )
        group[0].derived_images.append(ImageAsset("crop", "sha-crop"))

        for peer in group[1:]:
            self.assertEqual(peer.messages, original.messages)
            self.assertEqual(peer.observation_events, original.observation_events)
            self.assertEqual(peer.boundaries, original.boundaries)
            self.assertEqual(peer.derived_images, original.derived_images)

    def test_has_images_handles_none_sequences_and_numpy(self):
        self.assertFalse(has_images(None))
        self.assertFalse(has_images([]))
        self.assertFalse(has_images(()))
        self.assertTrue(has_images(["image"]))
        self.assertTrue(has_images("image-path"))
        try:
            import numpy as np
        except ImportError:
            return
        self.assertFalse(has_images(np.array([])))
        self.assertTrue(has_images(np.array([1])))


if __name__ == "__main__":
    unittest.main()
