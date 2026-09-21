import ast

from tools.sft_builder import build
from tools.sft_builder.merge_sft import audit_record
from verl.utils.sandbox.executor import ImagePathTransformer


class QueueTeacher:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, messages, images, system_prompt):
        del messages, images, system_prompt
        if not self.responses:
            raise AssertionError("fake teacher response queue was exhausted")
        return self.responses.pop(0)


def test_low_confidence_repair_stays_in_one_auditable_sft_row(monkeypatch):
    def fake_execute_python(code_blocks, sandbox_timeout=10.0, images=()):
        del sandbox_timeout, images
        return [
            {"success": True, "stdout": "2", "stderr": ""}
            for _ in code_blocks
        ]

    monkeypatch.setattr(build, "execute_python", fake_execute_python)
    teacher = QueueTeacher([
        "```python\nprint(2)\n```",
        "The result is <answer>\n\\boxed{2}\n</answer>",
        '{"step_index": 1, "score": 0.0, "confidence": 0.4, "critique": "uncertain", "tool_check": true}',
        '{"action": "PATCH", "target_step": 1, "patch_type": "code", "new_content": "print(2)", "justification": "recheck the calculation"}',
        "```python\nprint(2)\n```",
        '{"step_index": 1, "score": 1.0, "confidence": 0.95, "critique": "fixed", "tool_check": true}',
        "The result is <answer>\n\\boxed{2}\n</answer>",
        '{"step_index": 2, "score": 1.0, "confidence": 0.98, "critique": "final", "tool_check": true}',
    ])
    sample = {
        "stage": 2,
        "question": "Compute 1 + 1.",
        "images": [],
        "ground_truth": "2",
        "ground_truth_aliases": [],
    }

    records, stats = build.build_records(
        [sample],
        teacher,
        "solver system prompt",
        max_tasks=1,
        quality_profile="stage2",
    )

    assert stats.exported == 1
    assert stats.quality_failures == 0
    assert len(records) == 1
    row = records[0]
    assert audit_record(row, stage=2) is None
    assert any(
        message["role"] == "user"
        and "Now switch to the Self-Repair role." in message["content"]
        for message in row["messages"]
    )
    assert any(
        message["role"] == "user"
        and "Apply the repair instruction" in message["content"]
        for message in row["messages"]
    )
    assert sum(
        message["role"] == "user"
        and "Now switch to the Verifier role." in message["content"]
        for message in row["messages"]
    ) == 3


def test_sandbox_rewrites_quoted_image_path_argument():
    tree = ast.parse('from PIL import Image\nimg = Image.open("image_path")\n')
    transformer = ImagePathTransformer('/tmp/input.png')
    transformed = transformer.visit(tree)
    ast.fix_missing_locations(transformed)

    assert transformer.path_was_replaced is True
    assert 'Image.open(image_path)' in ast.unparse(transformed)
