import ast

from PIL import Image

from tools.sft_builder import build
from tools.sft_builder.merge_sft import _audit_repair_flow, audit_record
from verl.utils.sandbox.executor import ImagePathTransformer


class QueueTeacher:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, messages, images, system_prompt):
        del messages, images, system_prompt
        if not self.responses:
            raise AssertionError("fake teacher response queue was exhausted")
        return self.responses.pop(0)


def _valid_stage2_row():
    return {
        "messages": [
            {"role": "user", "content": "Compute 1 + 1."},
            {"role": "assistant", "content": "```python\nprint(1 + 1)\n```"},
            {"role": "user", "content": "[Code Execution Result]\nOutput: 2"},
            {"role": "assistant", "content": "<answer>\\boxed{2}</answer>"},
            {"role": "user", "content": "Now switch to the Verifier role. Verify the answer."},
            {
                "role": "assistant",
                "content": (
                    '{"step_index": 1, "score": 1, "confidence": 0.95, '
                    '"critique": "correct", "tool_check": true}'
                ),
            },
        ],
        "images": [],
    }


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
    assert any('"confidence": 0.4' in message["content"] for message in row["messages"])
    assert any('"confidence": 0.95' in message["content"] for message in row["messages"])
    assert audit_record(row, stage=2) is None
    assert any(
        message["role"] == "user"
        and "Now switch to the Self-Repair role." in message["content"]
        for message in row["messages"]
    )
    assert any(
        message["role"] == "user"
        and "A repair instruction has been issued for the previous step:" in message["content"]
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


def test_audit_rejects_non_object_record():
    assert audit_record(1, stage=2) == "top_level_schema"


def test_audit_rejects_invalid_role_order():
    row = _valid_stage2_row()
    row["messages"].insert(1, {"role": "user", "content": "unexpected second user"})
    assert audit_record(row, stage=2) == "message_role_order"


def test_audit_rejects_stage2_images():
    row = _valid_stage2_row()
    row["messages"][0]["content"] = "<image>" + row["messages"][0]["content"]
    row["images"] = ["/definitely/missing.png"]
    assert audit_record(row, stage=2) == "stage2_unexpected_image"


def test_audit_allows_valid_stage2_images_when_requested(tmp_path):
    image_path = tmp_path / "stage2.png"
    Image.new("RGB", (1, 1), color="white").save(image_path)
    row = _valid_stage2_row()
    row["messages"][0]["content"] = "<image>" + row["messages"][0]["content"]
    row["images"] = [str(image_path)]
    assert audit_record(row, stage=2, allow_stage2_images=True) is None


def test_repair_audit_requires_post_repair_verifier_response():
    messages = [
        {"role": "user", "content": "Now switch to the Self-Repair role."},
        {
            "role": "assistant",
            "content": (
                '{"action": "PATCH", "target_step": 1, "patch_type": "code", '
                '"new_content": "print(2)", "justification": "fix"}'
            ),
        },
        {"role": "user", "content": "A repair instruction has been issued for the previous step:"},
        {"role": "assistant", "content": "```python\nprint(2)\n```"},
        {"role": "user", "content": "Now switch to the Verifier role."},
    ]
    assert _audit_repair_flow(messages) == "missing_post_repair_verifier_response"
