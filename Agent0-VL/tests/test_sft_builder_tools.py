"""Exercise the actual builder sandbox without a live teacher or GPU."""
from PIL import Image

from tools.sft_builder import build
from tools.sft_builder.merge_sft import audit_record


def test_image_tool_round_trip_and_export(tmp_path):
    image = tmp_path / 'input.png'
    Image.new('RGB', (32, 24), (255, 0, 0)).save(image)

    class Teacher:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, images, system_prompt):
            self.calls += 1
            assert images == [str(image)]
            if self.calls == 1:
                return '```python\nfrom PIL import Image\nprint(Image.open(image_path).width)\n```'
            observations = [m['content'] for m in messages if '[Code Execution Result]' in m['content']]
            assert any('Output: 32' in text for text in observations)
            if self.calls in (2, 4):
                return '{"step_index": 1, "score": 1, "confidence": 0.99, "critique": "correct", "tool_check": true}'
            assert self.calls == 3
            return 'The image width is \\boxed{32}.'

    records, stats = build.build_records(
        [{'stage': 2, 'question': 'What is the image width?', 'images': [str(image)],
          'ground_truth': '32', 'ground_truth_aliases': []}],
        Teacher(), 'solver system prompt', max_tasks=1, quality_profile='stage2',
    )
    assert stats.exported == 1
    assert stats.successful_tool_calls == 1
    assert audit_record(records[0], stage=2, allow_stage2_images=True) is None


def test_tool_error_becomes_observation(tmp_path):
    image = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(image)
    results = build.execute_python(['print(1 / 0)'], sandbox_timeout=10, images=[str(image)])
    assert not results[0]['success']
    assert 'division by zero' in build.format_observation(results)


def test_text_tool_round_trip():
    results = build.execute_python(['print(40 + 2)'], sandbox_timeout=10)
    assert results[0]['success']
    assert 'Output: 42' in build.format_observation(results)


def test_optional_tools_and_final_with_code():
    from tests.test_sft_builder_repair import QueueTeacher
    verification = '{"step_index": 1, "score": 1, "confidence": 0.99, "critique": "correct", "tool_check": false}'
    sample = {'stage': 2, 'question': 'Compute 40 + 2.', 'images': [], 'ground_truth': '42'}
    records, stats = build.build_records([sample], QueueTeacher(['\\boxed{42}', verification]),
                                       'solver', max_tasks=1, quality_profile='stage2')
    assert stats.exported == 1 and stats.tool_calls == 0
    assert audit_record(records[0], stage=2) is None
    teacher = QueueTeacher(['```python\nprint(40+2)\n```\n\\boxed{42}', verification.replace('false', 'true')])
    records, stats = build.build_records([sample], teacher, 'solver', max_tasks=1, quality_profile='stage2')
    assert stats.exported == 1 and stats.successful_tool_calls == 1
    assert audit_record(records[0], stage=2) is None


def test_multiple_text_reasoning_steps_without_tools():
    from tests.test_sft_builder_repair import QueueTeacher
    verification = '{"step_index": 1, "score": 1, "confidence": 0.99, "critique": "sound", "tool_check": false}'
    teacher = QueueTeacher([
        'First identify the quantities.', verification,
        'Next compute the sum of 40 and 2.', verification,
        '\\boxed{42}', verification,
    ])
    sample = {'stage': 2, 'question': 'Compute 40 + 2.', 'images': [], 'ground_truth': '42'}
    records, stats = build.build_records([sample], teacher, 'solver', max_tasks=1,
                                       max_reasoning_steps=16, quality_profile='stage2')
    assert stats.exported == 1 and stats.tool_calls == 0
    assert audit_record(records[0], stage=2) is None
