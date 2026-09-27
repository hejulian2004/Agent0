import io
import json
import urllib.error

from tools.sft_builder import build
from tools.sft_builder.merge_sft import audit_record
from tools.sft_builder.teacher import OpenAICompatibleTeacher, TeacherError


def test_repair_code_braces_and_verifier_latex():
    code = 'print(f"value={40 + 2}")'
    patch = {'action': 'PATCH', 'target_step': 1, 'patch_type': 'code',
             'new_content': code, 'justification': 'recompute'}
    assert build.parse_repair_output(json.dumps(patch))['new_content'] == code
    text = '{"step_index":1,"score":1,"confidence":0.95,"critique":"angle is 70^\\circ","tool_check":false}'
    assert build.parse_verification_output(text)['critique'] == 'angle is 70^\\circ'
    raw_newline = '{"action":"PATCH","new_content":"```python\nprint(42)\n```"}'
    assert build.parse_repair_output(raw_newline)['new_content'] == '```python\nprint(42)\n```'
    assert build.parse_verification_output('```python\nprint(42)\n```') is None
    assert build.parse_repair_output('{"action":"PATCH",') is None


def test_no_tool_repair_can_export_with_code_quoted_in_patch():
    from tests.test_sft_builder_repair import QueueTeacher
    low = '{"step_index":1,"score":0,"confidence":0.4,"critique":"recompute","tool_check":false}'
    high = '{"step_index":1,"score":1,"confidence":0.99,"critique":"correct","tool_check":false}'
    patch = json.dumps({'action':'PATCH','target_step':1,'patch_type':'text',
                        'new_content':'For example ```python\nprint(42)\n```; calculate mentally instead.',
                        'justification':'correct arithmetic'})
    teacher = QueueTeacher(['A tentative result.', low, patch, '\\boxed{42}', high])
    sample = {'stage':2,'question':'40 + 2?','images':[],'ground_truth':'42'}
    records, stats = build.build_records([sample], teacher, 'solver', max_tasks=1, quality_profile='stage2')
    assert stats.exported == 1
    assert audit_record(records[0], stage=2) is None


def test_context_400_reduces_output_budget(monkeypatch):
    budgets = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'\\boxed{42}'}}]}).encode()
    class Opener:
        def open(self, request, timeout):
            budgets.append(json.loads(request.data)['max_tokens'])
            if len(budgets) == 1:
                detail = {"error":{"message":"This model's maximum context length is 24576 tokens. However, your prompt contains at least 16385 input tokens."}}
                raise urllib.error.HTTPError(request.full_url, 400, 'bad', {}, io.BytesIO(json.dumps(detail).encode()))
            return Response()
    monkeypatch.setattr('urllib.request.build_opener', lambda *args: Opener())
    teacher = OpenAICompatibleTeacher('http://127.0.0.1:8000/v1','qwen3.8-27b')
    assert teacher.generate([{'role':'user','content':'40+2?'}], [], 'solver') == '\\boxed{42}'
    assert budgets == [8192, 4096]


def test_final_no_tool_flag_does_not_override_verified_earlier_tool():
    from tests.test_sft_builder_repair import QueueTeacher
    verified = '{"step_index":1,"score":1,"confidence":0.99,"critique":"correct","tool_check":true}'
    final = '{"step_index":2,"score":1,"confidence":0.99,"critique":"correct","tool_check":false}'
    sample = {'stage':2,'question':'40 + 2?','images':[],'ground_truth':'42'}
    teacher = QueueTeacher(['```python\nprint(42)\n```', verified, '\\boxed{42}', final])
    records, stats = build.build_records([sample],teacher,'solver',max_tasks=1,quality_profile='stage2')
    assert stats.exported == 1
    assert audit_record(records[0],stage=2) is None
    teacher = QueueTeacher(['```python\nprint(42)\n```', verified.replace('true','false'), '\\boxed{42}', final])
    records, stats = build.build_records([sample],teacher,'solver',max_tasks=1,quality_profile='stage2')
    assert not records
    assert stats.failure_reasons['quality_verifier_tool_check_false'] == 1
