import json

from tools.sft_builder import build
from tools.sft_builder.merge_sft import audit_record

VERIFIER = '{"step_index":1,"score":1,"confidence":0.99,"critique":"correct","tool_check":false}'


class Teacher:
    def __init__(self, solver, verdict=None):
        self.responses = [solver, VERIFIER]
        self.verdict = verdict
        self.judge_calls = []

    def generate(self, messages, images, system_prompt):
        if system_prompt == build.ANSWER_JUDGE_PROMPT:
            assert len(messages) == 1 and images == []
            payload = json.loads(messages[0]['content'])
            assert set(payload) == {'reference_answers', 'candidate_answer'}
            assert 'SECRET QUESTION' not in messages[0]['content']
            self.judge_calls.append(payload)
            return json.dumps(self.verdict)
        return self.responses.pop(0)


def run(teacher, reference):
    sample = {'stage':2,'question':'SECRET QUESTION','images':[], 'ground_truth':reference}
    return build.build_records([sample],teacher,'solver',max_tasks=1,max_reasoning_steps=1,quality_profile='stage2')


def test_exact_answer_bypasses_judge():
    teacher = Teacher('\\boxed{42}')
    records, stats = run(teacher, '42')
    assert stats.exported == 1 and not teacher.judge_calls


def test_semantic_answer_uses_isolated_judge():
    teacher = Teacher('\\boxed{Michigan Avenue}', {'equivalent':True,'has_final_answer':True,'candidate_answer':'Michigan Avenue'})
    records, stats = run(teacher, 'Michigan Ave.')
    assert stats.exported == 1 and len(teacher.judge_calls) == 1
    assert stats.answer_judge_accepted == 1
    assert audit_record(records[0],stage=2) is None
    assert all('Michigan Ave.' not in m['content'] for m in records[0]['messages'])


def test_unparsed_final_answer_recovered_without_reference_leak():
    teacher = Teacher('The final answer is Michigan Avenue.', {'equivalent':True,'has_final_answer':True,'candidate_answer':'Michigan Avenue'})
    records, stats = run(teacher, 'Michigan Ave.')
    assert stats.exported == 1
    assert audit_record(records[0],stage=2) is None
    assert any('FINAL_ANSWER: Michigan Avenue' in m['content'] for m in records[0]['messages'])


def test_judge_cannot_replace_answer_or_accept_uncertainty():
    for verdict in [
        {'equivalent':False,'has_final_answer':True,'candidate_answer':'wrong'},
        {'equivalent':True,'has_final_answer':False,'candidate_answer':'wrong'},
        {'equivalent':True,'has_final_answer':True,'candidate_answer':'42'},
        {'equivalent':'true','has_final_answer':True,'candidate_answer':'wrong'},
    ]:
        teacher = Teacher('\\boxed{wrong}',verdict)
        records, stats = run(teacher, '42')
        assert not records
        assert stats.answer_judge_calls == 1
        assert stats.answer_judge_accepted == 0
