from tools.sft_builder.sources import _extract_question, _raw_ground_truth


def test_multi_turn_reference_matches_first_user():
    row = {'conversations': [
        {'from':'human','value':"What are the five R's?"},
        {'from':'gpt','value':'Refuse, reduce, reuse, recycle, rot.'},
        {'from':'human','value':'How can I apply them at home?'},
        {'from':'gpt','value':'Begin by assessing your household waste.'},
    ]}
    assert _extract_question(row) == "What are the five R's?"
    assert _raw_ground_truth(row, 'llava_ov_image') == 'Refuse, reduce, reuse, recycle, rot.'


def test_explicit_reference_and_single_turn_preserved():
    row = {'answer':'gold','messages':[{'role':'user','content':'question'},{'role':'assistant','content':'other'}]}
    assert _raw_ground_truth(row,'llava_ov_image') == 'gold'
    del row['answer']
    assert _raw_ground_truth(row,'llava_ov_image') == 'other'


def test_same_question_tool_turn_uses_final_answer():
    row = {'messages':[
        {'role':'user','content':'How many palindromes?'},
        {'role':'assistant','content':'Let me count using Python.'},
        {'role':'tool','content':'30'},
        {'role':'assistant','content':'\\boxed{30}'},
        {'role':'user','content':'Now solve another question.'},
        {'role':'assistant','content':'\\boxed{999}'},
    ]}
    assert _raw_ground_truth(row,'retool') == '\\boxed{30}'


def test_user_tool_observation_is_not_a_new_question():
    row = {'conversations':[
        {'from':'human','value':'40+2?'},
        {'from':'gpt','value':'Computing.'},
        {'from':'human','value':'[Code Execution Result]\nOutput: 42'},
        {'from':'gpt','value':'\\boxed{42}'},
    ]}
    assert _raw_ground_truth(row,'smr') == '\\boxed{42}'
