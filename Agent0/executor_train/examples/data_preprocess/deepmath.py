# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
import fire
from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.reward_score import prime_math

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


simple_rl_system_prompt = '''Please reason step by step, and put your final answer within \\boxed{}.'''

# torl_system_prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.
# '''
torl_system_prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. If you want to run any python code, write code in the python markdown code block and the execution will be appended in an output code block like "```python\nyou code here\n```\n```output\nresult here\n```". Please put your final answer within \\boxed{}.'''


def apply_system_prompt(sys_prompt_style:str, question:str):
    """
    Apply the system prompt style to the question.
    Args:
        sys_prompt_style (str): The system prompt style to apply. Can be 'simple_rl' or 'torl'.
        question (str): The question to apply the system prompt to.
    Returns:
        list: A list of dictionaries representing the conversation with the system prompt applied.
    """
    if sys_prompt_style == 'simple_rl':
        return [{'role': 'user', 'content': question + '\n' + simple_rl_system_prompt}]
    elif sys_prompt_style == 'torl':
        return [{'role': 'system', 'content': torl_system_prompt}, {'role': 'user', 'content': question}]
    else:
        raise ValueError(f"Unknown system prompt style: {sys_prompt_style}")

def main(
    data_source='zwhe99/DeepMath-103K',
    local_dir='~/data/deepmath_torl',
    hdfs_dir=None,
    sys_prompt_style= 'torl',
):
    
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # dataset = datasets.load_dataset(data_source, trust_remote_code=True, split='train')

    # dataset = datasets.load_dataset('/data_r1v4/data_r1v4/peng.xia/verl-tool/data/DeepMath-103K', split='train')
    # dataset = dataset.train_test_split(test_size=500, seed=42)
    # train_dataset = dataset['train']
    # test_dataset = dataset['test']
    
    math500_test_dataset = datasets.load_dataset('/data_r1v4/data_r1v4/peng.xia/verl-tool/data/MATH-500', split='test')
    
    # add a row to each data item that represents a unique id
    def make_train_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('question')
            answer = example.pop('final_answer')
            solution = answer
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
	
    def make_map_fn_math500(split, data_source):

        def process_fn(example, idx):
            question = example.pop('problem')
            answer = example.pop('solution')
            solution = extract_solution(answer)
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    # train_dataset = train_dataset.map(function=make_train_map_fn('train', data_source), with_indices=True, remove_columns=train_dataset.column_names)
    # test_dataset = test_dataset.map(function=make_train_map_fn('test', data_source), with_indices=True, remove_columns=test_dataset.column_names)
    math500_test_dataset = math500_test_dataset.map(function=make_map_fn_math500('test', 'MATH-500'), with_indices=True, remove_columns=math500_test_dataset.column_names)
    
    # print(train_dataset)
    # print(train_dataset[0])
    
    # train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    # test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    math500_test_dataset.to_parquet(os.path.join(local_dir, 'math500_test.parquet'))
    
    # aime24
    aime24_dataset = datasets.load_dataset('/data_r1v4/data_r1v4/peng.xia/verl-tool/data/AIME_2024', split='train') # actually test set
    def make_map_fn_aime24(split, data_source):

        def process_fn(example, idx):
            question = example.pop('Problem')
            answer = str(example.pop('Answer'))
            solution = example.pop('Solution')
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }

            return data

        return process_fn
    
    aime24_dataset = aime24_dataset.map(function=make_map_fn_aime24('test', 'aime24'), with_indices=True, remove_columns=aime24_dataset.column_names)
    df_aime24 = aime24_dataset.to_pandas()
    df_aime24_duplicated = df_aime24.loc[df_aime24.index.repeat(32)].reset_index(drop=True)
    aime24_dataset = datasets.Dataset.from_pandas(df_aime24_duplicated)
    aime24_dataset.to_parquet(os.path.join(local_dir, 'aime24_32_test.parquet'))
    print("*"*50)
    print("aime24:\n")
    print(aime24_dataset)
    print(aime24_dataset[0])
    
    # aime25
    aime25_dataset = datasets.load_dataset('/data_r1v4/data_r1v4/peng.xia/verl-tool/data/AIME2025', 'AIME2025-I', split='test') # actually test set
    aime25_dataset2 = datasets.load_dataset('/data_r1v4/data_r1v4/peng.xia/verl-tool/data/AIME2025', 'AIME2025-II', split='test') # actually test set
    # concatenate the two datasets
    aime25_dataset = datasets.concatenate_datasets([aime25_dataset, aime25_dataset2])
    
    def make_map_fn_aime25(split, data_source):

        def process_fn(example, idx):
            question = example.pop('question')
            answer = str(example.pop('answer'))
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn


    aime25_dataset = aime25_dataset.map(function=make_map_fn_aime25('test', 'aime25'), with_indices=True, remove_columns=aime25_dataset.column_names)
    df_aime25 = aime25_dataset.to_pandas()
    df_aime25_duplicated = df_aime25.loc[df_aime25.index.repeat(32)].reset_index(drop=True)
    aime25_dataset = datasets.Dataset.from_pandas(df_aime25_duplicated)
    aime25_dataset.to_parquet(os.path.join(local_dir, 'aime25_32_test.parquet'))
    print("*"*50)
    print("aime25:\n")
    print(aime25_dataset)
    print(aime25_dataset[0])

    # amc
    amc_dataset = datasets.load_dataset("/data_r1v4/data_r1v4/peng.xia/repo/R-Zero/data/amc23", split='test')
    amc_dataset = amc_dataset.map(function=make_map_fn_aime25('test', 'amc'), with_indices=True, remove_columns=amc_dataset.column_names)
    df_amc = amc_dataset.to_pandas()
    df_amc_duplicated = df_amc.loc[df_amc.index.repeat(32)].reset_index(drop=True)
    amc_dataset = datasets.Dataset.from_pandas(df_amc_duplicated)
    amc_dataset.to_parquet(os.path.join(local_dir, 'amc_32_test.parquet'))
    # amc_dataset.to_parquet(os.path.join(local_dir, 'amc_test.parquet'))
    print("*"*50)
    print("amc:\n")
    print(amc_dataset)
    print(amc_dataset[0])

    # gsm8k
    gsm8k_dataset = datasets.load_dataset("/data_r1v4/data_r1v4/peng.xia/repo/R-Zero/data/gsm8k", 'main', split='test')

    def make_map_fn_gsm8k(split, data_source):

        def process_fn(example, idx):
            question = example.pop('question')
            answer = str(example.pop('answer').split('#### ')[-1])
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn

    gsm8k_dataset = gsm8k_dataset.map(function=make_map_fn_gsm8k('test', 'gsm8k'), with_indices=True, remove_columns=gsm8k_dataset.column_names)
    gsm8k_dataset.to_parquet(os.path.join(local_dir, 'gsm8k_test.parquet'))
    print("*"*50)
    print("gsm8k:\n")
    print(gsm8k_dataset)
    print(gsm8k_dataset[0])

    # minerva
    minerva_dataset = datasets.load_dataset("/data_r1v4/data_r1v4/peng.xia/repo/R-Zero/data/simplerl-minerva-math", split='test')

    def make_map_fn_minerva(split, data_source):

        def process_fn(example, idx):
            question = example.pop('problem')
            answer = str(example.pop('answer'))
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn

    minerva_dataset = minerva_dataset.map(function=make_map_fn_minerva('test', 'minerva'), with_indices=True, remove_columns=minerva_dataset.column_names)
    minerva_dataset.to_parquet(os.path.join(local_dir, 'minerva_test.parquet'))
    print("*"*50)
    print("minerva:\n")
    print(minerva_dataset)
    print(minerva_dataset[0])

    # olympiad
    olympiad_dataset = datasets.load_dataset("/data_r1v4/data_r1v4/peng.xia/repo/R-Zero/data/simplerl-OlympiadBench", split='test')
    def make_map_fn_olympiad(split, data_source):

        def process_fn(example, idx):
            question = example.pop('question')
            answer = str(example.pop('final_answer'))
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    olympiad_dataset = olympiad_dataset.map(function=make_map_fn_olympiad('test', 'olympiad'), with_indices=True, remove_columns=olympiad_dataset.column_names)
    olympiad_dataset.to_parquet(os.path.join(local_dir, 'olympiad_test.parquet'))
    print("*"*50)
    print("olympiad:\n")
    print(olympiad_dataset)
    print(olympiad_dataset[0])

    # mmlu_pro
    mmlupro_dataset = datasets.load_dataset('/data_r1v4/data_r1v4/peng.xia/repo/R-Zero/data/MMLU-Pro', split='test')
    def make_map_fn_mmlupro(split, data_source):
        def process_fn(example, idx):
            original_question = example['question']
            options = example['options']
            
            formatted_question = original_question + "\n\nOptions:\n"
            for i, opt in enumerate(options):
                formatted_question += f"{chr(65+i)}. {opt}\n"

            answer = str(example['answer'])
            category = example['category']

            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, formatted_question),
                "ability": category,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': original_question
                }
            }
            return data
        return process_fn

    mmlupro_dataset = mmlupro_dataset.map(function=make_map_fn_mmlupro('test', 'mmlupro'), with_indices=True, remove_columns=mmlupro_dataset.column_names)
    mmlupro_dataset.to_parquet(os.path.join(local_dir, 'mmlupro_test.parquet'))
    print("*"*50)
    print("mmlupro:\n")
    print(mmlupro_dataset)
    print(mmlupro_dataset[0])

    # bbeh
    bbeh_dataset = datasets.load_dataset("/data_r1v4/data_r1v4/peng.xia/repo/R-Zero/data/bbeh-eval", split='train')
    bbeh_dataset = bbeh_dataset.map(function=make_map_fn_aime25('test', 'bbeh'), with_indices=True, remove_columns=bbeh_dataset.column_names).select(range(1000))
    bbeh_dataset.to_parquet(os.path.join(local_dir, 'bbeh_test.parquet'))
    print("*"*50)
    print("bbeh:\n")
    print(bbeh_dataset)
    print(bbeh_dataset[0])

    # supergpqa
    supergpqa_dataset = datasets.load_dataset('/data_r1v4/data_r1v4/peng.xia/repo/R-Zero/data/SuperGPQA', split="train")
    def make_map_fn_supergpqa(split, data_source):
        def process_fn(example, idx):
            original_question = example['question']
            options = example['options']
            
            formatted_question = original_question + "\n\nOptions:\n"
            for i, opt in enumerate(options):
                formatted_question += f"{chr(65+i)}. {opt}\n"

            answer = str(example['answer_letter'])

            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, formatted_question),
                "ability": "supergpqa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': original_question
                }
            }
            return data
        return process_fn
    supergpqa_dataset = supergpqa_dataset.map(function=make_map_fn_supergpqa('test', 'supergpqa'), with_indices=True, remove_columns=supergpqa_dataset.column_names)
    supergpqa_dataset.to_parquet(os.path.join(local_dir, 'supergpqa_test.parquet'))
    print("*"*50)
    print("supergpqa:\n")
    print(supergpqa_dataset)
    print(supergpqa_dataset[0])


    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
    

if __name__ == '__main__':
    fire.Fire(main)
    
"""
# simple rl system prompt (no tool)
python examples/data_preprocess/deepmath.py --data_source zwhe99/DeepMath-103K --local_dir data/deepmath_simple_rl --sys_prompt_style simple_rl
# torl system prompt (with code interpreter tool)
python examples/data_preprocess/deepmath.py --data_source zwhe99/DeepMath-103K --local_dir data/deepmath_torl --sys_prompt_style torl
"""