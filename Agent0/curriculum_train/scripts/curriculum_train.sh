#!/bin/bash

project_name=agent0

executor_agent_path=$1
curriculum_agent_path=$2
save_path=$3
echo "save_path: $save_path"
RUN_ID=$(date +%s%N)
export RUN_ID

echo "RUN_ID=$RUN_ID"

# vLLM server
bash vllm_service_init/start.sh $executor_agent_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

echo "Start training curriculum: $curriculum_agent_path -> $save_path"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$curriculum_agent_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/curriculum_reward.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=128 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.max_steps=6 \
    trainer.save_freq=1

sleep 5

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10

pkill python

echo "curriculum agent training finished"
