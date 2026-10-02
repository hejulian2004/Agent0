# 单共享 LoRA 实施记录

目标仓库：`/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL`，分支 `update`。
本次仅修改源代码、配置、相关测试和说明；未启动 GPU、教师服务、数据生成、SFT 或 RL，未停止或调整现有任务。

## Adapter 与 Mode

```python
MODES = ("solve", "repair", "verify")
ADAPTERS = ("shared",)
ADAPTER_FOR_MODE = {"solve": "shared", "repair": "shared", "verify": "shared"}
```

冻结 Base 和视觉编码器；Actor 只有一套共享 LoRA 参数、一个 optimizer、一个 scheduler 和 `versions={"shared": n}`。Reference 独立保留初始 SFT 的冻结副本，不维护训练 optimizer。没有第三套 Repair LoRA，也不相加或平均旧的三套权重。

## SFT 路径

保留三类正样本，增加 `roles/shared.jsonl` 训练视图及来源/内容哈希、逐 mode 数量的 manifest。该视图是三类既有正样本的联合集合，不改变教师问题配额、不重采样样本、不拼接不同角色会话。

一次共享 SFT 更新同一份参数。逐行校验 Solve/Repair/Verify 提示词、模式、长度和 loss boundary。Swift 编码器再次检查逐 mode 实际 token 限制；Repair prefix/Checkpoint 不进入监督 target，丢弃的旧 suffix 不进入正样本 target。

入口：`sft-local --profile local_4090_checkpointed --adapter shared`；旧的 `--mode` 分别训练/export 调用显式拒绝。HF export 保持 `--merge_lora false`。

## RL 路径与权重

Solve、Repair、Verify GRPO 分组和 advantage 分别计算。全局有效 group 数在 padding 和 rank 切分前统计：完整 group、reward 可用、每条 completion 有可训练 action；不计入 infrastructure failure、reward=None、不完整 group、全零 action mask、dummy padding、关闭的 Verify batch。

有效零方差 group 保留现有零 advantage 行为，仍可计算现有 entropy/KL；不会仅因零方差被标记为失败。

配置：`checkpointed.joint_training.rl_weighting.method=prior_sqrt_groups`，先验 `solve=0.5, repair=0.3, verify=0.2`。

```text
lambda_i = prior_i * sqrt(global_valid_group_count_i)
           / sum_active(prior_j * sqrt(global_valid_group_count_j))
```

各 mode 按有效 group 平均，组内按 completion 平均，completion 内保留配置中的 token loss 聚合。独立 DataProto 通过同一个联合 RPC 分发；全部 rollout 和 Reference logprob 完成后，逐 mode/逐 completion 累积梯度。全部消费完才裁剪和执行一次 optimizer/scheduler step，成功后 shared version 加一。保持 `ppo_epochs=1`，其他取值在联合更新时显式拒绝。

无有效加权目标或非有限梯度不推进 scheduler/version。旧版本或已消费的 phase 拒绝再次使用；消费记录随 checkpoint 保存恢复。

保留原 action-conditioned reward、tool bonus 和 `repair_credit_mode=main|mean`。源码 AST 对照确认 reward、提示词、Verification Checkpoint、repair point、suffix regeneration 和 Episode 分支逻辑未变。

日志保持 `solve/*`, `repair/*`, `verify/*`，增加 `valid_groups`, `loss_weight`, `adapter/shared_version`, `adapter/update_applied`。轨迹记录共享 Adapter、native LoRA ID 和 policy version。

## vLLM 与消融

每个 rollout phase 只注册一份 shared LoRARequest；Solve→Verify→Repair→Verify 使用同一 Adapter ID。模式切换不 remove/add 或重载文件；在下一 rollout phase 同步新版本。

保留 context_isolation 和 suffix_repair。`train_verifier_rl=false` 跳过 Verify RL loss 和 Reference 计算，Verifier 仍推理；其共享参数仍随 Solve/Repair 更新，此开关不表示冻结独立 Verifier 参数。

## Checkpoint 与 Bundle

bundle 注册：`python -m tools.checkpointed_bundle --base-model BASE --shared SHARED --output BUNDLE`。只记录 shared 权重、映射、内容哈希和训练布局 `agent0.checkpointed.shared_lora.v1`。

PEFT 和 FSDP checkpoint 只存一份 optimizer/scheduler/version；旧三 Adapter bundle/checkpoint、缺少新布局字段的文件、错误映射和不兼容 fingerprint 明确拒绝。FSDP adapter-only exporter 识别 shared 参数名；generic default-LoRA 导出保持原有行为。

数据协议仍是 `agent0.checkpointed.v1`，Verification Checkpoint schema 不变。没有重写现有数据、旧 checkpoint 或模型 artifacts；已有合规 checkpointed 数据可用于新共享训练。

## 实际验证

以下命令从目标组件目录执行，全部禁用 GPU：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_checkpointed_*.py tests/test_local_4090.py \
  tests/test_agent0_protocol.py tests/test_sft_migration.py \
  tests/test_build_sft_dataset.py tests/test_local_rollout.py \
  tests/test_image_tool_context.py tests/test_teacher_backends.py \
  tests/test_agent0_schema.py --disable-warnings --maxfail=2
```

结果：136 passed，14 项已有依赖警告，86.04 秒。包含真实两 rank CPU/Gloo/FSDP 联合更新与全局参考参数对照、共享 optimizer checkpoint 往返，以及真实 worker RPC 的 CPU 替身测试、KL/entropy 梯度对照。

最后增加 Adapter 审计字段后复核：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_checkpointed_runtime.py tests/test_checkpointed_trainer.py \
  tests/test_checkpointed_gloo.py --disable-warnings --maxfail=1
```

结果：14 passed，59.78 秒。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.launch \
  sft-local --profile local_4090_checkpointed --adapter shared --dry-run

CUDA_VISIBLE_DEVICES='' .venv/bin/python -m tools.checkpointed_rl \
  --base-model /tmp/base-placeholder \
  --adapter-bundle /tmp/shared-bundle-placeholder --dry-run

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m tools.data_builder.local_data \
  preflight --profile local_4090_checkpointed
```

以上均退出 0；preflight 的 SFT 环境检查 ready=true。另执行 load_config/validate_local、修改文件语法检查和 git diff --check，均通过。占位模型路径的 dry-run 仅检查命令/配置解析；bundle 哈希与旧格式拒绝使用 CPU 小模型文件在测试中验证。

尚未验证：实际 GPU Swift/Megatron 联合 SFT、Ray/FSDP GPU 联合更新、vLLM GPU LoRA 加载/同步/推理、GPU 显存与吞吐。不将 CPU 测试或 dry-run 描述为 GPU 训练成功。

## 本次修改文件

- [agent0_protocol/checkpointed.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/agent0_protocol/checkpointed.py)
- [config.yaml](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/config.yaml)
- [docs/memory/checkpointed_protocol.md](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/docs/memory/checkpointed_protocol.md)
- [docs/memory/shared_lora_implementation.md](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/docs/memory/shared_lora_implementation.md)
- [tests/test_checkpointed_gloo.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tests/test_checkpointed_gloo.py)
- [tests/test_checkpointed_protocol.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tests/test_checkpointed_protocol.py)
- [tests/test_checkpointed_runtime.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tests/test_checkpointed_runtime.py)
- [tests/test_checkpointed_shared.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tests/test_checkpointed_shared.py)
- [tests/test_checkpointed_trainer.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tests/test_checkpointed_trainer.py)
- [tools/checkpointed_bundle.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/checkpointed_bundle.py)
- [tools/checkpointed_rl.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/checkpointed_rl.py)
- [tools/checkpointed_sft.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/checkpointed_sft.py)
- [tools/export_fsdp_lora_hf.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/export_fsdp_lora_hf.py)
- [tools/local_sft.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/local_sft.py)
- [tools/local_workflows.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/local_workflows.py)
- [tools/swift_canonical_plugin.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/swift_canonical_plugin.py)
- [tools/training/checkpointed_batch.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/checkpointed_batch.py)
- [tools/training/checkpointed_records.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/checkpointed_records.py)
- [tools/training/checkpointed_rollout.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/checkpointed_rollout.py)
- [tools/training/checkpointed_trainer.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/checkpointed_trainer.py)
- [tools/training/checkpointed_update.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/checkpointed_update.py)
- [tools/training/native_role_lora.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/native_role_lora.py)
- [tools/training/role_adapters.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/role_adapters.py)
- [tools/training/role_checkpoint.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/role_checkpoint.py)
- [tools/training/role_fsdp_checkpoint.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/role_fsdp_checkpoint.py)
- [tools/training/role_weight_sync.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/tools/training/role_weight_sync.py)
- [verl/workers/fsdp_workers.py](/mnt/d/Agent0-dev-hjl/Agent0-VL-HJL/verl/workers/fsdp_workers.py)
