# Rebuild the 500 + 500 SFT and 200 + 200 RL training data

Run from `Agent0-VL/`. The published paper lists sources but does not publish
per-source quotas or its original data-generation program. This local rebuild
allocates each stage's rows in proportion to **downloaded, deduplicated train
record counts before teacher filtering**. Integer quotas use largest remainders
and reserve one row per included source. Failed teacher generations are replaced
only by later candidates from the same source. The manifest records all counts.

The SFT sources are Geometry3K, GeoQA, Mulberry, LLaVA-OV-Image, MM-RLHF,
SMR, MM-Eureka, ReTool, and arXivQA. Existing local copies cover six of these;
the preflight reports missing paths for LLaVA-OV-Image, MM-RLHF, and SMR.
Official source locations are
[LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data),
[MM-RLHF](https://huggingface.co/datasets/yifanzhang114/MM-RLHF), and
[SMR](https://huggingface.co/datasets/yifanzhang114/SMR).
Download only their official training data to `data/raw/.staging/` or pass an
alternative `--source NAME=PATH` to `tools.sft_builder.rebuild_paper_1000`.
Keep raw records and images together; image paths must resolve locally or be
embedded in Parquet records. The script refuses to substitute test data.
For example, if the three extra downloads use different directories:

```bash
.venv/bin/python -m tools.sft_builder.rebuild_paper_1000 preflight \
  --source llava_ov_image=/path/to/llava/train \
  --source mm_rlhf=/path/to/mm_rlhf/train \
  --source smr=/path/to/smr/train
```

Pass the same `--source` values to `prepare`; `build` reads the saved source
manifest and needs no source overrides.

```bash
.venv/bin/python -m tools.clean_rebuild_artifacts            # list deletion targets
.venv/bin/python -m tools.clean_rebuild_artifacts --apply    # remove old generated data/logs
bash scripts/rebuild-paper-data.sh preflight                  # no training
bash scripts/rebuild-paper-data.sh prepare                    # index/count raw train rows
TEACHER_API_KEY_FILE=/path/to/key bash scripts/rebuild-paper-data.sh sft
bash scripts/rebuild-paper-data.sh rl
PREFLIGHT_ONLY=1 bash scripts/sft-agent0-2x4090-full.sh
bash scripts/sft-agent0-2x4090-full.sh                       # Stage 1 -> Stage 2
bash scripts/rl-agent0-2x4090-optimized.sh                   # 3 + 1 epochs
```

The SFT teacher must be the locally served Qwen3.8-27B model at
`http://127.0.0.1:8000/v1`; the rebuild script does not start or stop it.
`scripts/rebuild-paper-data.sh all` runs preparation, teacher generation, and
RL data export after preflight. Stage 2 generation starts only after Stage 1
reaches 500 rows and passes strict audit. Outputs are
`data/sft/large/stage1_500.jsonl`, `data/sft/large/stage2_500.jsonl`,
`data/rl/rl_warmup_200_multisource.parquet`, and
`data/rl/rl_200_multisource.parquet`. SFT rows retain the released `main`
`{messages, images}` format and `scripts/prompt.txt`; QLoRA runtime settings
remain local to the two-GPU launchers.

The RL subset uses ChartQA, arXivQA, and ThinkLite train records. MathVerse,
MathVista, and WeMath are listed by the paper, but the public artifacts checked
for this rebuild are evaluation splits; MathVista's dataset card explicitly
prohibits training on its test set. This difference is recorded in the RL
manifest. Warm-up and formal RL each contain 200 disjoint rows, allocated by
the downloaded train-record counts. With batch size 2, the launch targets 300
warm-up steps (3 epochs), then 100 additional formal steps (1 epoch).
For the current downloads, the raw training counts are ChartQA 28,299,
arXivQA 100,000, and ThinkLite 69,997; each 200-row RL subset receives
28, 101, and 71 rows respectively. The arXivQA RL partition has 39,807
eligible rows, enough to fill its quota without reusing SFT rows.

The cleanup keeps `data/raw/`, `data/processed/` provenance indexes,
`data/sft/validation/`, `data/rl/validation_*`, and all checkpoints. It will
refuse to remove files while a training or builder process is active.
