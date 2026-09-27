# Single-stage light reproduction

This run uses nine locally downloaded SFT sources: Geometry3K (112 final rows),
GeoQA, Mulberry, LLaVA-OV-Image, MM-RLHF, SMR, MM-Eureka, ReTool and arXivQA
(111 final rows each). Total: 1,000. Source selection is balanced, not the
unpublished full-paper distribution. The requested single-stage run replaces
the paper's Stage 1 / Stage 2 curriculum. Teacher: local Qwen3.8-27B, replacing
the teachers named in the paper.

Appendix D.1 also mentions MathVerse, MathVista, WeMath and ChartQA as additional
math/chart sources. This subset uses the nine principal SFT sources with usable
local training candidates; it does not use the downloaded evaluation sets.
Appendix D.3 lists six RL sources. The current usable local training pool uses
ChartQA, arXivQA and ThinkLite; MathVerse, MathVista and WeMath local copies are
evaluation artifacts. This is a source-coverage limitation of this reproduction.
SFT and RL may share a source (arXivQA), but their selected questions must be disjoint.

## Data construction

From Agent0-VL:

```bash
bash scripts/rebuild-balanced-data.sh prepare
bash scripts/build-balanced-sft-with-teacher.sh
```

The second command starts the four-GPU teacher profile on GPUs 0,1,2,3 with TP=4 and 64 concurrent requests when the
service is stopped, prints loading and generation progress, resumes source
outputs, audits the complete 1,000-row SFT output, generates the two disjoint
200-row RL subsets, and stops only the teacher it started. The four-GPU generation profile has not been GPU smoke-tested in this change; the user runs generation. Logs and status are
registered in logs/balanced_sft_build_<timestamp>/README.md.

SFT final output: data/sft/large/mixed_balanced_1000.jsonl. Every row contains
only `messages` and `images`; images remain local paths; image markers match
image counts. There are no stored system messages. Code observations are user
messages containing `[Code Execution Result]`. Generation uses the unmodified
main scripts/prompt.txt and main Verifier/Repair injections. Teacher samples
must execute tools successfully and pass verification; missing source references are rejected by the current strict build. Existing
source references are checked. Source answers are never included in teacher requests.

The paper's prose describes think tags and JSON tool calls. Released main uses
fenced Python in code tags through scripts/prompt.txt. This executable pipeline
follows main. Final answer extraction accepts main's boxed / FINAL_ANSWER forms;
there is no locally added answer-tag-only or plain-answer fallback. Both RL
reward managers use main's math_verify correctness scorer, including its
300-character answer window. Local benchmark-specific scoring is not used for
training. This preserves main behavior, including its limits on free-text answers.

RL outputs: data/rl/rl_warmup_200_multisource.parquet and
 data/rl/rl_200_multisource.parquet. Fields: prompt, images, reward_model,
data_source, extra_info. Prompts contain a user question and image marker;
images contain embedded bytes; reward_model contains ground_truth. Both sets
exclude validation and final SFT questions and are mutually disjoint. Each uses
28 ChartQA, 101 arXivQA and 71 ThinkLite rows, proportional to downloaded train
counts. SFT balancing does not change RL quotas.

## Train after data generation succeeds

```bash
PREFLIGHT_ONLY=1 bash scripts/sft-agent0-2x4090-mixed.sh
bash scripts/sft-agent0-2x4090-mixed.sh
# Or use all four idle GPUs:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft-agent0-2x4090-mixed.sh

bash scripts/rl-agent0-2x4090-optimized.sh
```

SFT: 3 epochs, lr 1e-5, warmup 0.05, global batch 128 by default,
main max_length 10240, per-device batch 1, QLoRA NF4/BF16 storage and compute,
rank 16/alpha 64, frozen visual modules, gradient checkpointing, ZeRO-3.
SFT progress prints every training step. A new run directory stores logs,
adapters, merged weights and its README inventory. The launcher updates
checkpoints/sft_2x4090/latest_mixed_merged after success. MAX_LENGTH can be
lowered explicitly for a local memory limit; doing so changes truncation behavior.

RL uses that mixed model by default on GPUs 0,1,2,3 with TP=4 and 64 concurrent requests: correctness warm-up for
3 epochs (300 steps with 200 rows and global batch 2), then SERC for 1 epoch
(100 additional steps). The launcher resumes the warm-up checkpoint for SERC.
QLoRA rank 8/alpha 32, BF16/NF4, FSDP original parameters, CPU offload and
BNB dequantization + LoRA merge for vLLM are retained. load_format=dummy_hf
and vLLM sleep level 2 remain compatibility/memory adaptations. Rollout has
n=8, TP=2 and max_num_seqs=16. A heartbeat prints GPU/RAM/stage status every
30 seconds, and rollout samples show generated answers. Training remains on
the two-GPU verified path; only SFT supports the four-GPU alternative here.

## Configurable launch arguments

Both launchers accept command-line options, which override environment variables.
Use `--help` for the complete option list and `--dry-run` to print settings
without model loading, training or a new run directory.

```bash
bash scripts/sft-agent0-2x4090-mixed.sh --gpus 0,1,2,3 --batch-size 1 --grad-accum 32 --epochs 3 --workers 4 --dry-run
bash scripts/rl-agent0-2x4090-optimized.sh --gpus 0,1,2,3 --tp 2 --batch-size 4 --rollout-n 8 --concurrency 32 --mini-batch-size 1 --micro-batch-size 1 --dry-run
```

SFT batch size is per GPU; global batch is GPU count * batch * accumulation.
SFT `--concurrency` is an alias for CPU dataloader `--workers`. RL batch size
is the global prompt batch; rollout-n multiplies it into trajectories.
RL concurrency sets vLLM's sequence cap. TP must divide GPU count; actor
batch parameters must satisfy FSDP divisibility. Four-GPU RL is configurable
but has not been validated with a GPU run. Default phase steps scale with
raw Parquet rows, global batch and epoch count; prompt filtering can reduce
the available steps. Explicit step caps need enough epochs to reach them.
Remove `--dry-run` to launch training yourself.

## CPU loading memory threshold

Both training scripts default to `--workers 1 --prefetch-factor 1
--memory-limit-percent 90 --memory-wait-seconds 180`. SFT uses lazy tokenization,
one preprocessing process and no pinned memory. RL worker/prefetch settings
also apply to validation. These settings bound the multiprocessing waiting
queue; each distributed SFT rank has its own queue.

Before each batch fetch, the guard checks host and accessible cgroup v2 memory
pressure. Loading pauses at 90%, resumes below 88%, and fails after 180 seconds
of sustained pressure. It logs its pause/wait/resume state. This does not enforce
a hard limit on total RAM: model/optimizer CPU offload, active batches and other
processes are outside the waiting queue. Set `--workers 0` to eliminate the
multiprocessing prefetch queue; memory checks still run before batch fetches.

## Final Data Build Preflight Review (2026-09-27)

Do not resume the older balanced_1000 output: its first geometry run used the old verification policy and attempted 960 prompts for only one accepted row. Fresh strict candidates now use `data/sft/balanced_1000_v2/`; the final dataset name remains `data/sft/large/mixed_balanced_1000.jsonl`. Prepare re-normalizes selected raw train records using the existing deduplicated question index, reflecting the corrected reference and choice extraction. arXivQA reads `label` and preserves both choice-letter/text aliases; geometry/GeoQA/arXivQA include answer choices in the user question. LLaVA and SMR read assistant references from conversations. Missing-reference candidates are excluded. Original source assistant answers are references, not guaranteed human-verified gold labels; exact/numeric checks can reject valid open-ended paraphrases.

```bash
cd /mnt/d/Agent0-dev-codex/Agent0-VL
bash scripts/rebuild-balanced-data.sh prepare
bash scripts/rebuild-balanced-data.sh preflight
bash scripts/build-balanced-sft-with-teacher.sh
```

The teacher wrapper runs data preflight before loading the teacher. Preparation can take time because it scans raw data; it prints per-source candidate counts. Old candidate/output files are preserved. Preflight checks reference presence, image paths, candidate counts, total quotas and compatible state versions. Full input/prompt/config hashes are additionally checked by build_stream on resume.

Qwen3 teacher requests explicitly set `chat_template_kwargs.enable_thinking=false` so hidden reasoning does not consume the public-response budget. This is a teacher API/template adaptation; released Solver/Verifier/Repair prompt text is unchanged. Empty public content and token-limit-truncated responses are rejected, not accepted as trajectories. Fresh generation is still needed to assess real yield under the stricter rules. Four-GPU/64-concurrency teacher execution has not been validated by a new live generation run. All 18 local CPU tests passed, and real-source reference/choice plus empty/truncated response checks were performed. Candidate exhaustion fails; never loosen quotas or correctness gates to announce 1,000 rows.
