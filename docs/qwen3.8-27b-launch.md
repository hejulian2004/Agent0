# Qwen3.8-27B Teacher Launch Reference

## Canonical local profile

The verified local launcher is `/mnt/d/qwen3.8-27B/start_qwen3_8_27b_vllm_local.sh`, using vLLM 0.29.0 from `/mnt/d/qwen3.8-27B/.venv`. It serves `/mnt/d/qwen3.8-27B/model` as `qwen3.8-27b` on `127.0.0.1:8000`, with GPUs `2,3`, tensor parallelism 2, BF16 model weights, FP8 KV cache, 24,576-token context, 8,192 batched tokens, prefix caching, chunked prefill, `qwen3_coder` tool parsing, and the `qwen3` reasoning parser.

Start it without exposing the key:

```bash
export VLLM_API_KEY="$(tr -d '\r\n' < /mnt/d/Agent0/bench/.api_key)"
cd /mnt/d/qwen3.8-27B
bash start_qwen3_8_27b_vllm_local.sh
```

The checked launcher hard-codes `--max-num-seqs 1`. For the SFT builder's 64-way requests, use the same flags with `--max-num-seqs 64` explicitly:

```bash
CUDA_VISIBLE_DEVICES=2,3 /mnt/d/qwen3.8-27B/.venv/bin/vllm serve /mnt/d/qwen3.8-27B/model \
  --served-model-name qwen3.8-27b --tensor-parallel-size 2 \
  --max-model-len 24576 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --kv-cache-dtype fp8 --gpu-memory-utilization 0.95 \
  --enable-prefix-caching --mamba-cache-mode align \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --mm-processor-cache-gb 0.5 --host 127.0.0.1 --port 8000 \
  --api-key "$VLLM_API_KEY"
```

Before SFT, check `GET http://127.0.0.1:8000/v1/models` with the bearer key and confirm the model id is `qwen3.8-27b`. Stop it with `Ctrl-C` or a targeted `SIGTERM` only after generation finishes. The alternative `start_qwen3_8_27b_vllm_requested.sh` binds `0.0.0.0` and enables MTP/KV offload; historical logs show an OOM under visual requests, so it is not the default SFT profile.
