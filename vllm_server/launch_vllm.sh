#!/bin/bash
# ── USER CONFIGURATION ────────────────────────────────────────────────────────
MODEL_PATH=/work/jaylin0418/models/Qwen2.5-32B-Instruct
PORT=8000
TENSOR_PARALLEL=2   # 32B bfloat16 ~64GB，需要 2 張 A100 80G
MAX_MODEL_LEN=8192
# ─────────────────────────────────────────────────────────────────────────────

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --port "$PORT" \
    --served-model-name "Qwen/Qwen2.5-32B-Instruct" \
    --enable-prefix-caching \
    --trust-remote-code
