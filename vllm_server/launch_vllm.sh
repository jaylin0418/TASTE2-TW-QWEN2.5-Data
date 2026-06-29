#!/bin/bash
# ── USER CONFIGURATION ────────────────────────────────────────────────────────
MODEL_PATH=/work/jaylin0418/home_models/Qwen2.5-7B-Instruct
PORT=8000
TENSOR_PARALLEL=1   # 調整為可用 GPU 數
MAX_MODEL_LEN=8192
# ─────────────────────────────────────────────────────────────────────────────

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --port "$PORT" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    --enable-prefix-caching \
    --trust-remote-code
