#!/bin/bash
# Launch vLLM on login node GPU (H100)
MODEL=/work/jaylin0418/models/Qwen2.5-32B-Instruct
PYTHON=/home/jaylin0418/miniconda3/envs/vllm_py312/bin/python
LOGFILE=/work/jaylin0418/TASTE2-TW-QWEN2.5-Data/logs/vllm_local.log

mkdir -p /work/jaylin0418/TASTE2-TW-QWEN2.5-Data/logs

echo "Starting vLLM on localhost:8000 ..."
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCHINDUCTOR_CACHE_DIR=/work/jaylin0418/.cache/torch_inductor
mkdir -p $TORCHINDUCTOR_CACHE_DIR
CUDA_VISIBLE_DEVICES=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --port 8000 \
    --served-model-name "Qwen/Qwen2.5-32B-Instruct" \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    > "$LOGFILE" 2>&1 &
echo $! > /work/jaylin0418/TASTE2-TW-QWEN2.5-Data/logs/vllm_local.pid
echo "PID: $! — log: $LOGFILE"
