#!/bin/bash
# 在登入節點執行一次即可，建立專用 vLLM conda env
# Usage: bash vllm_server/install_vllm.sh

set -e

CONDA=/home/jaylin0418/miniconda3/bin/conda
ENV_NAME=vllm_py312

echo "Creating conda env: $ENV_NAME (Python 3.12)"
$CONDA create -n $ENV_NAME python=3.12 -y

PYTHON=/home/jaylin0418/miniconda3/envs/${ENV_NAME}/bin/python

echo "Installing vLLM (CUDA 12.1 wheel)..."
$PYTHON -m pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

echo "Verifying..."
$PYTHON -c "import vllm; print('vLLM', vllm.__version__, 'installed OK')"
echo "Done. VLLM_PYTHON=$PYTHON"
