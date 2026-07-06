#!/bin/bash
# Per-node worker for DC remaining (Mode A: flat remaining_daily_conv.jsonl).
# All env vars exported by tts_dc_remaining.job.

NODE_OFFSET=$((SLURM_NODEID * N_WORKERS_PER_NODE))

echo "[node $SLURM_NODEID / $(hostname)] launching $N_WORKERS_PER_NODE workers (offset $NODE_OFFSET)"

for LOCAL_W in $(seq 0 $((N_WORKERS_PER_NODE - 1))); do
    WORKER_ID=$((NODE_OFFSET + LOCAL_W))
    GPU_ID=$((LOCAL_W / N_PARALLEL_PER_GPU))
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $REPO/tts/tts_runner.py \
        --input         "$INPUT_FILE" \
        --output        "$OUTPUT_DIR" \
        --config        "$REPO/conf/base.yaml" \
        --indextts-dir  "$INDEXTTS_DIR" \
        --ref-pool      "$REF_POOL" \
        --breezy-repo   "$BREEZY_REPO" \
        --breezy-python "$BREEZY_PYTHON" \
        --breezy-model  "$BREEZY_MODEL" \
        --worker-id     "$WORKER_ID" \
        --num-workers   "$N_TOTAL_WORKERS" \
        >> "$LOG_DIR/tts_dc_rem_node${SLURM_NODEID}_w${LOCAL_W}.log" 2>&1 &
done

wait
echo "[node $SLURM_NODEID] all workers done"
