#!/bin/bash
# Production pipeline submission
#
# GPU layout (pending GPU measurement results from measure_gpu.job):
#   8 × vLLM  (p_8gpus,  8 GPU each) = 64 GPU, 32 instances  ~2.7h text gen
#   1 × tts_ua (p_32gpus, 4 nodes)   = 32 GPU, 128 workers   ~3.9h TTS ua
#   1 × tts_dc (p_32gpus, 4 nodes)   = 32 GPU, 128 workers   ~3.9h TTS dc
#   Total peak: 128 GPU
#   NOTE: Update N_PARALLEL_PER_GPU in tts_ua/dc.job based on measure_gpu results
#
# Resume: safe to re-run. Text gen skips existing dialogues (existing_ids check).
#         TTS skips done dialogues (done_ids check + done.flag per dialogue).
#
# Usage:
#   bash submit_all.sh          # launch everything
#   bash submit_all.sh --dry    # print commands without submitting

set -euo pipefail
REPO=/work/jaylin0418/TASTE2-TW-QWEN2.5-Data
DRY=${1:-}

submit() {
    if [ "$DRY" = "--dry" ]; then
        echo "[DRY] sbatch $*" >&2
        echo "00000"
    else
        sbatch --parsable "$@"
    fi
}

# Clear previous vLLM endpoint files
rm -f $REPO/logs/vllm_node_*.json

echo "=== Step 1: 8 × vLLM (p_8gpus, 64 GPU total, 32 instances) ==="
VLLM_JIDS=()
for i in $(seq 1 8); do
    JID=$(submit "$REPO/vllm_server/launch_vllm.job")
    VLLM_JIDS+=($JID)
    echo "  vllm $i: $JID"
done
FIRST_VLLM=${VLLM_JIDS[0]}

echo ""
echo "=== Step 2: Text generation (dev, N_SHARDS=4 sub-workers per topic) ==="
GEN_UA_JID=$(submit --dependency=after:${FIRST_VLLM} "$REPO/slurm/gen_text_ua.job")
GEN_DC_JID=$(submit --dependency=after:${FIRST_VLLM} "$REPO/slurm/gen_text_dc.job")
echo "  gen_ua: $GEN_UA_JID"
echo "  gen_dc: $GEN_DC_JID"

echo ""
echo "=== Step 3: TTS (p_32gpus, 4 nodes each, 128 workers each) ==="
TTS_UA_JID=$(submit --dependency=after:${GEN_UA_JID} "$REPO/slurm/tts_ua.job")
TTS_DC_JID=$(submit --dependency=after:${GEN_DC_JID} "$REPO/slurm/tts_dc.job")
echo "  tts_ua: $TTS_UA_JID"
echo "  tts_dc: $TTS_DC_JID"

echo ""
echo "=== All jobs submitted ==="
echo "p_8gpus  : ${VLLM_JIDS[*]}"
echo "dev      : $GEN_UA_JID  $GEN_DC_JID"
echo "p_32gpus : $TTS_UA_JID  $TTS_DC_JID"
echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "Expected timeline (estimates):"
echo "  t=0      vLLM start, gen starts, TTS waits for done.flags"
echo "  t~15min  vLLM ready, text gen begins"
echo "  t~2.7h   text gen done (32 instances × N_SHARDS=4 per topic)"
echo "  t~3.9h   TTS ua + dc both finish (128 workers each)"
echo ""
echo "To resume after interruption: re-run 'bash submit_all.sh'"
echo "  gen: skips existing dialogues (existing_ids check)"
echo "  TTS: skips done dialogues (done.flag per dialogue)"
