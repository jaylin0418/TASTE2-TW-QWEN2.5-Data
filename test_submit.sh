#!/bin/bash
# test_submit.sh — Small-scale pipeline test before production
# Tests: vLLM fix, text gen (ua+dc), TTS parallelism (2 GPU), data quality
#
# 3 topics × 2 scenarios × 2 types = 12 dialogues total
# After completion, review:
#   - Text:  output/user_agent_test/  output/daily_conv_test/
#   - Audio: tts_output/user_agent_test/  tts_output/daily_conv_test/
#
# Usage:
#   bash test_submit.sh          # submit
#   bash test_submit.sh --dry    # dry run (print only)

set -euo pipefail
REPO=/work/jaylin0418/TASTE2-TW-QWEN2.5-Data
DRY=${1:-}

submit() {
    if [ "$DRY" = "--dry" ]; then
        echo "[DRY] sbatch $*" >&2
        echo "99999"
    else
        sbatch --parsable "$@"
    fi
}

echo "=== Step 1: vLLM server ==="
VLLM_JID=$(submit "$REPO/vllm_server/launch_vllm.job")
echo "  vllm job: $VLLM_JID"

echo ""
echo "=== Step 2: Text gen test (3 topics × 2 scenarios, ua + dc) ==="
TEXT_JID=$(submit --dependency=after:${VLLM_JID} "$REPO/slurm/test_text.job")
echo "  test_text job: $TEXT_JID"

echo ""
echo "=== Step 3: TTS test (2 GPU, all 3 topics) ==="
TTS_JID=$(submit --dependency=afterok:${TEXT_JID} "$REPO/slurm/test_tts.job")
echo "  test_tts job: $TTS_JID"

echo ""
echo "=== Jobs submitted ==="
echo "Monitor: squeue -u \$USER"
echo ""
echo "After jobs finish, review outputs:"
echo "  Text (ua): $REPO/output/user_agent_test/"
echo "  Text (dc): $REPO/output/daily_conv_test/"
echo "  Audio:     $REPO/tts_output/user_agent_test/"
echo "             $REPO/tts_output/daily_conv_test/"
echo ""
echo "Check sample dialogue text:"
echo "  cat $REPO/output/user_agent_test/烹飪/dialogues.jsonl | python3 -m json.tool | head -60"
echo "  cat $REPO/output/daily_conv_test/烹飪/dialogues.jsonl | python3 -m json.tool | head -60"
