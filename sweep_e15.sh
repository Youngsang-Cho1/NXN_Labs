#!/bin/bash
# Sweep FITB across all 15 epochs of a given prefix.
# Usage: ./sweep_e15.sh <save_prefix>
# Example: ./sweep_e15.sh v3_text_e15

set -e
cd "$(dirname "$0")"

PREFIX="${1:-v3_text_e15}"
RESULTS="sweep_${PREFIX}.csv"
echo "epoch,fitb_accuracy" > "$RESULTS"

for ep in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    ckpt="${PREFIX}_epoch_${ep}.pt"
    [ -f "$ckpt" ] || { echo "skip: $ckpt"; continue; }
    echo ""
    echo "==== $PREFIX epoch $ep ===="
    python eval_fitb.py --model_path "$ckpt" --split disjoint --num_samples 10000
    acc=$(cat fitb_accuracy.txt)
    echo "${ep},${acc}" >> "$RESULTS"
done

echo ""
echo "=== $PREFIX sweep done ==="
cat "$RESULTS"
