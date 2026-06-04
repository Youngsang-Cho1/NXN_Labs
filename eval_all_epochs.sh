#!/bin/bash
# Sweep FITB accuracy across every saved epoch for the disjoint-trained model.
# Runs sequentially because there's only one MPS device.

set -e
cd "$(dirname "$0")"

RESULTS_FILE="disjoint_sweep.csv"
echo "epoch,fitb_accuracy" > "$RESULTS_FILE"

for ep in 1 2 3 4 5 6 7 8 9 10 11; do
    ckpt="v3_disjoint_epoch_${ep}.pt"
    if [ ! -f "$ckpt" ]; then
        echo "skip: $ckpt not found"
        continue
    fi
    echo ""
    echo "==================================================="
    echo "  Evaluating epoch $ep"
    echo "==================================================="
    python eval_fitb.py --model_path "$ckpt" --split disjoint --num_samples 10000
    acc=$(cat fitb_accuracy.txt)
    echo "${ep},${acc}" >> "$RESULTS_FILE"
done

echo ""
echo "==================================================="
echo "  Sweep complete. Results:"
echo "==================================================="
cat "$RESULTS_FILE"