#!/bin/bash
# Run all three ablations sequentially on the disjoint split.
# Each run takes ~1.5h; sweep eval adds ~30min each -> total ~6 hours.
#
# Baseline (already done): v3_disjoint_epoch_11.pt = 63.78%
# Ablations measure how much each design choice contributes.

set -e
cd "$(dirname "$0")"

SPLIT=disjoint
EPOCHS=11

run_ablation () {
    local name="$1"
    shift
    local prefix="ablation_${name}"
    local final_ckpt="${prefix}_epoch_${EPOCHS}.pt"

    echo ""
    echo "==============================================="
    echo "  Ablation: $name"
    echo "  Extra args: $@"
    echo "==============================================="

    if [ -f "$final_ckpt" ]; then
        echo "  ✓ $final_ckpt already exists, skipping training"
    else
        python train.py --epochs $EPOCHS --split $SPLIT --save_prefix "$prefix" "$@"
    fi

    # Eval only the final epoch (full sweep is overkill for ablations;
    # baseline showed monotonic improvement, last epoch ~= best).
    echo ""
    echo "  Evaluating $final_ckpt …"
    python eval_fitb.py --model_path "$final_ckpt" --split $SPLIT --num_samples 10000
    acc=$(cat fitb_accuracy.txt)
    echo "${name},${acc}" >> ablation_results.csv
}

# Init results file with baseline for easy comparison
echo "ablation,fitb_accuracy" > ablation_results.csv
echo "baseline_full,63.78" >> ablation_results.csv

# 1. No per-item text fusion in context
run_ablation "no_context_text" --no_context_text

# 2. Hard negatives = naive top-K (instead of mid-sim band)
run_ablation "topk_neg" --neg_strategy topk

# 3. No sample-wise text dropout (mask_token under-trained for text-free FITB)
run_ablation "no_text_dropout" --text_dropout 0.0

echo ""
echo "==============================================="
echo "  All ablations complete. Results:"
echo "==============================================="
cat ablation_results.csv
