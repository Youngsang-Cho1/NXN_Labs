"""
OutfitTransformer Hyperparameter Grid Search
Automatically tries different parameter combinations and logs the best result.
"""
import subprocess
import json
import os
from itertools import product
from datetime import datetime

# ============================
# 🔍 Search Space Definition
# ============================
SEARCH_SPACE = {
    "lr":           [1e-3, 1e-4, 5e-5],
    "epochs":       [2, 3],
    "text_dropout": [0.1, 0.2, 0.3],
    "num_layers":   [4, 6],
}

RESULTS_FILE = "grid_search_results.json"

def run_train(params: dict) -> str:
    """Runs train.py with given params and returns the saved model filename."""
    cmd = [
        "python", "train.py",
        f"--lr={params['lr']}",
        f"--epochs={params['epochs']}",
        f"--text_dropout={params['text_dropout']}",
        f"--num_layers={params['num_layers']}",
    ]
    print(f"\n🚂 Training with: {params}")
    subprocess.run(cmd, check=True)
    # train.py saves as outfit_transformer_epoch_N.pt
    return f"outfit_transformer_epoch_{params['epochs']}.pt"


def run_eval(model_path: str) -> float:
    """Runs eval_fitb.py for the given model checkpoint and returns accuracy."""
    cmd = ["python", "eval_fitb.py", f"--model_path={model_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Parse "Final FITB Accuracy: XX.XX%" from stdout
    for line in result.stdout.split("\n"):
        if "Final FITB Accuracy" in line:
            try:
                acc = float(line.split(":")[1].strip().replace("%", ""))
                return acc
            except:
                pass
    return 0.0


def main():
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = list(product(*values))
    total = len(combos)

    print(f"\n🔬 Grid Search: {total} combinations to try")
    print(f"   Params: {keys}\n")

    results = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{total}] Testing: {params}")
        print(f"{'='*60}")

        try:
            model_path = run_train(params)
            accuracy = run_eval(model_path)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e}")
            accuracy = 0.0

        entry = {**params, "fitb_accuracy": accuracy, "timestamp": datetime.now().isoformat()}
        results.append(entry)

        # Sort and save after every run
        results.sort(key=lambda x: x["fitb_accuracy"], reverse=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Accuracy: {accuracy:.2f}%  |  Best so far: {results[0]['fitb_accuracy']:.2f}%")
        print(f"   Best params: { {k: results[0][k] for k in keys} }")

    print(f"\n{'='*60}")
    print(f"🏆 GRID SEARCH COMPLETE!")
    print(f"   Best Accuracy : {results[0]['fitb_accuracy']:.2f}%")
    print(f"   Best Params   : { {k: results[0][k] for k in keys} }")
    print(f"   Full results  : {RESULTS_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
