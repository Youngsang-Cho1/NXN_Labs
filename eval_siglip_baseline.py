"""
Frozen SigLIP baseline on Polyvore disjoint FITB.

No transformer, no training — just aggregate the context item embeddings and
pick the candidate closest to that aggregate. Three simple aggregators:

  - mean:   centroid of context embeddings, cosine-rank candidates
  - max:    candidate's max cosine to any context item
  - sum:    candidate's sum of cosines over all context items

This isolates how much of our +4.30 over the CVPR 2022 paper comes from the
backbone alone, vs the learned transformer.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="disjoint", choices=["nondisjoint", "disjoint"])
    parser.add_argument("--num_samples", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading cached SigLIP embeddings…")
    embeddings = torch.load("polyvore_embeddings.pt", map_location="cpu", weights_only=True)

    print("Loading item_id → row map…")
    polyvore_items = load_dataset("owj0421/polyvore", split="data")
    item_id_map = {iid: i for i, iid in enumerate(polyvore_items["item_id"])}

    print(f"Loading {args.split} outfits to resolve set_id+index → item_id…")
    nd_def = load_dataset("owj0421/polyvore-outfits", f"{args.split}_default")
    set_to_item = {}
    for split in ["train", "validation", "test"]:
        for row in nd_def[split]:
            for it in row["items"]:
                set_to_item[f"{row['set_id']}_{it['index']}"] = it["item_id"]

    fitb_test = load_dataset(
        "owj0421/polyvore-outfits", f"{args.split}_fill_in_the_blank", split="test"
    )
    fitb_test = fitb_test.select(range(min(args.num_samples, len(fitb_test))))

    def emb_for(key):
        iid = set_to_item.get(key)
        if iid is None or iid not in embeddings:
            return None
        return embeddings[iid]["image"]  # already L2-normalized

    correct = {"mean": 0, "max": 0, "sum": 0}
    total = 0

    for row in tqdm(fitb_test):
        ctx = [emb_for(k) for k in row["items"]]
        ctx = [e for e in ctx if e is not None]
        if not ctx:
            continue

        cands = [emb_for(k) for k in row["candidates"]]
        if len(cands) != 4 or any(c is None for c in cands):
            continue

        ctx_t = torch.stack(ctx).to(device)         # [N, D]
        cand_t = torch.stack(cands).to(device)      # [4, D]
        label = row["label"]

        # mean: cosine(candidate, mean(context))
        centroid = F.normalize(ctx_t.mean(dim=0, keepdim=True), p=2, dim=-1)  # [1, D]
        mean_scores = (cand_t * centroid).sum(dim=-1)                        # [4]
        if mean_scores.argmax().item() == label:
            correct["mean"] += 1

        # max: max over context of cosine(candidate, ctx_i)
        sim = cand_t @ ctx_t.t()                                              # [4, N]
        if sim.max(dim=1).values.argmax().item() == label:
            correct["max"] += 1

        # sum: sum over context of cosine
        if sim.sum(dim=1).argmax().item() == label:
            correct["sum"] += 1

        total += 1

    print(f"\nEvaluated on {total} questions ({args.split} split)")
    for k in ("mean", "max", "sum"):
        acc = 100.0 * correct[k] / max(total, 1)
        print(f"  SigLIP-{k:<4}: {acc:.2f}%")


if __name__ == "__main__":
    main()
