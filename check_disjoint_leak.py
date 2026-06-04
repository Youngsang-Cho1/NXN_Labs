"""
Check whether item_ids used in nondisjoint_default training overlap with
items appearing in the disjoint FITB test set.

If the intersection is non-empty, the +9.12 disjoint result is contaminated.
If it's empty, disjoint generalization is clean.
"""
from datasets import load_dataset


def collect_train_item_ids():
    ds = load_dataset("owj0421/polyvore-outfits", "nondisjoint_default", split="train")
    ids = set()
    for row in ds:
        for it in row["items"]:
            ids.add(it["item_id"])
    return ids


def collect_disjoint_test_item_ids():
    # disjoint FITB test references items by set_id+index, so we have to resolve
    # via the disjoint_default outfits to get real item_ids.
    nd_def = load_dataset("owj0421/polyvore-outfits", "disjoint_default")
    set_to_item = {}
    for split in ["train", "validation", "test"]:
        for row in nd_def[split]:
            for it in row["items"]:
                set_to_item[f"{row['set_id']}_{it['index']}"] = it["item_id"]

    fitb_test = load_dataset(
        "owj0421/polyvore-outfits", "disjoint_fill_in_the_blank", split="test"
    )
    ids = set()
    for row in fitb_test:
        for key in row["items"]:
            if key in set_to_item:
                ids.add(set_to_item[key])
        for key in row["candidates"]:
            if key in set_to_item:
                ids.add(set_to_item[key])
    return ids


def main():
    print("Collecting nondisjoint train item_ids...")
    train_ids = collect_train_item_ids()
    print(f"  {len(train_ids):,} unique train items")

    print("Collecting disjoint FITB test item_ids (context + candidates)...")
    test_ids = collect_disjoint_test_item_ids()
    print(f"  {len(test_ids):,} unique disjoint test items")

    overlap = train_ids & test_ids
    pct = 100.0 * len(overlap) / max(len(test_ids), 1)
    print()
    print(f"Overlap: {len(overlap):,} items ({pct:.2f}% of disjoint test)")
    if len(overlap) == 0:
        print("CLEAN: no leak. Disjoint result is honest.")
    elif pct < 5:
        print("MINOR: small overlap, likely incidental.")
    else:
        print("LEAK: substantial overlap. Disjoint result is contaminated.")


if __name__ == "__main__":
    main()
