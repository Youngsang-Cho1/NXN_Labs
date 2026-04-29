import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from datasets import load_dataset
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

from model import OutfitTransformer

EMBED_PATH = "polyvore_embeddings.pt"
DB_SIZE = 5000

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {device}")

    # ---- 1. Load Model ----
    print("Loading Trained Model (outfit_transformer_epoch_3.pt)...")
    model = OutfitTransformer(num_layers=4).to(device)
    try:
        model.load_state_dict(torch.load("outfit_transformer_epoch_3.pt", weights_only=True, map_location=device))
        print("Model weights fully loaded.")
    except Exception as e:
        print(f"Could not load weights ({e}), using untrained weights for architecture test...")
    model.eval()

    # ---- 2. Load Pre-computed Embeddings (polyvore_embeddings.pt) ----
    print(f"\nLoading pre-computed embeddings from {EMBED_PATH}...")
    embeddings_dict = torch.load(EMBED_PATH, map_location="cpu", weights_only=False)
    all_item_ids = list(embeddings_dict.keys())[:DB_SIZE]
    print(f"Loaded {len(all_item_ids)} items from vector DB.")

    # ---- 3. Load Dataset (images + metadata only, NO SigLIP encoding) ----
    print(f"Loading Polyvore dataset for images & metadata...")
    polyvore_items = load_dataset("owj0421/polyvore", split="data")
    db_items = polyvore_items.select(range(DB_SIZE))

    # Build index: item_id -> dataset row index
    id_to_idx = {item["item_id"]: i for i, item in enumerate(db_items)}

    db_features = []
    db_categories = []
    db_images = []
    db_item_ids_ordered = []

    print("Assembling vector DB from cached embeddings (no SigLIP needed)...")
    for item_id in all_item_ids:
        if item_id not in id_to_idx:
            continue
        idx = id_to_idx[item_id]
        item = db_items[idx]

        img = item["image"]
        if getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")

        cat = str(item.get("category") or "")
        if not cat: cat = str(item.get("title") or "")
        if not cat: cat = str(item.get("description") or "")
        cat = cat.lower()

        db_features.append(embeddings_dict[item_id]["image"])
        db_categories.append(cat)
        db_images.append(img)
        db_item_ids_ordered.append(item_id)

    db_features_tensor = torch.stack(db_features).to(device)  # [N, 768]
    print(f"Vector DB ready: {db_features_tensor.shape[0]} items × {db_features_tensor.shape[1]}-D")

    # ---- 4. Search Function ----
    def search_closest_item(target_emb, require_keyword=None):
        distances = torch.norm(db_features_tensor - target_emb, p=2, dim=-1)
        sorted_indices = torch.argsort(distances)
        for idx in sorted_indices:
            idx = idx.item()
            if require_keyword:
                if require_keyword.lower() in db_categories[idx]:
                    return db_images[idx], db_item_ids_ordered[idx]
            else:
                return db_images[idx], db_item_ids_ordered[idx]
        return db_images[sorted_indices[0].item()], db_item_ids_ordered[sorted_indices[0].item()]

    # ---- 5. Autoregressive Retrieval Demo ----
    print("\n Starting Autoregressive Outfit Generation!")

    seed_idx = random.randint(0, len(db_item_ids_ordered) - 1)
    seed_item_id = db_item_ids_ordered[seed_idx]
    seed_img = db_images[seed_idx]

    # Pre-computed embedding for seed item
    context_embeddings = [embeddings_dict[seed_item_id]["image"].unsqueeze(0)]  # [1, 768]
    context_images = [seed_img]

    dataset_idx = id_to_idx.get(seed_item_id, seed_idx)
    seed_name = db_items[dataset_idx].get("title") or db_items[dataset_idx].get("category") or "Item"
    print(f"Random Seed Item (index {seed_idx}): {seed_name}")

    queries = ["pants", "top", "shoe", "bag"]

    for query in queries:
        print(f"\nThinking about what '{query}' would look good...")

        # Stack pre-computed context embeddings — no SigLIP call!
        context_img_features = torch.cat(context_embeddings, dim=0).unsqueeze(0).to(device)  # [1, Seq, 768]
        context_mask = torch.zeros((1, context_img_features.shape[1]), dtype=torch.bool).to(device)

        # Only text needs encoding (fast)
        target_token = model.tokenizer([query]).to(device)
        with torch.no_grad():
            text_features = model.siglip.encode_text(target_token, normalize=True)
            ideal_emb = model.encode_features(context_img_features, context_mask, text_features)

        retrieved_img, retrieved_id = search_closest_item(ideal_emb, require_keyword=query)
        print(f" Found the perfect '{query}'!")

        context_images.append(retrieved_img)
        # Use cached embedding of retrieved item for next step
        context_embeddings.append(embeddings_dict[retrieved_id]["image"].unsqueeze(0))

    print(f"\n Outfit Complete! Total items matched: {len(context_images)}")

    # ---- 6. Visualization ----
    fig, axes = plt.subplots(1, len(context_images), figsize=(15, 5))
    fig.suptitle("Autoregressive Outfit Generation (OutfitTransformer)", fontsize=16)
    for i, ax in enumerate(axes):
        ax.imshow(context_images[i])
        ax.axis("off")
        title = "Seed Item" if i == 0 else f"Retrieved: {queries[i-1]}"
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    save_path = "outfit_demo_result.png"
    plt.savefig(save_path, dpi=150)
    print(f" Outfit visual saved to: {save_path}")

if __name__ == "__main__":
    main()
