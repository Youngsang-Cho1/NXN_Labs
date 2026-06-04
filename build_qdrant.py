"""
Build a Qdrant collection from polyvore_embeddings.pt.

Run once after `docker run ... qdrant/qdrant`.

For each item we store the SigLIP image vector and a clean `slot` label
derived from a zero-shot CLIP-style classifier (image_emb vs slot prompt embs).
This frees the app from having to keyword-match Polyvore's noisy `category`.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from tqdm import tqdm

import torch
import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType


COLLECTION = "polyvore_items"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBED_PATH = "polyvore_embeddings.pt"
SIGLIP_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"

# Slot prompts. Order matters only for human readability — the classifier picks
# whichever prompt has the highest cosine with the item image.
SLOT_PROMPTS = {
    "one-piece":  "a photo of a jumpsuit, romper, or playsuit",
    "dresses":    "a photo of a dress or gown",
    "tops":       "a photo of a shirt, t-shirt, blouse, or sweater",
    "bottoms":    "a photo of pants, jeans, a skirt, or shorts",
    "outerwear":  "a photo of a jacket, coat, or blazer",
    "shoes":      "a photo of shoes, sneakers, boots, or heels",
    "bags":       "a photo of a bag, purse, backpack, or clutch",
    "hats":       "a photo of a hat or cap",
    "jewelry":    "a photo of jewelry, a necklace, ring, or earrings",
    "accessories": "a photo of a fashion accessory like a scarf, belt, or sunglasses",
}


def encode_slot_prompts(device):
    """Encode each slot prompt once with SigLIP text encoder → [num_slots, 768]."""
    print(f"Loading {SIGLIP_MODEL} for slot-prompt encoding…")
    model, _, _ = open_clip.create_model_and_transforms(SIGLIP_MODEL)
    tokenizer = open_clip.get_tokenizer(SIGLIP_MODEL)
    model = model.to(device).eval()

    slot_names = list(SLOT_PROMPTS.keys())
    prompts = list(SLOT_PROMPTS.values())
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embs = model.encode_text(tokens, normalize=True)
    return slot_names, text_embs.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Index only first N items (None = all 251k)")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--classify_batch", type=int, default=4096,
                        help="How many item image vectors to classify per matmul")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading embeddings from {EMBED_PATH}…")
    embeddings = torch.load(EMBED_PATH, map_location="cpu", weights_only=False)
    item_ids = list(embeddings.keys())
    if args.limit:
        item_ids = item_ids[:args.limit]
    print(f"Indexing {len(item_ids):,} items")

    # Zero-shot slot classification: image_emb · slot_prompt_emb → argmax
    slot_names, slot_text_embs = encode_slot_prompts(device)
    slot_text_embs = slot_text_embs.to(device)  # [num_slots, 768]
    print(f"Slot vocabulary: {slot_names}")

    # Classify in chunks to avoid blowing past MPS memory
    print("Classifying every item into a slot…")
    item_to_slot = {}
    for start in tqdm(range(0, len(item_ids), args.classify_batch), desc="Slot classify"):
        chunk_ids = item_ids[start:start + args.classify_batch]
        img_batch = torch.stack([embeddings[iid]["image"] for iid in chunk_ids]).to(device)
        # Vectors are L2-normalized, so dot product == cosine sim
        sims = img_batch @ slot_text_embs.t()  # [chunk, num_slots]
        best = sims.argmax(dim=-1).cpu().tolist()
        for iid, bi in zip(chunk_ids, best):
            item_to_slot[iid] = slot_names[bi]

    # Sanity check: slot distribution
    from collections import Counter
    dist = Counter(item_to_slot.values())
    print(f"\nSlot distribution: {dict(dist)}\n")

    # Push to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="slot",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    points = []
    for idx, item_id in enumerate(tqdm(item_ids, desc="Upserting")):
        img_vec = embeddings[item_id]["image"].tolist()
        points.append(PointStruct(
            id=idx,
            vector=img_vec,
            payload={"item_id": item_id, "slot": item_to_slot[item_id]},
        ))
        if len(points) >= args.batch:
            client.upsert(collection_name=COLLECTION, points=points)
            points = []
    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    info = client.get_collection(COLLECTION)
    print(f"\n✅ Done. Collection '{COLLECTION}' has {info.points_count:,} points on {QDRANT_HOST}:{QDRANT_PORT}")


if __name__ == "__main__":
    main()