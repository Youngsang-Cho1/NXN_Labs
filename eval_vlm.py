"""
VLM-as-judge evaluation for OutfitTransformer.

Generates N outfits with the trained model, asks a vision-language model
to score them on color harmony / style coherence / occasion fit / balance,
and saves the results to vlm_results.csv.

Provider is swappable via env: VLM_PROVIDER=gemini (default) or openrouter.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import io
import base64
import json
import random
import argparse
import csv
import glob
import time
from dataclasses import dataclass

import torch
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from model import OutfitTransformer

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Provider config
# ──────────────────────────────────────────────────────────────────────────────
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen-2-vl-7b-instruct:free",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


def make_client(provider: str):
    cfg = PROVIDERS[provider]
    key = os.getenv(cfg["api_key_env"])
    if not key:
        raise RuntimeError(f"Missing env var {cfg['api_key_env']} for provider {provider}")
    return OpenAI(api_key=key, base_url=cfg["base_url"]), cfg["model"]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt — designed for normalization with the team. Asks for 4 sub-scores
# and a single total, returned strictly as JSON.
# ──────────────────────────────────────────────────────────────────────────────
PROMPT = """You are a fashion stylist evaluating a complete outfit.

The image shows the items in this outfit side by side, left to right.

Rate the outfit on each criterion below from 0 to 100, then give a total from 0 to 100.

Criteria:
- color_harmony: do the colors work together?
- style_coherence: do the items share a consistent style (e.g. all casual, all formal)?
- occasion_fit: would this outfit make sense for some real occasion?
- balance: is the overall silhouette / proportion balanced?

Respond ONLY with a single JSON object, no prose, in this exact shape:
{
  "color_harmony": <int>,
  "style_coherence": <int>,
  "occasion_fit": <int>,
  "balance": <int>,
  "total": <int>,
  "reason": "<one short sentence>"
}"""


# ──────────────────────────────────────────────────────────────────────────────
# Outfit generation (mirrors app.py's autoregressive logic, headless)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_BLUEPRINT = ["tops", "bottoms", "shoes", "bags"]
DRESS_BLUEPRINT = ["outerwear", "shoes", "bags", "jewelry"]
INCOMPATIBLE_SLOTS = {
    "dresses": {"tops", "bottoms", "dresses"},
    "tops":    {"tops", "dresses"},
    "bottoms": {"bottoms", "dresses"},
    "shoes":   {"shoes"},
    "bags":    {"bags"},
    "outerwear": {"outerwear"},
    "hats":    {"hats"},
}


def map_to_blueprint(raw_cat: str) -> str:
    c = raw_cat.lower()
    rules = [
        (["shirt", "top", "blouse", "t-shirt", "sweater", "hoodie", "camisole", "knit"], "tops"),
        (["pant", "jean", "skirt", "bottom", "short", "legging", "trouser"], "bottoms"),
        (["shoe", "sneaker", "boot", "heel", "sandal", "flat", "wedge"], "shoes"),
        (["bag", "purse", "tote", "backpack", "clutch", "satchel"], "bags"),
        (["jacket", "coat", "outer", "blazer", "cardigan", "vest"], "outerwear"),
        (["dress", "gown", "romper"], "dresses"),
        (["jewelry", "necklace", "ring", "earring", "bracelet", "watch"], "jewelry"),
        (["hat", "cap", "beanie"], "hats"),
    ]
    for keywords, label in rules:
        if any(k in c for k in keywords):
            return label
    return "accessories"


def best_checkpoint() -> str:
    preferred = ["v3_epoch_10.pt", "v3_epoch_11.pt", "v3_epoch_9.pt", "best_v3.pt"]
    for p in preferred:
        if os.path.exists(p):
            return p
    candidates = glob.glob("v3_epoch_*.pt") or glob.glob("v2full_epoch_*.pt")
    if not candidates:
        raise FileNotFoundError("No checkpoint found")
    return sorted(candidates, key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]


@dataclass
class Catalogue:
    model: OutfitTransformer
    embeddings: dict
    db_features: torch.Tensor
    db_categories: list
    db_images: list
    db_ids: list
    device: torch.device


def load_catalogue(db_size: int = 50_000) -> Catalogue:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = best_checkpoint()
    print(f"Loading model from {ckpt}")
    blob = torch.load(ckpt, map_location=device, weights_only=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        state_dict = blob["state_dict"]
        num_layers = blob.get("hparams", {}).get("num_layers", 4)
    else:
        state_dict = blob
        num_layers = 4
    model = OutfitTransformer(num_layers=num_layers).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Loading embeddings (cap = {db_size})…")
    embeddings = torch.load("polyvore_embeddings.pt", map_location="cpu", weights_only=False)
    all_ids = list(embeddings.keys())[:db_size]

    polyvore_items = load_dataset("owj0421/polyvore", split="data")
    db_items = polyvore_items.select(range(min(db_size, len(polyvore_items))))
    id_to_idx = {item["item_id"]: i for i, item in enumerate(db_items)}

    feats, cats, imgs, ids = [], [], [], []
    for item_id in all_ids:
        if item_id not in id_to_idx:
            continue
        item = db_items[id_to_idx[item_id]]
        img = item["image"]
        if getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")
        cat = str(item.get("category") or item.get("title") or "").lower()
        feats.append(embeddings[item_id]["image"])
        cats.append(cat)
        imgs.append(img)
        ids.append(item_id)

    db_features = torch.stack(feats).to(device)
    print(f"Catalogue ready: {len(ids):,} items")
    return Catalogue(model, embeddings, db_features, cats, imgs, ids, device)


def search_top_k(target_emb, cat: Catalogue, require_keyword, k, exclude_ids):
    sims = (cat.db_features @ target_emb.squeeze(0)).squeeze(-1)
    order = torch.argsort(sims, descending=True).tolist()
    results = []
    for idx in order:
        if cat.db_ids[idx] in exclude_ids:
            continue
        if require_keyword and require_keyword.lower() not in cat.db_categories[idx]:
            continue
        results.append((cat.db_images[idx], cat.db_ids[idx]))
        if len(results) >= k:
            break
    return results


def generate_outfit(seed_idx: int, cat: Catalogue, use_text_hint: bool = True):
    """Return list of (image, item_id, slot) including the seed at index 0."""
    seed_id = cat.db_ids[seed_idx]
    seed_img = cat.db_images[seed_idx]
    seed_cat = map_to_blueprint(cat.db_categories[seed_idx])

    blocked = INCOMPATIBLE_SLOTS.get(seed_cat, {seed_cat})
    if seed_cat == "dresses":
        queries = [c for c in DRESS_BLUEPRINT if c not in blocked]
    else:
        queries = [c for c in DEFAULT_BLUEPRINT if c not in blocked]

    ctx_imgs = [cat.embeddings[seed_id]["image"].unsqueeze(0)]
    ctx_txts = [cat.embeddings[seed_id]["text"].unsqueeze(0)]
    used = {seed_id}
    items = [(seed_img, seed_id, seed_cat)]

    for query in queries:
        ctx_feat = torch.cat(ctx_imgs, dim=0).unsqueeze(0).to(cat.device)
        ctx_txt = torch.cat(ctx_txts, dim=0).unsqueeze(0).to(cat.device)
        ctx_mask = torch.zeros((1, ctx_feat.shape[1]), dtype=torch.bool, device=cat.device)

        text_features = None
        if use_text_hint:
            token = cat.model.tokenizer([query]).to(cat.device)
            with torch.no_grad():
                text_features = cat.model.siglip.encode_text(token, normalize=True)

        with torch.no_grad():
            ideal = cat.model.encode_features(
                ctx_feat, ctx_mask, text_features, context_text_features=ctx_txt,
            )

        results = search_top_k(ideal, cat, require_keyword=query, k=1, exclude_ids=used)
        if not results:
            continue
        best_img, best_id = results[0]
        used.add(best_id)
        ctx_imgs.append(cat.embeddings[best_id]["image"].unsqueeze(0))
        ctx_txts.append(cat.embeddings[best_id]["text"].unsqueeze(0))
        items.append((best_img, best_id, query))

    return items


# ──────────────────────────────────────────────────────────────────────────────
# Image grid + VLM call
# ──────────────────────────────────────────────────────────────────────────────
def compose_grid(images, cell=256, pad=12) -> Image.Image:
    n = len(images)
    grid_w = cell * n + pad * (n + 1)
    grid_h = cell + pad * 2
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for i, img in enumerate(images):
        thumb = img.copy()
        thumb.thumbnail((cell, cell), Image.LANCZOS)
        x = pad + i * (cell + pad) + (cell - thumb.width) // 2
        y = pad + (cell - thumb.height) // 2
        grid.paste(thumb, (x, y))
    return grid


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def score_outfit(client: OpenAI, model_name: str, grid: Image.Image, retries: int = 2) -> dict:
    b64 = img_to_b64(grid)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            # Strip code-fence if VLM wraps the JSON
            if text.startswith("```"):
                text = text.strip("`").lstrip("json").strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            if attempt == retries:
                return {"error": f"json_decode: {e}", "raw": text[:200]}
        except Exception as e:
            if attempt == retries:
                return {"error": str(e)[:200]}
            time.sleep(2 ** attempt)
    return {"error": "unreachable"}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of outfits to evaluate")
    parser.add_argument("--db_size", type=int, default=50000)
    parser.add_argument("--output", type=str, default="vlm_results.csv")
    parser.add_argument("--save_grids_to", type=str, default=None,
                        help="Optional dir to save outfit grid PNGs for inspection")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    provider = os.getenv("VLM_PROVIDER", "gemini")
    print(f"Provider: {provider}")
    client, model_name = make_client(provider)
    print(f"Model: {model_name}")

    cat = load_catalogue(db_size=args.db_size)
    random.seed(args.seed)
    seed_indices = random.sample(range(len(cat.db_ids)), args.n)

    if args.save_grids_to:
        os.makedirs(args.save_grids_to, exist_ok=True)

    rows = []
    totals = []
    for i, seed_idx in enumerate(seed_indices, start=1):
        items = generate_outfit(seed_idx, cat)
        grid = compose_grid([img for img, _, _ in items])
        if args.save_grids_to:
            grid.save(os.path.join(args.save_grids_to, f"outfit_{i:03d}.png"))

        result = score_outfit(client, model_name, grid)
        time.sleep(5)  # stay under free-tier rate limits (~15 req/min)
        total = result.get("total")
        if isinstance(total, (int, float)):
            totals.append(total)

        slots = "+".join(slot for _, _, slot in items)
        print(f"[{i:>3}/{args.n}] seed={seed_idx:<6} slots={slots:<40} → {result}")

        rows.append({
            "i": i,
            "seed_idx": seed_idx,
            "seed_id": items[0][1],
            "slots": slots,
            "color_harmony": result.get("color_harmony"),
            "style_coherence": result.get("style_coherence"),
            "occasion_fit": result.get("occasion_fit"),
            "balance": result.get("balance"),
            "total": result.get("total"),
            "reason": result.get("reason"),
            "error": result.get("error"),
        })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {args.output}")
    if totals:
        print(f"Mean total: {sum(totals)/len(totals):.1f} / 100  (n={len(totals)})")


if __name__ == "__main__":
    main()