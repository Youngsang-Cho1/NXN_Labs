import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import glob
import random

import streamlit as st
import torch
from datasets import load_dataset

from model import OutfitTransformer

# ──────────────────────────────────────────────────────────────────────────────
# Page config & global styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OutfitTransformer · AI Stylist",
    page_icon="🧥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Tighten default Streamlit chrome */
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }
      header[data-testid="stHeader"] { background: transparent; }
      #MainMenu, footer { visibility: hidden; }

      /* Typography */
      html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
      }
      h1.app-title {
        font-weight: 600; font-size: 2.0rem; letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
      }
      p.app-subtitle {
        color: #6b7280; font-size: 0.95rem; margin-top: 0; margin-bottom: 1.6rem;
      }

      /* Section headers */
      h2.section { font-weight: 600; font-size: 1.15rem; letter-spacing: -0.01em;
        margin-top: 1.8rem; margin-bottom: 0.8rem; color: #111827; }
      h2.section .muted { color: #9ca3af; font-weight: 400; font-size: 0.9rem; margin-left: 0.5rem;}

      /* Item card */
      .item-card {
        border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px;
        background: #ffffff; transition: box-shadow .15s ease;
      }
      .item-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.05); }
      .item-card.selected { border-color: #111827; box-shadow: 0 0 0 1px #111827; }
      .item-label {
        font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: #6b7280; text-align: center; margin-top: 8px;
      }
      .item-label.primary { color: #111827; font-weight: 600; }

      /* Tag pills */
      .pill {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.75rem; background: #f3f4f6; color: #374151;
        margin-right: 6px; margin-bottom: 4px;
      }
      .pill.dark { background: #111827; color: #fafafa; }

      /* Stat strip */
      .stat-strip { display:flex; gap: 18px; margin: 0.4rem 0 1.2rem 0; }
      .stat { font-size: 0.82rem; color: #6b7280;}
      .stat b { color: #111827; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="app-title">OutfitTransformer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Pick a seed garment. The model assembles a coordinated outfit autoregressively.</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
EMBED_PATH = "polyvore_embeddings.pt"
DB_SIZE = 5000
DEFAULT_BLUEPRINT = ["tops", "bottoms", "shoes", "bags"]
DRESS_BLUEPRINT = ["shoes", "bags", "jewelry"]
AVAILABLE_CATEGORIES = ["tops", "bottoms", "dresses", "outerwear", "shoes", "bags", "accessories", "jewelry", "hats"]


def latest_checkpoint() -> str | None:
    ckpts = sorted(
        glob.glob("outfit_transformer_epoch_*.pt"),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    )
    return ckpts[-1] if ckpts else None


# ──────────────────────────────────────────────────────────────────────────────
# Asset loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = OutfitTransformer(num_layers=4).to(device)
    ckpt = latest_checkpoint()
    ckpt_label = "random init"
    if ckpt:
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True),
            strict=False,
        )
        ckpt_label = ckpt
    model.eval()

    embeddings_dict = torch.load(EMBED_PATH, map_location="cpu", weights_only=False)
    all_item_ids = list(embeddings_dict.keys())[:DB_SIZE]

    polyvore_items = load_dataset("owj0421/polyvore", split="data")
    db_items = polyvore_items.select(range(DB_SIZE))
    id_to_idx = {item["item_id"]: i for i, item in enumerate(db_items)}

    db_features, db_categories, db_images, db_ids = [], [], [], []
    for item_id in all_item_ids:
        if item_id not in id_to_idx:
            continue
        item = db_items[id_to_idx[item_id]]
        img = item["image"]
        if getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")
        cat = str(item.get("category") or item.get("title") or item.get("description") or "").lower()

        db_features.append(embeddings_dict[item_id]["image"])
        db_categories.append(cat)
        db_images.append(img)
        db_ids.append(item_id)

    db_features_tensor = torch.stack(db_features).to(device)
    return {
        "model": model,
        "embeddings": embeddings_dict,
        "db_features": db_features_tensor,
        "db_categories": db_categories,
        "db_images": db_images,
        "db_ids": db_ids,
        "device": device,
        "ckpt": ckpt_label,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────
def search_top_k(target_emb, db_features, db_categories, db_images, db_ids,
                 require_keyword: str | None = None, k: int = 3, exclude_ids: set | None = None):
    """Cosine similarity (vectors are L2-normalized)."""
    sims = (db_features @ target_emb.squeeze(0)).squeeze(-1)
    order = torch.argsort(sims, descending=True)
    exclude_ids = exclude_ids or set()

    results = []
    for idx in order.tolist():
        if db_ids[idx] in exclude_ids:
            continue
        if require_keyword and require_keyword.lower() not in db_categories[idx]:
            continue
        results.append((db_images[idx], db_ids[idx]))
        if len(results) >= k:
            break

    if not results:  # fallback ignoring keyword
        for idx in order[:k].tolist():
            results.append((db_images[idx], db_ids[idx]))
    return results


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


# ──────────────────────────────────────────────────────────────────────────────
# Boot
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading model and vector index…"):
    A = load_assets()

model = A["model"]
embeddings = A["embeddings"]
db_features = A["db_features"]
db_categories = A["db_categories"]
db_images = A["db_images"]
db_ids = A["db_ids"]
device = A["device"]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Seed")
    if "seed_idx" not in st.session_state:
        st.session_state.seed_idx = random.randint(0, len(db_ids) - 1)

    if st.button("Shuffle seed", use_container_width=True):
        st.session_state.seed_idx = random.randint(0, len(db_ids) - 1)

    st.session_state.seed_idx = st.number_input(
        f"Seed index (0 – {len(db_ids) - 1})",
        min_value=0,
        max_value=len(db_ids) - 1,
        value=st.session_state.seed_idx,
        step=1,
    )

    seed_idx = st.session_state.seed_idx
    seed_item_id = db_ids[seed_idx]
    seed_img = db_images[seed_idx]
    seed_raw_cat = db_categories[seed_idx]
    mapped_seed_cat = map_to_blueprint(seed_raw_cat)

    st.markdown("### Blueprint")
    st.caption(f"Seed category detected as **{mapped_seed_cat}**.")

    blueprint = DRESS_BLUEPRINT if mapped_seed_cat == "dresses" else [
        c for c in DEFAULT_BLUEPRINT if c != mapped_seed_cat
    ]

    queries = st.multiselect(
        "Components to generate",
        options=AVAILABLE_CATEGORIES,
        default=blueprint,
    )

    st.markdown("### Settings")
    top_k = st.slider("Alternatives per slot", 2, 6, 3)
    use_text_hint = st.toggle("Use category text hint", value=True,
                              help="Off = pure visual reasoning (closer to FITB eval).")

    st.divider()
    st.caption(f"Checkpoint: `{A['ckpt']}`")
    st.caption(f"Device: `{device}` · DB: `{len(db_ids):,}` items")

if not queries:
    st.warning("Select at least one component in the sidebar.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Inference (autoregressive)
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Composing outfit…"):
    context_embeddings = [embeddings[seed_item_id]["image"].unsqueeze(0)]
    context_text_embs = [embeddings[seed_item_id]["text"].unsqueeze(0)]
    context_images = [seed_img]
    used_ids = {seed_item_id}
    history = []

    for query in queries:
        context_feats = torch.cat(context_embeddings, dim=0).unsqueeze(0).to(device)
        context_txts = torch.cat(context_text_embs, dim=0).unsqueeze(0).to(device)
        context_mask = torch.zeros((1, context_feats.shape[1]), dtype=torch.bool, device=device)

        text_features = None
        if use_text_hint:
            target_token = model.tokenizer([query]).to(device)
            with torch.no_grad():
                text_features = model.siglip.encode_text(target_token, normalize=True)

        with torch.no_grad():
            ideal_emb = model.encode_features(
                context_feats, context_mask, text_features,
                context_text_features=context_txts,
            )

        results = search_top_k(
            ideal_emb, db_features, db_categories, db_images, db_ids,
            require_keyword=query, k=top_k, exclude_ids=used_ids,
        )

        best_img, best_id = results[0]
        used_ids.add(best_id)
        context_images.append(best_img)
        context_embeddings.append(embeddings[best_id]["image"].unsqueeze(0))
        context_text_embs.append(embeddings[best_id]["text"].unsqueeze(0))
        history.append({"query": query, "results": results})

# ──────────────────────────────────────────────────────────────────────────────
# Output — Final outfit
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<h2 class="section">The Outfit <span class="muted">final composition</span></h2>',
            unsafe_allow_html=True)

st.markdown(
    f'<div class="stat-strip">'
    f'<div class="stat">Seed · <b>{mapped_seed_cat}</b></div>'
    f'<div class="stat">Components · <b>{len(queries)}</b></div>'
    f'<div class="stat">Mode · <b>{"text-guided" if use_text_hint else "pure visual"}</b></div>'
    f'</div>',
    unsafe_allow_html=True,
)

cols = st.columns(len(context_images))
labels = ["Seed"] + [q.title() for q in queries]
for i, col in enumerate(cols):
    with col:
        klass = "item-card selected" if i == 0 else "item-card"
        st.markdown(f'<div class="{klass}">', unsafe_allow_html=True)
        st.image(context_images[i], use_container_width=True)
        klass_label = "item-label primary" if i == 0 else "item-label"
        st.markdown(f'<div class="{klass_label}">{labels[i]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Output — Alternatives
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<h2 class="section">Alternatives <span class="muted">per slot</span></h2>',
            unsafe_allow_html=True)

tabs = st.tabs([q.title() for q in queries])
for tab, step in zip(tabs, history):
    with tab:
        alt_cols = st.columns(len(step["results"]))
        for i, (img, _id) in enumerate(step["results"]):
            with alt_cols[i]:
                klass = "item-card selected" if i == 0 else "item-card"
                st.markdown(f'<div class="{klass}">', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                rank_label = "Top pick" if i == 0 else f"Rank {i + 1}"
                klass_label = "item-label primary" if i == 0 else "item-label"
                st.markdown(f'<div class="{klass_label}">{rank_label}</div>',
                            unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
