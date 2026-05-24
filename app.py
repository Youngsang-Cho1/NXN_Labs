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
    page_title="ATELIER — AI Stylist",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
      /* ── Light theme override ── */
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
      [data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] {
        background: #ffffff !important;
        color: #111111 !important;
      }
      .stApp * { color: #1a1a1a; }

      /* ── Layout ── */
      .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }
      #MainMenu, footer { visibility: hidden; }

      /* ── Typography: Vogue-ish ── */
      html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label,
      [data-testid="stWidgetLabel"] {
        font-family: "Inter", -apple-system, sans-serif;
      }
      h1.brand {
        font-family: "Playfair Display", "Didot", "Times New Roman", serif;
        font-weight: 800; font-size: 3.4rem; letter-spacing: 0.18em;
        margin: 0.5rem 0 0.2rem 0; text-transform: uppercase;
        color: #0a0a0a; text-align: center;
      }
      .brand-rule {
        height: 1px; background: #111; width: 60px; margin: 0.4rem auto 0.6rem auto;
      }
      p.tagline {
        text-align: center; color: #6b6b6b; font-style: italic;
        font-size: 0.95rem; letter-spacing: 0.04em; margin-bottom: 2.4rem;
      }

      /* ── Section headers ── */
      h2.section {
        font-family: "Playfair Display", serif;
        font-weight: 600; font-size: 1.6rem; letter-spacing: 0.04em;
        margin: 2.4rem 0 0.4rem 0; color: #0a0a0a;
        border-bottom: 1px solid #1a1a1a; padding-bottom: 0.5rem;
      }
      h2.section .muted {
        font-family: "Inter", sans-serif; font-style: italic;
        color: #888; font-weight: 400; font-size: 0.85rem;
        letter-spacing: 0.05em; margin-left: 0.6rem; text-transform: lowercase;
      }

      /* ── Outfit card row header ── */
      .outfit-header {
        display:flex; align-items:baseline; gap:18px;
        margin: 1.6rem 0 0.6rem 0; padding-bottom: 0.4rem;
        border-bottom: 1px dashed #d4d4d4;
      }
      .outfit-num {
        font-family: "Playfair Display", serif;
        font-weight: 700; font-size: 1.4rem; letter-spacing: 0.06em;
        color: #0a0a0a;
      }
      .outfit-badge {
        font-family: "Inter", sans-serif; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.15em;
        padding: 4px 12px; border: 1px solid #1a1a1a; border-radius: 0;
        color: #1a1a1a; background: #fff;
      }
      .outfit-badge.primary { background: #0a0a0a; color: #fff; border-color: #0a0a0a; }
      .outfit-score {
        font-family: "Inter", sans-serif; font-size: 0.8rem; color: #6b6b6b;
        letter-spacing: 0.03em; margin-left: auto;
      }
      .outfit-score b { color: #0a0a0a; font-weight: 600; }

      /* ── Item card ── */
      .item-card {
        background: #fafafa; border: 1px solid transparent;
        padding: 14px; border-radius: 2px;
        transition: all .2s ease;
      }
      .item-card:hover { border-color: #1a1a1a; background: #fff; }
      .item-card.selected { background: #fff; border-color: #1a1a1a; }
      .item-card img { background: #fff; }
      .item-label {
        font-family: "Inter", sans-serif; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: #999; text-align: center; margin-top: 12px;
      }
      .item-label.primary { color: #0a0a0a; font-weight: 500; }
      .item-label .score {
        display:inline-block; margin-left: 6px; color: #6b6b6b; letter-spacing: 0;
      }

      /* ── Stat strip ── */
      .stat-strip {
        display: flex; gap: 28px; justify-content: center;
        padding: 1rem 0; border-top: 1px solid #e5e5e5; border-bottom: 1px solid #e5e5e5;
        margin: 1rem 0 2rem 0;
      }
      .stat {
        font-family: "Inter", sans-serif; font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 0.15em; color: #888;
      }
      .stat b {
        display: block; font-family: "Playfair Display", serif;
        color: #0a0a0a; font-weight: 600; font-size: 1.1rem;
        letter-spacing: 0.02em; text-transform: none; margin-top: 2px;
      }

      /* ── Sidebar polish ── */
      [data-testid="stSidebar"] { border-right: 1px solid #e5e5e5; }
      [data-testid="stSidebar"] h3 {
        font-family: "Playfair Display", serif !important;
        font-weight: 600; font-size: 1rem; letter-spacing: 0.1em;
        text-transform: uppercase; margin-top: 1rem;
        color: #0a0a0a !important; border-bottom: 1px solid #1a1a1a;
        padding-bottom: 0.3rem;
      }
      [data-testid="stSidebar"] .stCaption,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        font-family: "Inter", sans-serif !important;
        font-size: 0.72rem !important; color: #888 !important;
        letter-spacing: 0.05em;
      }

      /* Inputs look more editorial */
      .stButton > button {
        border-radius: 0; border: 1px solid #1a1a1a; background: #fff;
        color: #1a1a1a; font-family: "Inter"; letter-spacing: 0.08em;
        text-transform: uppercase; font-size: 0.75rem; font-weight: 500;
      }
      .stButton > button:hover { background: #0a0a0a; color: #fff; }

      hr { border-color: #e5e5e5 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="brand">Atelier</h1>', unsafe_allow_html=True)
st.markdown('<div class="brand-rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="tagline">An AI atelier — composed outfits, retrieved from a curated archive.</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
EMBED_PATH = "polyvore_embeddings.pt"
DB_SIZE = 5000
DEFAULT_BLUEPRINT = ["tops", "bottoms", "shoes", "bags"]
DRESS_BLUEPRINT = ["outerwear", "shoes", "bags", "jewelry"]
AVAILABLE_CATEGORIES = ["tops", "bottoms", "dresses", "outerwear", "shoes", "bags", "accessories", "jewelry", "hats"]

# Slots that don't make sense to layer onto certain seeds.
# Dresses already cover top+bottom; suggesting tops/bottoms produces clashing outfits.
INCOMPATIBLE_SLOTS = {
    "dresses": {"tops", "bottoms", "dresses"},
    "tops":    {"tops", "dresses"},
    "bottoms": {"bottoms", "dresses"},
    "shoes":   {"shoes"},
    "bags":    {"bags"},
    "outerwear": {"outerwear"},
    "hats":    {"hats"},
}


def best_checkpoint() -> str | None:
    """Prefer the known-best v3 checkpoint; fall back to latest available."""
    preferred = ["v3_epoch_10.pt", "v3_epoch_11.pt", "v3_epoch_9.pt", "best_v3.pt"]
    for p in preferred:
        if os.path.exists(p):
            return p
    candidates = (
        glob.glob("v3_epoch_*.pt")
        or glob.glob("v2full_epoch_*.pt")
        or glob.glob("outfit_transformer_epoch_*.pt")
    )
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Asset loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = OutfitTransformer(num_layers=4).to(device)
    ckpt = best_checkpoint()
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

    if st.button("Shuffle seed", width="stretch"):
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
    st.caption(f"Seed detected as **{mapped_seed_cat}** · incompatible slots auto-hidden.")

    blocked = INCOMPATIBLE_SLOTS.get(mapped_seed_cat, {mapped_seed_cat})
    allowed_categories = [c for c in AVAILABLE_CATEGORIES if c not in blocked]

    if mapped_seed_cat == "dresses":
        blueprint = [c for c in DRESS_BLUEPRINT if c in allowed_categories]
    else:
        blueprint = [c for c in DEFAULT_BLUEPRINT if c in allowed_categories]

    queries = st.multiselect(
        "Components to generate",
        options=allowed_categories,
        default=blueprint,
    )

    st.markdown("### Settings")
    num_outfits = st.slider("Outfits to generate", 1, 3, 3,
                            help="Each outfit explores a different first-slot pick.")
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
# Inference — generate N diverse outfits + per-slot retrieval scores
# ──────────────────────────────────────────────────────────────────────────────
def predict_next(context_img_embs, context_txt_embs, query: str):
    """Run one retrieval step. Returns (ideal_emb [768], top_k results)."""
    ctx_img = torch.cat(context_img_embs, dim=0).unsqueeze(0).to(device)
    ctx_txt = torch.cat(context_txt_embs, dim=0).unsqueeze(0).to(device)
    ctx_mask = torch.zeros((1, ctx_img.shape[1]), dtype=torch.bool, device=device)

    text_features = None
    if use_text_hint:
        token = model.tokenizer([query]).to(device)
        with torch.no_grad():
            text_features = model.siglip.encode_text(token, normalize=True)

    with torch.no_grad():
        ideal = model.encode_features(
            ctx_img, ctx_mask, text_features, context_text_features=ctx_txt,
        )
    return ideal.squeeze(0)


def cosine_score(ideal_emb, item_id: str) -> float:
    """Raw cosine sim in roughly [0.05, 0.35] — normalize to a friendly 0-100 band."""
    item_vec = embeddings[item_id]["image"].to(device)
    raw = torch.nn.functional.cosine_similarity(
        ideal_emb.unsqueeze(0), item_vec.unsqueeze(0)
    ).item()
    # Linearly map [0.05, 0.35] -> [40, 100], clamp
    pct = 40.0 + (raw - 0.05) * (60.0 / 0.30)
    return max(40.0, min(100.0, pct))


with st.spinner("Composing outfits…"):
    # Step 1: get first-slot candidates (one per outfit, all start from same seed)
    first_query = queries[0]
    first_ideal = predict_next(
        [embeddings[seed_item_id]["image"].unsqueeze(0)],
        [embeddings[seed_item_id]["text"].unsqueeze(0)],
        first_query,
    )
    first_candidates = search_top_k(
        first_ideal, db_features, db_categories, db_images, db_ids,
        require_keyword=first_query, k=max(num_outfits, top_k),
        exclude_ids={seed_item_id},
    )

    outfits = []  # each: {"items": [(img, id, slot, score)], "total": float}
    for outfit_idx in range(min(num_outfits, len(first_candidates))):
        first_img, first_id = first_candidates[outfit_idx]
        first_score = cosine_score(first_ideal, first_id)

        ctx_imgs = [embeddings[seed_item_id]["image"].unsqueeze(0),
                    embeddings[first_id]["image"].unsqueeze(0)]
        ctx_txts = [embeddings[seed_item_id]["text"].unsqueeze(0),
                    embeddings[first_id]["text"].unsqueeze(0)]
        used = {seed_item_id, first_id}

        items = [
            (seed_img, seed_item_id, mapped_seed_cat, None),
            (first_img, first_id, first_query, first_score),
        ]
        per_slot_history = [{"query": first_query, "results": first_candidates,
                             "chosen_idx": outfit_idx, "ideal": first_ideal}]

        # Remaining slots: greedy rank-1
        for query in queries[1:]:
            ideal = predict_next(ctx_imgs, ctx_txts, query)
            results = search_top_k(
                ideal, db_features, db_categories, db_images, db_ids,
                require_keyword=query, k=top_k, exclude_ids=used,
            )
            best_img, best_id = results[0]
            score = cosine_score(ideal, best_id)
            used.add(best_id)
            ctx_imgs.append(embeddings[best_id]["image"].unsqueeze(0))
            ctx_txts.append(embeddings[best_id]["text"].unsqueeze(0))
            items.append((best_img, best_id, query, score))
            per_slot_history.append({"query": query, "results": results,
                                     "chosen_idx": 0, "ideal": ideal})

        slot_scores = [s for _, _, _, s in items if s is not None]
        total = sum(slot_scores) / len(slot_scores) if slot_scores else 0.0  # already 0-100
        outfits.append({"items": items, "total": total, "history": per_slot_history})

    outfits.sort(key=lambda o: o["total"], reverse=True)

# ──────────────────────────────────────────────────────────────────────────────
# Output — Ranked outfit cards
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="stat-strip">'
    f'<div class="stat">Seed<b>{mapped_seed_cat}</b></div>'
    f'<div class="stat">Slots<b>{len(queries)}</b></div>'
    f'<div class="stat">Looks<b>{len(outfits)}</b></div>'
    f'<div class="stat">Mode<b>{"text-guided" if use_text_hint else "pure visual"}</b></div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h2 class="section">The Edit <span class="muted">curated looks</span></h2>',
    unsafe_allow_html=True,
)

for rank, outfit in enumerate(outfits, start=1):
    score_pct = outfit["total"]
    badge_text = "Featured" if rank == 1 else f"Look {rank:02d}"
    badge_klass = "outfit-badge primary" if rank == 1 else "outfit-badge"
    st.markdown(
        f'<div class="outfit-header">'
        f'<span class="outfit-num">Look {rank:02d}</span>'
        f'<span class="{badge_klass}">{badge_text}</span>'
        f'<span class="outfit-score">Harmony · <b>{score_pct:.1f}</b> / 100</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(outfit["items"]))
    for i, ((img, _id, slot, score), col) in enumerate(zip(outfit["items"], cols)):
        with col:
            klass = "item-card selected" if i == 0 else "item-card"
            st.markdown(f'<div class="{klass}">', unsafe_allow_html=True)
            st.image(img, width="stretch")
            klass_label = "item-label primary" if i == 0 else "item-label"
            if i == 0:
                sub = "Seed"
            else:
                sub = f'{slot}<span class="score">· {score:.0f}</span>'
            st.markdown(f'<div class="{klass_label}">{sub}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Output — Alternatives (from top pick)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 class="section">The Atelier&#39;s Notes <span class="muted">alternates per slot</span></h2>',
    unsafe_allow_html=True,
)

top_history = outfits[0]["history"]
tabs = st.tabs([step["query"].title() for step in top_history])
for tab, step in zip(tabs, top_history):
    with tab:
        alt_cols = st.columns(len(step["results"]))
        for i, (img, _id) in enumerate(step["results"]):
            with alt_cols[i]:
                klass = "item-card selected" if i == step["chosen_idx"] else "item-card"
                st.markdown(f'<div class="{klass}">', unsafe_allow_html=True)
                st.image(img, width="stretch")
                rank_label = "Top pick" if i == 0 else f"Rank {i + 1}"
                klass_label = "item-label primary" if i == 0 else "item-label"
                st.markdown(f'<div class="{klass_label}">{rank_label}</div>',
                            unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
