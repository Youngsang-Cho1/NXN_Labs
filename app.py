import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import streamlit as st
import torch
from datasets import load_dataset
from PIL import Image
import random

from model import OutfitTransformer

# --- 1. UI Configuration ---
st.set_page_config(page_title="AI Fashion Stylist", layout="wide", initial_sidebar_state="expanded")
st.title("👗 AI Fashion Stylist: Autoregressive Outfit Generation")
st.markdown("Build your perfect outfit autoregressively based on a starting seed item using the OutfitTransformer.")

EMBED_PATH = "polyvore_embeddings.pt"
DB_SIZE = 5000

# --- 2. Model & Data Loading (Cached) ---
@st.cache_resource
def load_assets():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Model
    model = OutfitTransformer(num_layers=4).to(device)
    try:
        model.load_state_dict(torch.load("outfit_transformer_epoch_3.pt", map_location=device, weights_only=True), strict=False)
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print("⚠️ Could not find weights. Initializing with random weights.")
        
    # [HOTFIX] Zero out the newly added, UNTRAINED modality embeddings
    # so the demo UI performs at its true 64% capacity instead of spewing random noise!
    with torch.no_grad():
        if hasattr(model, 'text_type_emb'): model.text_type_emb.zero_()
        if hasattr(model, 'image_type_emb'): model.image_type_emb.zero_()
        if hasattr(model, 'mask_token'): model.mask_token.zero_()
        
    model.eval()

    # 2. Load Embeddings & Build DB
    embeddings_dict = torch.load(EMBED_PATH, map_location="cpu", weights_only=False)
    all_item_ids = list(embeddings_dict.keys())[:DB_SIZE]
    
    polyvore_items = load_dataset("owj0421/polyvore", split="data")
    db_items = polyvore_items.select(range(DB_SIZE))
    id_to_idx = {item["item_id"]: i for i, item in enumerate(db_items)}

    db_features = []
    db_categories = []
    db_images = []
    db_item_ids_ordered = []

    for item_id in all_item_ids:
        if item_id not in id_to_idx:
            continue
        idx = id_to_idx[item_id]
        item = db_items[idx]

        img = item["image"]
        if getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")

        cat = str(item.get("category") or item.get("title") or item.get("description") or "").lower()

        db_features.append(embeddings_dict[item_id]["image"])
        db_categories.append(cat)
        db_images.append(img)
        db_item_ids_ordered.append(item_id)

    db_features_tensor = torch.stack(db_features).to(device)
    
    return model, embeddings_dict, db_features_tensor, db_categories, db_images, db_item_ids_ordered, device

# --- 3. Retrieval Logic ---
def search_top_k_items(target_emb, db_features_tensor, db_categories, db_images, db_item_ids_ordered, require_keyword=None, k=3):
    distances = torch.norm(db_features_tensor - target_emb, p=2, dim=-1)
    sorted_indices = torch.argsort(distances)
    
    results = []
    for idx in sorted_indices:
        idx = idx.item()
        if require_keyword:
            if require_keyword.lower() in db_categories[idx]:
                results.append((db_images[idx], db_item_ids_ordered[idx]))
        else:
            results.append((db_images[idx], db_item_ids_ordered[idx]))
            
        if len(results) >= k:
            break
            
    # Fallback if we couldn't find exactly K items matching the keyword
    if len(results) == 0:
        for idx in sorted_indices[:k]:
            results.append((db_images[idx.item()], db_item_ids_ordered[idx.item()]))
            
    return results

# --- Main App Execution ---
with st.spinner("Loading model and Vector DB... (This only happens once)"):
    model, embeddings_dict, db_features_tensor, db_categories, db_images, db_item_ids_ordered, device = load_assets()

# --- Sidebar: 1. Seed Selection ---
st.sidebar.header("🌱 1. Seed Selection")

if "seed_idx" not in st.session_state:
    st.session_state.seed_idx = random.randint(0, len(db_item_ids_ordered) - 1)

# Random selection
if st.sidebar.button("🎲 Pick Random Seed Item", use_container_width=True):
    st.session_state.seed_idx = random.randint(0, len(db_item_ids_ordered) - 1)

# Manual index input
manual_idx = st.sidebar.number_input(
    f"Or enter Seed Index (0 to {len(db_item_ids_ordered)-1}):", 
    min_value=0, 
    max_value=len(db_item_ids_ordered)-1, 
    value=st.session_state.seed_idx
)

# Update seed if manual input changes
if manual_idx != st.session_state.seed_idx:
    st.session_state.seed_idx = manual_idx

# Grab Seed Info
seed_idx = st.session_state.seed_idx
seed_item_id = db_item_ids_ordered[seed_idx]
seed_img = db_images[seed_idx]
seed_raw_cat = db_categories[seed_idx]

# Map raw polyvore category to structural blueprint category
def map_to_blueprint(raw_cat):
    raw_cat = raw_cat.lower()
    if any(k in raw_cat for k in ["shirt", "top", "blouse", "t-shirt", "sweater", "hoodie", "camisole", "knit"]): return "tops"
    if any(k in raw_cat for k in ["pant", "jean", "skirt", "bottom", "short", "legging", "trouser"]): return "bottoms"
    if any(k in raw_cat for k in ["shoe", "sneaker", "boot", "heel", "sandal", "flat", "wedge"]): return "shoes"
    if any(k in raw_cat for k in ["bag", "purse", "tote", "backpack", "clutch", "satchel"]): return "bags"
    if any(k in raw_cat for k in ["jacket", "coat", "outer", "blazer", "cardigan", "vest"]): return "outerwear"
    if any(k in raw_cat for k in ["dress", "gown", "romper"]): return "dresses"
    if any(k in raw_cat for k in ["jewelry", "necklace", "ring", "earring", "bracelet", "watch"]): return "jewelry"
    if any(k in raw_cat for k in ["hat", "cap", "beanie"]): return "hats"
    return "accessories"

mapped_seed_cat = map_to_blueprint(seed_raw_cat)

# --- Sidebar: 2. Outfit Blueprint ---
st.sidebar.divider()
st.sidebar.header("🎨 2. Outfit Blueprint")
st.sidebar.markdown(f"Detected Seed Category: **{mapped_seed_cat.upper()}**")

available_categories = ["tops", "bottoms", "dresses", "outerwear", "shoes", "bags", "accessories", "jewelry", "hats"]
standard_blueprint = ["tops", "bottoms", "shoes", "bags"]

# Smart Dynamic Logic: If seed is a dress, swap blueprint. Otherwise remove the seed's category from generation.
if mapped_seed_cat == "dresses":
    standard_blueprint = ["shoes", "bags", "jewelry"]
elif mapped_seed_cat in standard_blueprint:
    standard_blueprint.remove(mapped_seed_cat)

queries = st.sidebar.multiselect(
    "Smart Auto-Components (Editable):",
    options=available_categories,
    default=standard_blueprint
)

if not queries:
    st.sidebar.warning("Please select at least one component!")
    st.stop()

# 1. Run inference silently first to collect results
with st.spinner("🪄 AI is styling your outfit..."):
    context_embeddings = [embeddings_dict[seed_item_id]["image"].unsqueeze(0)]
    context_images = [seed_img]
    
    generation_history = []
    
    for query in queries:
        context_img_features = torch.cat(context_embeddings, dim=0).unsqueeze(0).to(device)
        context_mask = torch.zeros((1, context_img_features.shape[1]), dtype=torch.bool).to(device)
        
        target_token = model.tokenizer([query]).to(device)
        with torch.no_grad():
            text_features = model.siglip.encode_text(target_token, normalize=True)
            ideal_emb = model.encode_features(context_img_features, context_mask, text_features)
            
        top_k_results = search_top_k_items(ideal_emb, db_features_tensor, db_categories, db_images, db_item_ids_ordered, require_keyword=query, k=3)
        
        # Autoregressive Update: Next item uses Rank 1 as context
        best_img, best_id = top_k_results[0]
        context_images.append(best_img)
        context_embeddings.append(embeddings_dict[best_id]["image"].unsqueeze(0))
        
        generation_history.append({
            "query": query,
            "top_k": top_k_results
        })

# 2. Render Final Assembled Outfit at the top
st.markdown("## 🌟 Final Assembled Outfit")
st.markdown("Here is the complete look tailored by the OutfitTransformer.")

# Display all selected items (Seed + Rank 1s) in a single row
outfit_cols = st.columns(len(context_images))
labels = ["🌱 Seed Item"] + [f"✨ Added: {q.title()}" for q in queries]

for i, col in enumerate(outfit_cols):
    with col:
        # Subtle border styling using markdown
        st.markdown(f"<div style='text-align: center; font-weight: bold; color: #555;'>{labels[i]}</div>", unsafe_allow_html=True)
        st.image(context_images[i], use_container_width=True)

st.divider()

# 3. Render Alternatives in Expanders below
st.markdown("## 🔎 Styling Alternatives (Top 3 Picks)")
st.markdown("Not feeling the top pick? Here are the other options the AI considered.")

for step in generation_history:
    query = step["query"]
    top_k = step["top_k"]
    
    with st.expander(f"Alternatives for **{query.upper()}**", expanded=True):
        alt_cols = st.columns(3)
        for i, (retrieved_img, retrieved_id) in enumerate(top_k):
            with alt_cols[i]:
                # Rank 1 is the one chosen above
                badge = "🏆 Top Pick (Selected)" if i == 0 else f"Rank {i+1}"
                st.image(retrieved_img, use_container_width=True, caption=badge)

st.success("🎉 Outfit generation complete! Try testing with a different seed item or categories.")
