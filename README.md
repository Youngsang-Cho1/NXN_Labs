# OutfitTransformer: Complementary Fashion Item Retrieval

An AI-powered fashion recommendation system that predicts the missing item in a **partial outfit**.  
Rather than simple image similarity search, it understands the **contextual harmony** between multiple items to recommend the best-fitting complement.

---

## Core Idea

SigLIP alone can only understand individual garments **one at a time**.  
Our **OutfitTransformer** receives multiple items simultaneously, models inter-item relationships via Self-Attention, and predicts the embedding of the item needed to complete the outfit.

```
[SigLIP]             → Extract features from individual items (Translator, Frozen)
[OutfitTransformer]  → Reason over the full outfit context to predict missing item (Learner)
```

---

## Model Architecture (`model.py`)

```
Item Images / Texts
      |
[SigLIP Vision/Text Encoder]   <- Always Frozen
      | 768D vectors
      v
[TransformerEncoder]            <- Trained (Self-Attention, models outfit relationships)
      |
[MLP Projection Head]           <- Trained (Linear->GELU->Linear, aligns to SigLIP space)
      |
[L2 Normalize]
      |
  Predicted Vector (768D)   <->   Ground-truth item's SigLIP image vector
```

**Three trainable components:**

| Component | Role |
|---|---|
| `TransformerEncoder` | Learns relationships between outfit items via Self-Attention |
| `MLP Projection Head` | Maps Transformer output into SigLIP's image embedding space |
| `logit_scale` | Automatically tunes the temperature of InfoNCE Loss |

---

## Dataset & Metadata (`dataset.py`)

**Datasets (HuggingFace):**
- `owj0421/polyvore`: **251,008** individual fashion items (image + metadata)
- `owj0421/polyvore-outfits`: **53,306** outfit sets (item ID combinations)

**Item metadata fields:**

| Field | Content | Usage |
|---|---|---|
| `title` | "Black Slim Fit Leather Jacket" | SigLIP text encoding (primary) |
| `category` | "outerwear" | Hard Negative Mining (same-category negatives) |
| `description` | Detailed description | Fallback if title is empty |
| `url_name` | URL-based name | Last resort fallback |
| `image` | PIL Image | SigLIP image encoding |

**Training sample construction (Hard Negative Mining):**
```
Outfit: [Jacket, Pants, Shoes]
            | randomly hide one
Context:   [Jacket, Pants]       <- Hint
Target:    [Shoes]               <- Ground truth
Hard Neg:  [Other shoes]         <- Same category, genuinely confusing
Easy Neg:  [Random items]        <- Random negatives
```

---

## Speed Optimization: Vector Pre-Caching (`vectorize_data.py`)

| Approach | Time per Epoch |
|---|---|
| Naive (re-run SigLIP every batch) | ~2 hours+ |
| Optimized (pre-cached vectors) | **~7 minutes** |

All 251,008 items are passed through SigLIP **exactly once** and saved as `polyvore_embeddings.pt`.  
Subsequent training and evaluation load only vectors — no image processing required.

---

## Performance Improvement Strategies

| Strategy | Description | Category |
|---|---|---|
| **Learnable Temperature** | `logit_scale` trained as a parameter instead of fixed at 40.0 | Optimization |
| **MLP Projection Head** | Filter layer added after Transformer output | Architecture |
| **Text Dropout** | Text metadata randomly dropped 20% of the time during training | Regularization |
| **Grid Search** | Automated hyperparameter sweep across LR, layers, dropout, epochs | Tuning |

---

## File Structure (Execution Order)

```
1. vectorize_data.py  -> Pre-compute item vectors (run once, ~2 hours)
2. dataset.py         -> Generates training samples (called internally by train.py)
3. model.py           -> OutfitTransformer architecture definition
4. train.py           -> Model training (default: 3 epochs)
5. eval_fitb.py       -> FITB benchmark evaluation (500 4-choice questions)
6. grid_search.py     -> Automated hyperparameter optimization
7. demo.py            -> Autoregressive outfit completion demo
```

---

## Quick Start

```bash
# 1. Pre-compute vectors (run only once, ~2 hours)
python vectorize_data.py

# 2. Train the model (~7 min/epoch)
python train.py

# 3. Evaluate performance (seconds)
python eval_fitb.py

# 4. (Optional) Hyperparameter grid search
python grid_search.py

# 5. (Optional) Run outfit completion demo
python demo.py
```

**CLI arguments for train.py:**
```bash
python train.py --lr=1e-3 --epochs=5 --text_dropout=0.1 --num_layers=6
```

---

## Evaluation Metric

**FITB (Fill-In-The-Blank)**
- One item is removed from an outfit; the model picks the correct one from 4 candidates
- Random Baseline: **25%**
- Current Model: **~35%**
- Published SotA (without complex metric learning): ~50%+
