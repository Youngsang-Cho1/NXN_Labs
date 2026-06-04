# OutfitTransformer: Complementary Fashion Item Retrieval

An AI fashion stylist that completes a partial outfit. Given any seed garment and target slots (e.g. *bottoms, shoes, bag*), the model retrieves complementary items that match the global style of the rest of the outfit — not just visually similar items.

Built on a frozen fashion-specific SigLIP backbone with a trained transformer that reasons over multimodal (image + text) representations of every item in the outfit.

---

## Results

Evaluated on the Polyvore-Outfits Fill-in-the-Blank benchmark (4-choice multiple choice, 10,000 questions per split). Each model is trained on the train set of its own split — no cross-split item leakage.

We ship **two variants** of the model:

- **`text` variant** — image + per-item text fusion. The production model: enables natural-language query, occasion / brand filters, and any downstream feature that needs to read item descriptions.
- **`no_text` variant** — image only. The benchmark-optimal model: stronger on FITB itself, but cannot consume text at inference.

| Split                | Random | OutfitTransformer (paper, CVPR 2022) | **Ours (`text`)** | **Ours (`no_text`)** |
| -------------------- | :----: | :----------------------------------: | :---------------: | :------------------: |
| Nondisjoint (10k)    | 25.00  | 67.10                                | **66.85**         | —                    |
| **Disjoint (10k)**   | 25.00  | 59.48                                | **63.78** (+4.30) | **66.65** (+7.17)    |

**Disjoint** means train and test item pools do not overlap — a stronger test of generalization to unseen products. Both variants beat the CVPR 2022 SOTA on disjoint; the `no_text` variant is stronger on FITB itself because FITB is a text-free benchmark (see Ablations below).

State-of-the-art benchmark from: *Sarkar et al., "OutfitTransformer: Learning Outfit Representations for Fashion Recommendation", CVPR 2022* ([arXiv:2204.04812](https://arxiv.org/abs/2204.04812)).

### How much of this is the backbone vs the learned transformer?

To isolate the backbone's contribution, we evaluate frozen Marqo Fashion-SigLIP with no learning — just simple aggregation of context-item embeddings and cosine-rank the candidates. Same disjoint FITB, 10,000 questions.

| Method                              | Disjoint FITB | Lift vs random |
| ----------------------------------- | :-----------: | :------------: |
| Random                              |     25.00     |        —       |
| SigLIP-max  (cand vs closest ctx)   |     51.77     |     +26.77     |
| SigLIP-mean (cand vs centroid ctx)  |     54.22     |     +29.22     |
| Paper SOTA (ResNet, CVPR 2022)      |     59.48     |     +34.48     |
| Ours `text`  (learned transformer)  |     63.78     |     +38.78     |
| **Ours `no_text`** (learned transformer) | **66.65** | **+41.65**     |

The frozen Fashion-SigLIP backbone alone already clears **54%** with no training — that's most of our score. The learned transformer adds another **+9.56** (`text`) or **+12.43** (`no_text`) on top, which is the part that's actually doing complementary (vs. similar) reasoning across the outfit. mean ≈ sum because L2-normalized features have a constant scale; max being lower than mean confirms that *outfit-level* aggregation matters more than picking the candidate closest to any single context item.

Reproduce: `python eval_siglip_baseline.py --split disjoint --num_samples 10000`

### Ablations (disjoint split, 10k FITB)

Each row keeps everything from the `text` baseline fixed except the one design choice listed.

| Variant                                | FITB  | Δ      | What it shows                                                                 |
| -------------------------------------- | :---: | :----: | ----------------------------------------------------------------------------- |
| **Baseline** (`text`)                  | 63.78 |   —    | image + text fusion, mid-sim hard negatives, sample-wise text dropout 0.3     |
| `no_context_text`                      | **66.65** | **+2.87** | Removing per-item text fusion **improves** FITB (see frame below)        |
| `topk_neg` (naive most-similar negs)   | 62.02 | −1.76  | Mid-similarity hard-negative mining gives a small lift                        |
| `no_text_dropout` (broken — see note)  | 30.33 | −33.45 | Without text dropout, `mask_token` is never trained; FITB is text-free, so it collapses. Confirms text dropout is load-bearing for text-free eval. |

**Why does removing text help FITB?** Two reasons compound:

1. **Polyvore titles are noisy.** Lots of placeholders, brand junk, and broken descriptions get baked into the SigLIP text embedding and pollute the context.
2. **FITB itself is text-free** at the target side, so a model that learned to lean on text in training is slightly worse-calibrated at test time than one that learned image-only from the start.

**This does not mean text is useless.** It means *Polyvore FITB doesn't measure text utility.* In production — natural-language queries, occasion / brand filters, "navy wool overcoat" lookups — text is essential. We ship both checkpoints; FITB is reported on each so the trade-off is explicit.

> **Note on evaluation integrity.** An earlier version of this README reported **68.60%** (+9.12) on the disjoint split. That model was trained on the `nondisjoint_default` config and evaluated on the `disjoint` FITB test, and 79% of the disjoint test items had appeared in the nondisjoint training set — the "disjoint" guarantee only holds *within* the disjoint config, not across configs. The numbers above are from models retrained on `disjoint_default`, so no test item was seen during training. See `check_disjoint_leak.py`.

---

## Pipeline

```
                  ┌─────────────────────────────────────────┐
                  │  Pre-computation (once, cached on disk) │
                  └─────────────────────────────────────────┘
   251,008 items ──► SigLIP (frozen) ──► polyvore_embeddings.pt
                                          (image + text vectors, 768-d)

                  ┌─────────────────────────────────────────┐
                  │  Training / Inference                   │
                  └─────────────────────────────────────────┘

   target slot text ("shoes")         context items (image + text)
            │                                  │
            ▼                                  ▼
   [+ text_type_emb]              [context_fuse MLP: image ⊕ text]
   or mask_token (dropout)                     │
            │                                  + image_type_emb
            └──────┬───────────────────────────┘
                   ▼
        [query, ctx_1, ctx_2, ..., ctx_n]
                   │
                   ▼
        TransformerEncoder  (4 layers, 8 heads, d=768, self-attention)
                   │
                   ▼
        encoded[0]  (query position)
                   │
                   ▼
        Projection head (MLP)  →  L2 normalize
                   │
                   ▼
        predicted target vector  (768-d)
                   │
                   ▼
        Cosine search against the catalogue  →  top-K complementary items
```

### Why this design

* **Frozen Marqo Fashion SigLIP backbone.** Fashion-specific vision-language model; produces strong out-of-the-box embeddings, eliminates the cost of fine-tuning a large encoder, and (as shown by the disjoint result above) generalizes better to unseen items than end-to-end ResNets.
* **Per-item image + text fusion (`context_fuse`).** Each context item carries its description ("black leather jacket"), not just pixels — colour, fabric, and style cues enter the transformer as language.
* **Query-token at position 0.** The target slot text serves as a CLS-like query token; self-attention lets it pool whatever the rest of the outfit demands.
* **Sample-wise text dropout (30%).** Each training sample independently drops the target text, forcing the `mask_token` to learn — critical because FITB evaluation is text-free.
* **Mid-similarity hard negatives (10–40 percentile).** Same-category items in the middle similarity band, instead of the most-similar items. Avoids false negatives (the most-similar item is often also compatible).
* **Cosine LR scheduler.** Stabilizes late epochs; the best checkpoint lands near the end of training instead of oscillating mid-way.

---

## Trained Components

| Component                       | Params | Role                                                                       |
| ------------------------------- | :----: | -------------------------------------------------------------------------- |
| `context_fuse` (MLP)            | ~1.2M  | Fuses image and text embeddings per context item                           |
| `TransformerEncoder` (4 layers) | ~12M   | Self-attention across context + query                                      |
| `projection_head` (MLP)         | ~1M    | Maps transformer output back into the SigLIP retrieval space               |
| `logit_scale`                   | 1      | Learnable InfoNCE temperature                                              |
| `text_type_emb`, `image_type_emb`, `mask_token` | 3×768 | Modality / dropout tokens                                                  |

Total trainable: ~14M parameters. The SigLIP backbone (~200M) is frozen.

---

## Dataset

* `owj0421/polyvore`: 251,008 individual fashion items (image + title / category / description).
* `owj0421/polyvore-outfits`: 53,306 curated outfits.
* Both `nondisjoint_default` (item pools overlap between train/test) and `disjoint_default` (item pools fully separated) splits are evaluated.

### Training sample construction

```
Outfit:    [Jacket, Pants, Shoes]
              │ randomly hide one
Context:   [Jacket, Pants]        ← seen by the model
Target:    Shoes                  ← positive ground truth
Hard neg:  Same-category items, mid-similarity band   ← informative distractors
In-batch:  Other samples' targets ← free negatives
```

---

## Training

```bash
# 1. Cache SigLIP embeddings for the entire catalogue (one-time, ~2 hours)
python vectorize_data.py

# 2. Train (cosine LR, sample-wise text dropout, mid-similarity hard negatives).
#    The train split MUST match the FITB split used for eval — otherwise item-id
#    leakage inflates the disjoint score (see "Note on evaluation integrity" above).
#
#    Two variants — train both for parity with the published numbers:
python train.py --epochs 11 --split disjoint --save_prefix v3_disjoint                            # `text` variant
python train.py --epochs 11 --split disjoint --save_prefix v3_disjoint_notext --no_context_text   # `no_text` variant

# 3. Evaluate Fill-in-the-Blank on the matching split.
#    eval_fitb.py auto-detects --no_context_text from checkpoint hparams.
python eval_fitb.py --model_path v3_disjoint_epoch_11.pt        --split disjoint
python eval_fitb.py --model_path v3_disjoint_notext_epoch_11.pt --split disjoint

# (Optional) verify that train/test item pools don't overlap
python check_disjoint_leak.py

# 4. Launch the interactive demo
streamlit run app.py
```

`train.py` flags:

| Flag             | Default | Meaning                                          |
| ---------------- | :-----: | ------------------------------------------------ |
| `--epochs`       |   3     | Training epochs                                  |
| `--batch_size`   |   8     | Batch size                                       |
| `--lr`           |  1e-4   | Initial learning rate (cosine-annealed)          |
| `--num_layers`   |   4     | Transformer encoder layers                       |
| `--text_dropout` |  0.3    | Per-sample probability of replacing target text  |
| `--save_prefix`  | outfit_transformer | Checkpoint filename prefix             |
| `--split`        | nondisjoint | `nondisjoint` or `disjoint` — must match the eval split |
| `--no_context_text` | off   | Disable per-item text fusion (trains the `no_text` variant) |
| `--neg_strategy` | midsim  | `midsim` (10–40 pctile) or `topk` (naive most-similar) |
| `--resume_from`  |  None   | Warm-start from a checkpoint                     |

`eval_fitb.py` flags:

| Flag             | Default | Meaning                                          |
| ---------------- | :-----: | ------------------------------------------------ |
| `--model_path`   | latest  | Checkpoint to evaluate                           |
| `--split`        | nondisjoint | `nondisjoint` or `disjoint`                  |
| `--num_samples`  | 10000   | Number of FITB questions                         |

---

## Speed

| Stage                        | Time                |
| ---------------------------- | ------------------- |
| Catalogue vectorization      | ~2 hours (one-time) |
| Training epoch (M-series MPS)| ~9 minutes          |
| FITB evaluation (10k)        | ~3 minutes          |
| Demo inference per outfit    | ~1 second           |

All 251,008 items pass through SigLIP exactly once and are cached as `polyvore_embeddings.pt` (~1.7 GB). Training and evaluation use cached vectors only.

---

## File Structure

```
vectorize_data.py   Pre-compute SigLIP image+text vectors for the full catalogue
dataset.py          OutfitRetrievalDataset + collate; visual hard-negative mining
model.py            OutfitTransformer architecture
train.py            Training loop (InfoNCE + hard + in-batch negatives, cosine LR)
eval_fitb.py        Fill-in-the-blank evaluation on nondisjoint / disjoint splits
eval_siglip_baseline.py Backbone-only baselines (mean / max / sum) — no training
check_disjoint_leak.py  Sanity check: verifies train/test item pools are disjoint
app.py              Streamlit demo: autoregressive outfit composition with ranked looks
```

---

## Demo

The Streamlit app composes ranked outfits from a seed garment:

* Picks a seed (random or by index) from a 50k-item subset of the catalogue.
* Auto-blueprints incompatible slots (a dress hides `tops`/`bottoms`).
* Generates the top-1, -2, -3 looks autoregressively and ranks them by harmony score.
* Shows per-slot alternatives for the top pick.

```bash
streamlit run app.py
```