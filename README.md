# OutfitTransformer: Complementary Fashion Item Retrieval

An AI fashion stylist that completes a partial outfit. Given any seed garment and target slots (e.g. *bottoms, shoes, bag*), the model retrieves complementary items that match the global style of the rest of the outfit — not just visually similar items.

Built on a frozen fashion-specific SigLIP backbone with a trained transformer that reasons over multimodal (image + text) representations of every item in the outfit.

---

## Results

Evaluated on the Polyvore-Outfits Fill-in-the-Blank benchmark (4-choice multiple choice, 10,000 questions per split).

| Split                | Random | OutfitTransformer (paper, CVPR 2022) | **Ours** | Δ        |
| -------------------- | :----: | :----------------------------------: | :------: | :------: |
| Nondisjoint (10k)    | 25.00  | 67.10                                | **66.85**| −0.25    |
| **Disjoint (10k)**   | 25.00  | 59.48                                | **68.60**| **+9.12**|

**Disjoint** means train and test item pools do not overlap — a stronger test of generalization to unseen products. Our model is on-par on the standard nondisjoint setting and substantially ahead on disjoint, suggesting the frozen fashion-specific backbone generalizes to new inventory better than the end-to-end-trained ResNet baseline.

State-of-the-art benchmark from: *Sarkar et al., "OutfitTransformer: Learning Outfit Representations for Fashion Recommendation", CVPR 2022* ([arXiv:2204.04812](https://arxiv.org/abs/2204.04812)).

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

# 2. Train (cosine LR, sample-wise text dropout, mid-similarity hard negatives)
python train.py --epochs 11 --save_prefix v3

# 3. Evaluate Fill-in-the-Blank on either split
python eval_fitb.py --model_path v3_epoch_10.pt --split nondisjoint
python eval_fitb.py --model_path v3_epoch_10.pt --split disjoint

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
app.py              Streamlit demo: autoregressive outfit composition with ranked looks
grid_search.py      Optional hyperparameter sweep
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