import os
import torch
import argparse
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from dataset import OutfitRetrievalDataset, outfit_collate_fn
from model import OutfitTransformer

def infonce_loss_with_hard_negatives(pred_target, pos_features, neg_features, logit_scale=40.0):
    """
    InfoNCE loss utilizing both in-batch negatives and explicitly sampled hard negatives.
    pred_target: [B, D]
    pos_features: [B, D]
    neg_features: [B, Num_Neg, D]
    """
    B, D = pred_target.shape
    
    # 1. Similarity with Positives (shape: [B, 1])
    pos_sim = (pred_target * pos_features).sum(dim=-1, keepdim=True)
    
    # 2. Similarity with Explicit Negatives (shape: [B, Num_Neg])
    neg_sim = (pred_target.unsqueeze(1) * neg_features).sum(dim=-1)
    
    # 3. Similarity with In-Batch Negatives
    batch_sim = pred_target @ pos_features.t() # [B, B]
    
    if B > 1:
        # Mask out diagonal (which are positives)
        mask = ~torch.eye(B, dtype=torch.bool, device=pred_target.device)
        in_batch_neg_sim = batch_sim[mask].view(B, B - 1)
        # 4. Concatenate all similarities! [B, 1 (Pos) + Num_Neg (Hard) + B-1 (In-batch)]
        logits = torch.cat([pos_sim, neg_sim, in_batch_neg_sim], dim=1) * logit_scale
    else:
        logits = torch.cat([pos_sim, neg_sim], dim=1) * logit_scale
        
    # The positive class is ALWAYS at index 0 because of our concatenation order
    labels = torch.zeros(B, dtype=torch.long, device=pred_target.device)
    
    return torch.nn.functional.cross_entropy(logits, labels)

def main():
    parser = argparse.ArgumentParser(description="Train OutfitTransformer")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--text_dropout", type=float, default=0.3, help="Text dropout probability (per-sample)")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer encoder layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--resume_from", type=str, default=None, help="Optional checkpoint to warm-start from")
    parser.add_argument("--save_prefix", type=str, default="outfit_transformer", help="Filename prefix for saved checkpoints")
    parser.add_argument("--split", type=str, default="nondisjoint",
                        choices=["nondisjoint", "disjoint"],
                        help="Polyvore split to train on. Must match the FITB split used for eval to avoid item-id leak.")
    parser.add_argument("--no_context_text", action="store_true",
                        help="Ablation: disable per-item text fusion in context (image only)")
    parser.add_argument("--neg_strategy", type=str, default="midsim", choices=["midsim", "topk"],
                        help="Hard negative mining strategy. 'topk' ablation tests whether mid-sim is better than naive top-K.")
    args = parser.parse_args()
    print(f"\n🔧 Hyperparameters: lr={args.lr}, epochs={args.epochs}, text_dropout={args.text_dropout}, num_layers={args.num_layers}")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f" Using Device: {device}")
    
    model = OutfitTransformer(num_layers=args.num_layers).to(device)
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"📥 Warm-starting from {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location=device, weights_only=True), strict=False)
    model.train()
    
    print("Loading Polyvore Datasets...")
    polyvore_items = load_dataset("owj0421/polyvore", split="data")
    polyvore_outfits = load_dataset("owj0421/polyvore-outfits", f"{args.split}_default", split="train")
    print(f"📂 Training split: {args.split}_default ({len(polyvore_outfits)} outfits)")
    
    embeddings_dict = None
    emb_path = "polyvore_embeddings.pt"
    if os.path.exists(emb_path):
        print(f"✅ Found pre-computed embeddings at {emb_path}! Loading for hyper-fast vector training...")
        # map_location configures loading natively to cpu before dispatching to mps/cuda
        embeddings_dict = torch.load(emb_path, map_location="cpu", weights_only=True)
    else:
        print("⚠️ No pre-computed embeddings found. Defaulting to raw image processing (Slow). Run vectorize_data.py first for speed.")
    
    print("Preparing full dataset training. (This will parse all 53,000 outfits!)")
    
    train_dataset = OutfitRetrievalDataset(
        polyvore_outfits,
        polyvore_items,
        transform=model.preprocess_train if not embeddings_dict else None,
        embeddings_dict=embeddings_dict,
        num_negatives=2,
        neg_strategy=args.neg_strategy,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # With precomputed vectors, no disk I/O needed -> num_workers=0 avoids
        # forking 4 processes that each copy the multi-GB embeddings dict
        num_workers=0 if embeddings_dict else 4,
        collate_fn=outfit_collate_fn
    )
    
    trainable_params = (
        list(model.transformer.parameters()) +
        list(model.context_fuse.parameters()) +
        list(model.projection_head.parameters()) +
        [model.logit_scale, model.text_type_emb, model.image_type_emb, model.mask_token]
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"🧠 Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    for epoch in range(args.epochs):
        print(f"\n --- Epoch {epoch+1} Started ---")
        progress_bar = tqdm(train_loader)
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            if embeddings_dict:
                # FAST VECTOR ROUTE
                context_img_features = batch["context_images"].to(device).clone()
                context_mask = batch["context_mask"].to(device)

                # [Strategy 4] Context Shuffle Augmentation (out-of-place on clone above)
                B_size = context_mask.shape[0]
                for b_idx in range(B_size):
                    valid_idx = (~context_mask[b_idx]).nonzero(as_tuple=True)[0]
                    if len(valid_idx) > 1:
                        shuffled_features = context_img_features[b_idx, valid_idx][torch.randperm(len(valid_idx))]
                        context_img_features[b_idx, valid_idx] = shuffled_features

                target_img_features = batch["target_image"].to(device)
                negative_img_features = batch["negative_images"].to(device)
                target_text_features = batch["target_texts"].to(device)
                if args.no_context_text:
                    context_text_features = None
                else:
                    context_text_features = (
                        batch["context_texts"].to(device) if batch.get("context_texts") is not None else None
                    )

                # [Strategy 3] Sample-wise text dropout: each sample independently drops
                # text with prob=text_dropout. Critical because FITB eval is text-free,
                # so mask_token needs many more training samples than batch-level dropout
                # (1 coin flip per batch) ever delivered.
                text_drop_mask = torch.rand(B_size, device=device) < args.text_dropout

                pred_target_emb = model.encode_features(
                    context_img_features, context_mask,
                    target_text_features, text_drop_mask=text_drop_mask,
                    context_text_features=context_text_features,
                )
                
                # [Strategy 1] Calculate Loss with Learnable Temperature
                # Explicitly L2 normalize to guarantee cosine similarity
                target_img_features = torch.nn.functional.normalize(target_img_features, p=2, dim=-1)
                negative_img_features = torch.nn.functional.normalize(negative_img_features, p=2, dim=-1)
                
                loss = infonce_loss_with_hard_negatives(pred_target_emb, target_img_features, negative_img_features, logit_scale=model.logit_scale.exp())
                
            else:
                # SLOW RAW IMAGE ROUTE
                context_images = batch["context_images"].to(device)
                context_mask = batch["context_mask"].to(device)
                
                # [Strategy 4] Context Shuffle Augmentation
                B_size = context_mask.shape[0]
                for b_idx in range(B_size):
                    valid_idx = (~context_mask[b_idx]).nonzero(as_tuple=True)[0]
                    if len(valid_idx) > 1:
                        shuffled_imgs = context_images[b_idx, valid_idx][torch.randperm(len(valid_idx))]
                        context_images[b_idx, valid_idx] = shuffled_imgs

                target_image = batch["target_image"].to(device)
                negative_images = batch["negative_images"].to(device)
                
                target_text_tokens = model.tokenizer(batch["target_texts"]).to(device)
                
                # [Strategy 3] Modality/Text Dropout for raw tokens
                if torch.rand(1).item() < 0.2:
                    target_text_tokens = None
                
                # 1. Forward Pass -> Get Model's idealized target prediction
                pred_target_emb = model(context_images, context_mask, target_text_tokens)
                
                # 2. Extract Reality
                with torch.no_grad():
                    pos_features = model.siglip.encode_image(target_image, normalize=True)
                    B, num_neg, C, H, W = negative_images.shape
                    neg_flat = negative_images.view(-1, C, H, W)
                    neg_features = model.siglip.encode_image(neg_flat, normalize=True)
                    negative_img_features = neg_features.view(B, num_neg, model.embed_dim)
                    
                # 3. Calculate Outfit Complementary Retrieval Loss with Learnable Temperature
                loss = infonce_loss_with_hard_negatives(pred_target_emb, pos_features, negative_img_features, logit_scale=model.logit_scale.exp())
            
            # 4. Backward Pass
            loss.backward()
            optimizer.step()
            
            # Update tqdm progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        print(f"\n Epoch {epoch+1} completed! LR now {optimizer.param_groups[0]['lr']:.2e}. Saving weights...")
        torch.save({
            "state_dict": model.state_dict(),
            "hparams": {
                "num_layers": args.num_layers,
                "text_dropout": args.text_dropout,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epoch": epoch + 1,
                "split": args.split,
                "no_context_text": args.no_context_text,
                "neg_strategy": args.neg_strategy,
            },
        }, f"{args.save_prefix}_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
