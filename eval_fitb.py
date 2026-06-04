import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import argparse
from datasets import load_dataset
from tqdm import tqdm
from model import OutfitTransformer

def main():
    parser = argparse.ArgumentParser(description="Evaluate OutfitTransformer FITB")
    parser.add_argument("--model_path", type=str, default="outfit_transformer_epoch_11.pt", help="Path to model weights")
    parser.add_argument("--num_layers", type=int, default=4, help="Must match model's num_layers")
    parser.add_argument("--split", type=str, default="nondisjoint",
                        choices=["nondisjoint", "disjoint"],
                        help="Polyvore split (disjoint = items don't overlap between train/test)")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of FITB questions to eval")
    parser.add_argument("--no_context_text", action="store_true",
                        help="Disable context text in eval. Auto-set from checkpoint hparams if available.")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Using Device: {device}")

    # 1. Load Model
    num_layers = args.num_layers
    state_dict = None
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, weights_only=True, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            hparams = ckpt.get("hparams", {})
            if "num_layers" in hparams:
                num_layers = hparams["num_layers"]
                print(f"📐 Using num_layers={num_layers} from checkpoint hparams")
            if hparams.get("no_context_text", False) and not args.no_context_text:
                args.no_context_text = True
                print(f"📐 Auto-enabling --no_context_text from checkpoint hparams")
        else:
            state_dict = ckpt

    model = OutfitTransformer(num_layers=num_layers).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded weights from {args.model_path}")
    else:
        print(f"⚠️  No checkpoint found at {args.model_path}, using random weights.")

    model.eval()
    
    # 2. Load Datasets & Build the Translation Maps
    print("Loading Polyvore data splits to build item ID mappings...")
    polyvore_items = load_dataset("owj0421/polyvore", split="data")
    
    # O(1) direct dictionary mapping from item_id string to row index
    item_id_map = {item_id: idx for idx, item_id in enumerate(polyvore_items['item_id'])}
    
    embeddings_dict = None
    emb_path = "polyvore_embeddings.pt"
    if os.path.exists(emb_path):
        print(f"✅ Found pre-computed embeddings at {emb_path}! Loading for instant evaluation...")
        embeddings_dict = torch.load(emb_path, map_location="cpu", weights_only=True)
    
    default_config = f"{args.split}_default"
    fitb_config = f"{args.split}_fill_in_the_blank"
    print(f"Evaluating on split: {args.split}")

    nd_def = load_dataset("owj0421/polyvore-outfits", default_config)

    set_to_item = {}
    for split in ['train', 'validation', 'test']:
        for row in nd_def[split]:
            set_id = row['set_id']
            for it in row['items']:
                key = f"{set_id}_{it['index']}"
                set_to_item[key] = it['item_id']

    fitb_test = load_dataset("owj0421/polyvore-outfits", fitb_config, split="test")
    fitb_test = fitb_test.select(range(min(args.num_samples, len(fitb_test))))
    
    correct_count = 0
    total_count = 0
    
    def get_data(key):
        mapped_id = set_to_item.get(key)
        if not mapped_id: return None
        idx = item_id_map.get(mapped_id)
        if idx is None: return None
        return polyvore_items[idx]
        
    print(f"\n Running Fill-In-The-Blank (FITB) Evaluation Benchmark ({len(fitb_test)} Questions)...")
    
    progress_bar = tqdm(fitb_test)
    for row in progress_bar:
        context_keys = row['items']
        candidate_keys = row['candidates'] # List of exactly 4 multiple choices
        true_label = row['label'] # The correct answer index (0, 1, 2, or 3)
        
        # Build context image AND text tensors representing the partial outfit
        context_img_tensors = []
        context_txt_tensors = []
        for ck in context_keys:
            data = get_data(ck)
            if not data: continue

            if embeddings_dict and data['item_id'] in embeddings_dict:
                emb = embeddings_dict[data['item_id']]
                context_img_tensors.append(emb["image"].unsqueeze(0))
                context_txt_tensors.append(emb["text"].unsqueeze(0))
            else:
                img = data["image"]
                if getattr(img, "mode", "RGB") != "RGB": img = img.convert("RGB")
                context_img_tensors.append(model.preprocess_train(img).unsqueeze(0))

        if not context_img_tensors: continue

        context_images_tensor = torch.stack(context_img_tensors, dim=1).to(device)
        context_mask = torch.zeros((1, len(context_img_tensors)), dtype=torch.bool).to(device)
        context_texts_tensor = (
            torch.stack(context_txt_tensors, dim=1).to(device) if context_txt_tensors else None
        )
        if args.no_context_text:
            context_texts_tensor = None

        # Pure-visual FITB: no target text hint, but context text is available.
        with torch.no_grad():
            ideal_emb = model.encode_features(
                context_images_tensor, context_mask,
                target_text_features=None,
                context_text_features=context_texts_tensor,
            )
            
        candidate_distances = []
        
        for cand_key in candidate_keys:
            cand_data = get_data(cand_key)
            if not cand_data: 
                candidate_distances.append(999.0) # Penalize missing data
                continue
            
            if embeddings_dict and cand_data['item_id'] in embeddings_dict:
                cand_vis_emb = embeddings_dict[cand_data['item_id']]["image"].unsqueeze(0).to(device)
            else:
                c_img = cand_data["image"]
                if getattr(c_img, "mode", "RGB") != "RGB": c_img = c_img.convert("RGB")
                c_img_tensor = model.preprocess_train(c_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    cand_vis_emb = model.siglip.encode_image(c_img_tensor, normalize=True)
                
            dist = torch.norm(ideal_emb - cand_vis_emb, p=2, dim=-1).item()
            candidate_distances.append(dist)
            
        if len(candidate_distances) == 4 and min(candidate_distances) < 900.0:
            predicted_label = candidate_distances.index(min(candidate_distances))
            if predicted_label == true_label:
                correct_count += 1
            total_count += 1
            
            current_acc = (correct_count / total_count) * 100
            progress_bar.set_postfix({"Acc": f"{current_acc:.2f}%"})
            
    final_acc = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"\n🎉 Final FITB Accuracy: {final_acc:.2f}% over {total_count} questions")
    print("Random guessing for 4 choices is 25.00%. Anything above > 50% is state-of-the-art for visual semantic fusion without complex metric learning!")
    
    with open("fitb_accuracy.txt", "w") as f:
        f.write(f"{final_acc:.2f}")

if __name__ == "__main__":
    main()
