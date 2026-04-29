import torch
from datasets import load_dataset
from tqdm import tqdm
import open_clip

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"====================================")
    print(f" OutfitTransformer Vectorizer")
    print(f" Using Device: {device}")
    print(f"====================================")
    
    model_id = "hf-hub:Marqo/marqo-fashionSigLIP"
    print(f"Loading Base Model ({model_id})...")
    siglip, _, preprocess = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    siglip = siglip.to(device)
    siglip.eval()
    
    print("\nLoading Polyvore Item Dataset from HuggingFace...")
    polyvore_items = load_dataset("owj0421/polyvore", split="data")
    
    num_items = len(polyvore_items)
    print(f"Total Unique Items to Vectorize: {num_items}")
    
    embeddings_dict = {}
    
    # Process in batches for speed
    batch_size = 64
    
    print("\nStarting Vectorization Process (This only needs to be done ONCE!)")
    progress_bar = tqdm(range(0, num_items, batch_size))
    
    for i in progress_bar:
        # Extract batch rows from HF dataset
        batch = polyvore_items[i : i + batch_size]
        images = batch["image"]
        titles = batch.get("title", [""] * len(images))
        descs = batch.get("description", [""] * len(images))
        url_names = batch.get("url_name", [""] * len(images))
        categories = batch.get("category", [""] * len(images))
        item_ids = batch["item_id"]
        
        proc_images = []
        texts = []
        
        for j in range(len(images)):
            # 1. Process Image
            img = images[j]
            if getattr(img, "mode", "RGB") != "RGB":
                img = img.convert("RGB")
            proc_images.append(preprocess(img))
            
            # 2. Heuristic text fallback matching dataset.py logic
            text = str(titles[j] if titles and j < len(titles) else "")
            if not text and descs and j < len(descs): text = str(descs[j])
            if not text and url_names and j < len(url_names): text = str(url_names[j])
            if not text and categories and j < len(categories): text = str(categories[j])
            if not text: text = "clothing item"
            texts.append(text)
            
        img_tensor = torch.stack(proc_images).to(device)
        # Handle text tokenization padding
        text_tokens = tokenizer(texts).to(device)
        
        with torch.no_grad():
            img_feats = siglip.encode_image(img_tensor, normalize=True)
            text_feats = siglip.encode_text(text_tokens, normalize=True)
            
        img_feats = img_feats.cpu()
        text_feats = text_feats.cpu()
        
        for j, i_id in enumerate(item_ids):
            embeddings_dict[i_id] = {
                "image": img_feats[j].clone(),
                "text": text_feats[j].clone()
            }
            
    save_path = "polyvore_embeddings.pt"
    print(f"\nVectorization Complete!")
    print(f"Saving compiled vector dictionary to {save_path}...")
    torch.save(embeddings_dict, save_path)
    print("Done! You can now run train.py/eval_fitb.py at hyper-speed.")

if __name__ == "__main__":
    main()
