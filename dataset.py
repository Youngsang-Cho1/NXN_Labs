import torch
from torch.utils.data import Dataset
import random

class OutfitRetrievalDataset(Dataset):
    def __init__(self, outfit_dataset, items_dataset, transform=None, embeddings_dict=None, num_negatives=5, max_outfit_size=8):
        self.outfit_dataset = outfit_dataset
        self.items_dataset = items_dataset
        self.transform = transform
        self.embeddings_dict = embeddings_dict
        self.num_negatives = num_negatives
        self.max_outfit_size = max_outfit_size
        
        print("Building item_id to index lookup and category maps...")
        self.item_idx_map = {}
        self.category_to_item_ids = {}
        
        # ⚡ Convert to plain Python lists FIRST - this triggers one bulk Arrow read
        # instead of 251,008 individual HuggingFace __getitem__ calls
        all_ids        = list(self.items_dataset['item_id'])
        all_categories = list(self.items_dataset['category'])
        all_titles     = list(self.items_dataset['title'])
        
        for idx, (item_id, cat, title) in enumerate(zip(all_ids, all_categories, all_titles)):
            self.item_idx_map[item_id] = idx
            cat = str(cat).lower().strip() or str(title).lower().strip() or "unknown"
            if cat not in self.category_to_item_ids:
                self.category_to_item_ids[cat] = []
            self.category_to_item_ids[cat].append(item_id)
            
        self.all_item_ids = list(self.item_idx_map.keys())
        print(f"✅ Loaded {len(self.all_item_ids)} items across {len(self.category_to_item_ids)} categories.")

    def _get_item_data(self, item_id):
        if self.embeddings_dict is not None and item_id in self.embeddings_dict:
            emb = self.embeddings_dict[item_id]
            return emb["image"], emb["text"]
            
        idx = self.item_idx_map.get(item_id)
        if idx is None: return None, None
        
        item = self.items_dataset[idx]
        image = item["image"]
        
        text = item.get("title", "")
        if not text: text = item.get("description", "")
        if not text: text = item.get("url_name", "")
        if not text: text = item.get("category", "")
            
        if getattr(image, "mode", "RGB") != "RGB":
            image = image.convert("RGB")
            
        return self.transform(image) if self.transform else image, text

    def __len__(self):
        return len(self.outfit_dataset)

    def __getitem__(self, idx):
        outfit = self.outfit_dataset[idx]
        
        # Valid outfit items checking our index
        outfit_items = [i['item_id'] for i in outfit['items'] if i['item_id'] in self.item_idx_map]
        
        if len(outfit_items) < 2:
            return self.__getitem__((idx + 1) % len(self))
            
        # Target Item (Positive Match)
        target_idx = random.randint(0, len(outfit_items) - 1)
        target_item_id = outfit_items[target_idx]
        
        # Context Items (Partial Outfit)
        context_item_ids = [outfit_items[i] for i in range(len(outfit_items)) if i != target_idx]
        
        target_image, target_text = self._get_item_data(target_item_id)
        
        context_images = []
        for i_id in context_item_ids[:self.max_outfit_size]:
            img, _ = self._get_item_data(i_id)
            context_images.append(img)
            
        # Negatives (Hard + Easy Strategy)
        negative_images = []
        
        t_idx = self.item_idx_map[target_item_id]
        t_cat = str(self.items_dataset[t_idx].get("category", "")).lower().strip()
        if not t_cat: t_cat = str(self.items_dataset[t_idx].get("title", "")).lower().strip()
        if not t_cat: t_cat = "unknown"
        
        # 1. Visual Online Hard Negative Mining
        hard_candidates = self.category_to_item_ids.get(t_cat, [])
        valid_candidates = [c for c in hard_candidates if c not in outfit_items]
        
        # If we have precomputed embeddings, do Visual Hard Negative Mining
        if self.embeddings_dict is not None and target_item_id in self.embeddings_dict and len(valid_candidates) >= self.num_negatives:
            # Sample a pool to keep matrix multiplication very fast
            pool_size = min(100, len(valid_candidates))
            pool_ids = random.sample(valid_candidates, pool_size)
            
            # Fetch embeddings
            target_emb = self.embeddings_dict[target_item_id]["image"]
            candidate_embs = torch.stack([self.embeddings_dict[c_id]["image"] for c_id in pool_ids])
            
            # Compute Cosine Similarity (vectors are already L2 normalized)
            sims = torch.matmul(candidate_embs, target_emb)
            
            # Pick top-K most visually similar items as the hard negatives!
            top_k_indices = torch.topk(sims, k=self.num_negatives).indices
            hard_negative_ids = [pool_ids[i] for i in top_k_indices]
            
            for neg_id in hard_negative_ids:
                img, _ = self._get_item_data(neg_id)
                negative_images.append(img)
        else:
            # Fallback to random category / easy negatives if embeddings not loaded
            attempts = 0
            while len(negative_images) < 1 and attempts < 10:
                attempts += 1
                if not valid_candidates: break
                neg_id = random.choice(valid_candidates)
                img, _ = self._get_item_data(neg_id)
                negative_images.append(img)
                break
                
            # 2. Easy Negatives (Random pool)
            while len(negative_images) < self.num_negatives:
                neg_id = random.choice(self.all_item_ids)
                if neg_id not in outfit_items:
                    img, _ = self._get_item_data(neg_id)
                    negative_images.append(img)
                
        return {
            "context_images": context_images,
            "target_image": target_image,
            "target_text": target_text,
            "negative_images": negative_images
        }

def outfit_collate_fn(batch, max_outfit_size=8):
    """
    Collate function that handles variable length outfit contexts.
    Automatically supports both raw image tensors [C, H, W] and feature vectors [D].
    """
    target_images = torch.stack([item["target_image"] for item in batch])
    negative_images = torch.stack([torch.stack(item["negative_images"]) for item in batch])
    
    # target_texts can be strings (raw) or tensors (precomputed vectors)
    if isinstance(batch[0]["target_text"], torch.Tensor):
        target_texts = torch.stack([item["target_text"] for item in batch])
    else:
        target_texts = [item["target_text"] for item in batch]
    
    batch_size = len(batch)
    
    sample_ctx = None
    for item in batch:
        if len(item["context_images"]) > 0:
            sample_ctx = item["context_images"][0]
            break
            
    if sample_ctx is not None:
        shape = sample_ctx.shape
        padded_context = torch.zeros((batch_size, max_outfit_size, *shape))
        context_pad_mask = torch.ones((batch_size, max_outfit_size), dtype=torch.bool)
        
        for b_idx, item in enumerate(batch):
            seq_len = len(item["context_images"])
            if seq_len > 0:
                padded_context[b_idx, :seq_len] = torch.stack(item["context_images"])
                context_pad_mask[b_idx, :seq_len] = False
    else:
        # Fallback if no context items across entire batch (rare edge case)
        padded_context = torch.zeros((batch_size, max_outfit_size, 1))
        context_pad_mask = torch.ones((batch_size, max_outfit_size), dtype=torch.bool)

    return {
        "context_images": padded_context,
        "context_mask": context_pad_mask,
        "target_image": target_images,
        "target_texts": target_texts,
        "negative_images": negative_images
    }
