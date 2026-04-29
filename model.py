import torch
import torch.nn as nn
import open_clip
import math

class OutfitTransformer(nn.Module):
    def __init__(self, model_id="hf-hub:Marqo/marqo-fashionSigLIP", num_layers=4):
        super().__init__()
        # 1. Base Feature Extractor (Vision & Text) -> Frozen temporarily for efficiency
        self.siglip, _, self.preprocess_train = open_clip.create_model_and_transforms(model_id)
        self.tokenizer = open_clip.get_tokenizer(model_id)
        
        # Determine embedding dimension from the base model
        self.embed_dim = self.siglip.text_projection.shape[1] if hasattr(self.siglip, "text_projection") else 768
        
        # 2. Transformer Encoder Layer
        # Takes [Sequence of Outfits] and models interactions (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=8, 
            dim_feedforward=2048, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # [Strategy 1] Learnable Temperature for InfoNCE Loss
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        
        # [Strategy 3] Modality Embeddings and Mask Token
        self.text_type_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.image_type_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        
        # [Strategy 2] MLP Projection Head
        # Projects the reasoning outcome into the final visual compatibility space
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def encode_features(self, context_img_features, context_mask, target_text_features=None):
        """
        Directly process pre-computed embeddings for massive speedup.
        context_img_features: [Batch, Max_Seq, Embed_Dim]
        context_mask: [Batch, Max_Seq] (True means ignore/pad)
        target_text_features: [Batch, Embed_Dim] or None for Text Dropout
        """
        B, seq_len, embed_dim = context_img_features.shape
        
        # Apply Modality/Text Dropout logic
        if target_text_features is not None:
            text_features = target_text_features.unsqueeze(1) + self.text_type_emb
        else:
            text_features = self.mask_token.expand(B, 1, -1)
            
        # Apply Image Modality Embedding
        context_img_features = context_img_features + self.image_type_emb
        
        # The sequence is: [TARGET_ITEM_TOKEN, CONTEXT_ITEM_1, CONTEXT_ITEM_2, ...]
        transformer_input = torch.cat([text_features, context_img_features], dim=1) # [B, 1 + Seq, Embed]
        
        # The Target Token is at index 0 and shouldn't be masked out
        target_mask = torch.zeros((B, 1), dtype=torch.bool, device=context_mask.device)
        full_mask = torch.cat([target_mask, context_mask], dim=1)
        
        # Pass through Self-Attention block
        encoded_sequence = self.transformer(transformer_input, src_key_padding_mask=full_mask)
        
        # Extract the target item prediction embedding at index 0
        target_embedding_pred = encoded_sequence[:, 0, :]
        
        # Apply projection head
        target_embedding_pred = self.projection_head(target_embedding_pred)
        
        # L2 Normalize
        target_embedding_pred = torch.nn.functional.normalize(target_embedding_pred, p=2, dim=-1)
        
        return target_embedding_pred

    def forward(self, context_images, context_mask, target_text_tokens=None):
        """
        Backward compatible forward pass for raw images/tokens.
        Typically used in demo/inference mode where vectors aren't pre-computed.
        """
        B, seq_len, C, H, W = context_images.shape
        flat_images = context_images.view(-1, C, H, W)
        
        with torch.no_grad():
            img_features = self.siglip.encode_image(flat_images, normalize=True)
            if target_text_tokens is not None:
                text_features = self.siglip.encode_text(target_text_tokens, normalize=True)
            else:
                text_features = None
            
        img_features = img_features.view(B, seq_len, self.embed_dim)
        return self.encode_features(img_features, context_mask, text_features)
        
    def unfreeze_siglip(self, unfreeze_vision_blocks=1):
        """
        Unfreezes the last N blocks of the SigLIP vision encoder for Phase 2 fine-tuning.
        """
        self.siglip.train()
        for param in self.siglip.parameters():
            param.requires_grad = False
            
        # Unfreeze specific visual transformer blocks
        # (Marqo SigLIP uses a ViT backbone under OpenCLIP's standard module names)
        if hasattr(self.siglip, "visual") and hasattr(self.siglip.visual, "trunk") and hasattr(self.siglip.visual.trunk, "blocks"):
            blocks = self.siglip.visual.trunk.blocks
            for block in list(blocks)[-unfreeze_vision_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
                    
        print(f"SigLIP partially unfrozen: last {unfreeze_vision_blocks} vision blocks enabled for fine-tuning.")
