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
        
        # 2. Self-attention encoder: lets context items reason about each other.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=2048,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 2b. Cross-attention decoder: target query attends to context.
        # Splitting query from context (vs concatenating) gives the query a
        # dedicated path to ask "what do I need given this outfit?" instead of
        # being one undifferentiated token among many.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=2048,
            activation="gelu",
            batch_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Fuse per-item image + text emb into a single context token.
        self.context_fuse = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        
        # [Strategy 1] Learnable Temperature for InfoNCE Loss
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        
        # [Strategy 3] Modality Embeddings and Mask Token
        # Scale 0.1 (not 0.02): SigLIP features are L2-normalized (~1.0); too-small
        # modality emb becomes pure noise and gets ignored during attention.
        self.text_type_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.1)
        self.image_type_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.1)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.1)
        
        # [Strategy 2] MLP Projection Head
        # Projects the reasoning outcome into the final visual compatibility space
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def encode_features(self, context_img_features, context_mask,
                        target_text_features=None, text_drop_mask=None,
                        context_text_features=None):
        """
        context_img_features:   [B, Seq, D]
        context_text_features:  [B, Seq, D] or None (when None, fall back to image-only context)
        context_mask:           [B, Seq] (True = pad)
        target_text_features:   [B, D] or None
        text_drop_mask:         [B] bool, True = drop target text (use mask_token)
        """
        B, seq_len, D = context_img_features.shape

        # ---- Build target query token ----
        if target_text_features is not None:
            query = target_text_features.unsqueeze(1) + self.text_type_emb  # [B, 1, D]
            if text_drop_mask is not None:
                mask_tok = self.mask_token.expand(B, 1, -1)
                drop = text_drop_mask.view(B, 1, 1).to(query.dtype)
                query = drop * mask_tok + (1 - drop) * query
        else:
            query = self.mask_token.expand(B, 1, -1)

        # ---- Fuse per-item image + text into context tokens ----
        if context_text_features is not None:
            fused = self.context_fuse(
                torch.cat([context_img_features, context_text_features], dim=-1)
            )
        else:
            fused = context_img_features
        context_tokens = fused + self.image_type_emb  # [B, Seq, D]

        # ---- Self-attention: context items reason about each other ----
        context_encoded = self.transformer(context_tokens, src_key_padding_mask=context_mask)

        # ---- Cross-attention: query asks the context what it needs ----
        # tgt = query [B, 1, D], memory = context_encoded [B, Seq, D]
        decoded = self.cross_attn(
            tgt=query,
            memory=context_encoded,
            memory_key_padding_mask=context_mask,
        )  # [B, 1, D]

        target_embedding_pred = self.projection_head(decoded.squeeze(1))
        return torch.nn.functional.normalize(target_embedding_pred, p=2, dim=-1)

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
