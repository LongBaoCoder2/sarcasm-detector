import torch.nn as nn

from models.gating import GatingModule
from models.low_rank import LowRankFusion

# import torch.nn.functional as F
# from torch.nn import Parameter


class SarcasmModel(nn.Module):
    def __init__(
        self,
        text_embeder,
        image_embeder,
        dropout=0.3,
        text_dropout=0.3,
        image_dropout=0.3,
        fusion_dim=256,
        rank=64,
        attention_dim=1024,
        output_dim=4,
    ):
        super(SarcasmModel, self).__init__()

        self.fusion_dim = fusion_dim
        self.text_embeder = text_embeder
        self.image_embeder = image_embeder

        # Text projection to fusion dimension
        self.text_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, attention_dim, bias=False),
            nn.GELU(),
            nn.Linear(attention_dim, fusion_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Image projection to fusion dimension
        self.image_text_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, attention_dim, bias=False),
            nn.GELU(),
            nn.Linear(attention_dim, fusion_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.image_proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, attention_dim, bias=False),
            nn.GELU(),
            nn.Linear(attention_dim, fusion_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Low-rank fusion module
        self.low_rank_fusion = LowRankFusion(fusion_dim, rank, fusion_dim, dropout)
        self.low_rank_bottleneck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 4),
                    nn.GELU(),
                    nn.Linear(fusion_dim // 4, fusion_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 4),
                    nn.GELU(),
                    nn.Linear(fusion_dim // 4, fusion_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
            ]
        )

        self.fusion_last = LowRankFusion(fusion_dim, rank, fusion_dim, dropout)
        self.fusion_last_bottleneck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 4),
                    nn.GELU(),
                    nn.Linear(fusion_dim // 4, fusion_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 4),
                    nn.GELU(),
                    nn.Linear(fusion_dim // 4, fusion_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
            ]
        )

        self.text_proj_last = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, attention_dim, bias=False),
            nn.GELU(),
            nn.Linear(attention_dim, fusion_dim, bias=False),
            nn.GELU(),
            nn.Dropout(text_dropout),
        )

        self.image_text_proj_last = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, attention_dim, bias=False),
            nn.GELU(),
            nn.Linear(attention_dim, fusion_dim, bias=False),
            nn.GELU(),
            nn.Dropout(image_dropout),
        )

        self.gating = GatingModule(fusion_dim)
        self.gating_dropout = nn.Dropout(dropout)

        # Classifier
        self.text_classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, output_dim),
        )

        self.image_classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, output_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, output_dim),
        )

        self.init_weights()

    def init_weights(self):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Apply the initialization to each submodule
        self.apply(init_layer)

    def forward(self, image, img_txt, text):
        # Process the img_txt
        input_ids = img_txt["input_ids"]
        attention_mask = img_txt["attention_mask"]
        image_text_embedding = self.text_embeder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        # Project text embeddings to fusion dimension
        image_text_embedding = self.image_text_proj(image_text_embedding)

        # Get image embeddings
        image_embedding = self.image_embeder.get_image_features(**image)
        # Project image embeddings to fusion dimension
        image_embedding = self.image_proj(image_embedding)

        # Low-rank fusion of image and text (with residual connections)
        fused_embedding = self.low_rank_fusion(image_text_embedding, image_embedding)
        image_text_embedding = self.low_rank_bottleneck[0](image_text_embedding)
        image_embedding = self.low_rank_bottleneck[1](image_embedding)
        fused_embedding_res = fused_embedding + (
            image_text_embedding + image_embedding
        )  # Residual connection
        #         fused_embedding_res = fused_embedding

        # Get text embeddings and apply mean pooling
        input_ids = text["input_ids"]
        attention_mask = text["attention_mask"]
        text_embedding = self.text_embeder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        # Project text embeddings to fusion dimension
        text_embedding = self.text_proj(text_embedding)

        # Two path get lor, apply residual connection to both paths
        fused_out = (
            self.image_text_proj_last(fused_embedding_res) + fused_embedding_res
        )  # Residual connection
        text_caption_out = (
            self.text_proj_last(text_embedding) + text_embedding
        )  # Residual connection

        # Cross-attention between text and fused embeddings
        cross_attn_out = self.fusion_last(fused_out, text_caption_out)
        fused_out = self.fusion_last_bottleneck[0](fused_out)
        text_caption_out = self.fusion_last_bottleneck[1](text_caption_out)
        cross_attn_out_res = cross_attn_out + (fused_out + text_caption_out)
        #         cross_attn_out_res = cross_attn_out

        pooled_output = self.gating(cross_attn_out_res)
        pooled_output = self.gating_dropout(pooled_output)

        # Classification
        classify = self.classifier(pooled_output)
        text_aux_output = self.text_classifier(text_caption_out)
        image_aux_output = self.image_classifier(fused_out)

        return classify, text_aux_output, image_aux_output
