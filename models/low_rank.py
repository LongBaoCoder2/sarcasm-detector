import torch

# from torch import fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class SqueezeModule(nn.Module):
    def __init__(self, fusion_dim: int):
        self.norm = nn.LayerNorm(fusion_dim)

        self.squeeze = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 4, bias=False),
            nn.SiLU(),
            nn.Linear(fusion_dim // 4, fusion_dim, bias=True),
        )

    def forward(self, tensor_input):
        tensor_input = self.norm(tensor_input)
        squeezed_input = self.squeeze(tensor_input)
        squeezed_input = 1 + F.tanh(squeezed_input)
        tensor_input = tensor_input * squeezed_input
        return tensor_input


class LowRankFusion(nn.Module):
    def __init__(self, fusion_dim, rank, ff_dim=1024, lowrank_dropout=0.4):
        super(LowRankFusion, self).__init__()
        self.rank = rank
        self.fusion_dim = fusion_dim
        self.ff_dim = ff_dim

        self.image_squeeze = SqueezeModule(fusion_dim)
        self.text_squeeze = SqueezeModule(fusion_dim)

        self.text_factor = Parameter(
            torch.Tensor(self.rank, self.fusion_dim + 1, self.fusion_dim)
        )
        self.image_factor = Parameter(
            torch.Tensor(self.rank, self.fusion_dim + 1, self.fusion_dim)
        )
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.fusion_dim))
        self.dropout = nn.Dropout(lowrank_dropout)

        # Gating network for selective feature propagation
        self.gate_network = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.GELU(),
        )
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.text_factor)
        nn.init.xavier_normal_(self.image_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        nn.init.constant_(self.fusion_bias, 1e-2)

    def forward(self, text_embedding, image_embedding):
        batch_size = text_embedding.size(0)
        DTYPE = text_embedding.dtype

        image_embedding = self.image_squeeze(image_embedding)
        text_embedding = self.text_squeeze(text_embedding)

        # Add bias term to embeddings for low-rank fusion
        text_h = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=DTYPE, device=text_embedding.device),
                text_embedding,
            ),
            dim=1,
        )
        image_h = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=DTYPE, device=image_embedding.device),
                image_embedding,
            ),
            dim=1,
        )

        # Low-rank fusion
        fusion_text = torch.matmul(text_h, self.text_factor)
        fusion_image = torch.matmul(image_h, self.image_factor)
        fusion_zy = fusion_text * fusion_image

        # Compute final fused output
        fused_embedding = (
            torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze()
            + self.fusion_bias
        )
        fused_embedding = self.dropout(fused_embedding)
        gate_values = self.gate_network(fused_embedding)

        return gate_values
