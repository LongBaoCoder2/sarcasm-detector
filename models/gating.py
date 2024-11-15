import torch.nn as nn
import torch.nn.functional as F

# from torch.nn import Parameter


class GatingModule(nn.Module):
    def __init__(self, input_dim):
        super(GatingModule, self).__init__()
        # Gating mechanism using a simple linear layer
        self.d_dim = input_dim
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, fused_embedding):
        # Compute gating score
        gating_score = F.sigmoid(self.gate(fused_embedding))
        pooled_output = gating_score * fused_embedding
        return pooled_output
