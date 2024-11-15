import torch
import torch.nn as nn

# import torch.nn.functional as F
from attention import CrossAttention, ResidualBlock


class RollingQuery(nn.Module):
    def __init__(
        self, query_dim, modality_dim, output_dim, num_stages=3, num_queries=4
    ):
        super(RollingQuery, self).__init__()

        # Initialize learned query vectors (4 vectors of dimension query_dim)
        self.learned_queries = nn.Parameter(torch.randn(num_queries, query_dim))

        # Cross-attention for both modalities (text and image)
        self.text_cross_attention = CrossAttention(query_dim, modality_dim)
        self.image_cross_attention = CrossAttention(query_dim, modality_dim)

        # MLP to process the fused modality information
        self.fusion_mlp = nn.Linear(modality_dim, query_dim)

        # Stages like an RNN, updating the query vectors
        self.num_stages = num_stages
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(query_dim) for _ in range(num_stages)]
        )

        # Final output layer
        self.output_fc = nn.Linear(query_dim, output_dim)

    def forward(self, text_embedding, image_embedding):
        # Repeat learned queries for batch size
        batch_size = text_embedding.size(0)
        queries = self.learned_queries.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (batch_size, num_queries, query_dim)

        for stage in range(self.num_stages):
            updated_queries = []

            for query in queries.split(1, dim=1):  # Iterate over each query
                query = query.squeeze(1)  # Shape: (batch_size, query_dim)

                # Cross-attend to both modalities
                text_attended = self.text_cross_attention(query, text_embedding)
                image_attended = self.image_cross_attention(query, image_embedding)

                # Fuse the attended outputs
                fused = text_attended + image_attended
                fused = torch.tanh(self.fusion_mlp(fused))  # Fuse and transform

                # Pass through a residual block (akin to RNN update)
                updated_query = self.residual_blocks[stage](fused)
                updated_queries.append(
                    updated_query.unsqueeze(1)
                )  # Append updated query

            # Stack the updated queries back into a single tensor
            queries = torch.cat(
                updated_queries, dim=1
            )  # (batch_size, num_queries, query_dim)

        # Use the updated queries to produce the final output
        output = queries.mean(dim=1)  # Aggregate the queries (mean pooling)
        output = self.output_fc(output)  # Final linear transformation

        return output
