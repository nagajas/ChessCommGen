import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, hidden_dim=768):
        super(FusionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

    def forward(self, move_history_embeddings, eval_embeddings, phase_embeddings):
        combined_features = torch.cat([move_history_embeddings, eval_embeddings, phase_embeddings.unsqueeze(1)], dim=1)

        fused_features, _ = self.attention(combined_features, combined_features, combined_features)

        return fused_features
