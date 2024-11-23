import torch.nn as nn
from models.feature_extractor import FeatureExtractor
from models.fusion_module import FusionModule
from models.decoder import Decoder

class ChessCommentaryModel(nn.Module):
    def __init__(self, transformer_model="bert-base-uncased", hidden_dim=768, num_phases=3):
        super(ChessCommentaryModel, self).__init__()
        self.feature_extractor = FeatureExtractor(transformer_model, hidden_dim, num_phases)
        self.fusion_module = FusionModule(hidden_dim)
        self.decoder = Decoder(transformer_model, hidden_dim)

    def forward(self, move_history, eval_features, phase, target_commentary=None):
        move_history_embeddings, eval_embeddings, phase_embeddings = self.feature_extractor(move_history, eval_features, phase)
        fused_features = self.fusion_module(move_history_embeddings, eval_embeddings, phase_embeddings)
        logits = self.decoder(fused_features, target_commentary)

        return logits
