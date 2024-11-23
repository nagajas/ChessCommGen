import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class FeatureExtractor(nn.Module):
    def __init__(self, transformer_model="bert-base-uncased", hidden_dim=768, num_phases=3):
        super(FeatureExtractor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer_encoder = AutoModel.from_pretrained(transformer_model)

        self.eval_embedding = nn.Linear(1, hidden_dim) 
        self.phase_embedding = nn.Embedding(num_phases, hidden_dim)  

    def forward(self, move_history, eval_features, phase):
        tokens = self.tokenizer(move_history, return_tensors="pt", padding=True, truncation=True)
        move_history_embeddings = self.transformer_encoder(**tokens).last_hidden_state

        eval_features = eval_features.unsqueeze(-1).float() 
        eval_embeddings = self.eval_embedding(eval_features)

        phase_embeddings = self.phase_embedding(phase)  

        return move_history_embeddings, eval_embeddings, phase_embeddings
