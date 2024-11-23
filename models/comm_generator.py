import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

#Game aware commentary generator model
class ChessCommentaryModel(nn.Module):
    def __init__(self, transformer_model="bert-base-uncased", hidden_dim=768, num_phases=3):
        super(ChessCommentaryModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer_encoder = AutoModel.from_pretrained(transformer_model)

        self.eval_embedding = nn.Linear(1, hidden_dim)  
        self.phase_embedding = nn.Embedding(num_phases, hidden_dim) 

        self.fusion_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        self.transformer_decoder = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, self.tokenizer.vocab_size)
    
    def forward(self, move_history, eval_features, phase, target_commentary=None):
        history_tokens = self.tokenizer(move_history, return_tensors="pt", padding=True, truncation=True)
        history_embeddings = self.transformer_encoder(**history_tokens).last_hidden_state

        eval_features = eval_features.unsqueeze(-1).float()  
        eval_embeddings = self.eval_embedding(eval_features) 

        phase_embeddings = self.phase_embedding(phase) 

        combined_features = torch.cat([history_embeddings, eval_embeddings, phase_embeddings.unsqueeze(1)], dim=1)

        fused_features, _ = self.fusion_layer(combined_features, combined_features, combined_features)

        if target_commentary is not None:
            target_tokens = self.tokenizer(target_commentary, return_tensors="pt", padding=True, truncation=True)
            target_embeddings = self.transformer_encoder(**target_tokens).last_hidden_state

            output = self.transformer_decoder(
                src=fused_features,
                tgt=target_embeddings,
                src_mask=None,
                tgt_mask=None
            )
        else:

            start_token = self.tokenizer.cls_token_id
            target_embeddings = torch.zeros((fused_features.size(0), 1, fused_features.size(2)), device=fused_features.device)
            target_embeddings[:, 0, :] = self.tokenizer.convert_ids_to_tokens(start_token)

            output = self.transformer_decoder(
                src=fused_features,
                tgt=target_embeddings,
                src_mask=None,
                tgt_mask=None
            )

        logits = self.output_layer(output)

        return logits
