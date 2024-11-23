import torch
import torch.nn as nn
from transformers import AutoTokenizer
# Decoder model for commentary generator
class Decoder(nn.Module):
    def __init__(self, transformer_model="bert-base-uncased", hidden_dim=768):
        super(Decoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)

        self.decoder = nn.Transformer(
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

    def forward(self, fused_features, target_commentary=None):
        if target_commentary is not None:
            tokens = self.tokenizer(target_commentary, return_tensors="pt", padding=True, truncation=True)
            target_embeddings = self.decoder.encoder(**tokens).last_hidden_state

            output = self.decoder(
                src=fused_features,
                tgt=target_embeddings,
                src_mask=None,
                tgt_mask=None
            )
        else:
            start_token = self.tokenizer.cls_token_id
            target_embeddings = torch.zeros((fused_features.size(0), 1, fused_features.size(2)), device=fused_features.device)
            target_embeddings[:, 0, :] = start_token

            output = self.decoder(
                src=fused_features,
                tgt=target_embeddings,
                src_mask=None,
                tgt_mask=None
            )

        logits = self.output_layer(output)

        return logits
