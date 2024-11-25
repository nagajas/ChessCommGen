from transformers import T5Tokenizer, T5Model
import torch

model = T5Model.from_pretrained("./fine_tuned_t5")
tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_t5")

with open("principles.txt", "r") as f:
    principles = [line.strip() for line in f if line.strip()]

embeddings = {}
for principle in principles:
    inputs = tokenizer(principle, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
        embeddings[principle] = outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()

import json
with open("principles_embeddings.json", "w") as f:
    json.dump(embeddings, f)
