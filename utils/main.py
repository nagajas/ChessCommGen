import torch
from models.full_model import ChessCommentaryModel
from utils.data_loader import get_dataloader

transformer_model = "bert-base-uncased"
hidden_dim = 768
num_phases = 3
batch_size = 32
epochs = 5

train_loader = get_dataloader("/home/hari/Desktop/Chess commentary/LoveDaleNLP/data/train.csv", batch_size=batch_size, shuffle=True)
val_loader = get_dataloader("/home/hari/Desktop/Chess commentary/LoveDaleNLP/data/val.csv", batch_size=batch_size, shuffle=False)

model = ChessCommentaryModel(transformer_model, hidden_dim, num_phases)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for move_history, eval_features, phase, commentary in train_loader:
        logits = model(move_history, eval_features, phase, commentary)
        loss = criterion(logits.view(-1, logits.size(-1)), commentary.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            pass  
