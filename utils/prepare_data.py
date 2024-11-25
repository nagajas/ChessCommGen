import pandas as pd
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load embeddings
with open("principles_embeddings.json", "r") as f:
    principle_embeddings = json.load(f)

# Prepare dataset
df = pd.read_csv("train.csv")  # Assuming `train.csv` is prepared with move context and principles
tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_t5")

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()

# Encode moves and assign principle labels
X, y = [], []
for _, row in df.iterrows():
    move_text = row["History (PGN)"] + " " + row["Move"]
    X.append(encode_text(move_text))

    # Generate multi-hot labels for principles
    labels = [1 if principle in row["Mapped Principles"] else 0 for principle in principle_embeddings.keys()]
    y.append(labels)

X = torch.tensor(X)
y = torch.tensor(y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save classifier
import pickle
with open("chess_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)
