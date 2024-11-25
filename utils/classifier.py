import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

if __name__ == "__main__":
    X, y = torch.load("encoded_data.pt")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    with open("chess_classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)
