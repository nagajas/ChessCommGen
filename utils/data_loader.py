import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            row['History'], 
            row[['Eval Before', 'Eval After', 'Delta']].values, 
            row['Phase'], 
            row['Commentary']
        )

def get_dataloader(csv_file, batch_size=32, shuffle=True):
    dataset = ChessDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
