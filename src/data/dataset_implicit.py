import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class PairwiseImplicitDataset(Dataset):
    """
    For BPR training: (user, pos_item, neg_item).
    """
    def __init__(self, csv_file: str, num_items: int, num_neg: int = 1):
        df = pd.read_csv(csv_file)
        # expected columns: uid, iid
        self.ui = df[['uid', 'iid']].values
        self.num_items = num_items
        self.num_neg = num_neg

        # user -> set(pos_items) mapping
        self.user_pos = {}
        for u, i in self.ui:
            self.user_pos.setdefault(u, set()).add(i)

    def __len__(self):
        return len(self.ui)
    

    def __getitem__(self, idx):
        
        u, pos = self.ui[idx]
        # sample negatives in one go
        negs = np.random.randint(0, self.num_items, size=self.num_neg)
        # ensure not accidentally sampling a positive
        while any(n in self.user_pos[u] for n in negs):
            negs = np.random.randint(0, self.num_items, size=self.num_neg)
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(negs, dtype=torch.long),)

