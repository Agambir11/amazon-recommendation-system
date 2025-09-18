import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from src.models.bpr_mf import BPRMF, bpr_loss
from src.data.dataset_implicit import PairwiseImplicitDataset

from src.training.metrics import recall_at_k, ndcg_at_k

def build_user_pos(train_df, valid_df):
    user_pos = {}
    for u in train_df["uid"].unique():
        train_items = set(train_df.loc[train_df["uid"] == u, "iid"].tolist())
        val_items = set(valid_df.loc[valid_df["uid"] == u, "iid"].tolist())
        if len(val_items) > 0:
            user_pos[u] = {"train": train_items, "val": val_items}
    return user_pos


def train_bpr(data_dir, emb_dim=64, batch_size=512, epochs=5, lr=1e-3, num_neg=1):
    # Load train/valid
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    valid_df = pd.read_csv(f"{data_dir}/valid.csv")

    n_users = train_df["uid"].max() + 1
    n_items = train_df["iid"].max() + 1

    # dataset + dataloader
    train_ds = PairwiseImplicitDataset(f"{data_dir}/train.csv", n_items, num_neg=num_neg)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,       # ✅ speedup with multiprocessing
        pin_memory=True      # ✅ faster if using GPU
    )

    model = BPRMF(n_users, n_items, emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # build validation ground truth
    user_pos = build_user_pos(train_df, valid_df)

    # training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for u, pos, neg in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            pos_scores = model(u, pos)
            neg_scores = model(
                u.unsqueeze(1).expand(-1, neg.size(1)).reshape(-1),
                neg.reshape(-1)
            ).view(len(u), -1)

            loss = bpr_loss(pos_scores, neg_scores, reg_lambda=1e-4, model=model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        # Evaluate on validation set
        r10 = recall_at_k(model, user_pos, k=10)
        n10 = ndcg_at_k(model, user_pos, k=10)
        print(f"Validation Recall@10 = {r10:.4f}, NDCG@10 = {n10:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="path to preprocessed dataset folder")
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_neg", type=int, default=1)
    args = ap.parse_args()

    train_bpr(
        data_dir=args.data_dir,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_neg=args.num_neg,
    )
