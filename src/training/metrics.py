import numpy as np
import torch

def recall_at_k(model, user_pos, k=10, device="cpu"):
    """
    Compute Recall@K across users.
    user_pos: dict {user: {"train": set(), "val": set()}}
    """
    model.eval()
    recalls = []
    with torch.no_grad():
        for u, items in user_pos.items():
            u_tensor = torch.tensor([u], device=device)
            scores = model.score_all_items(u_tensor).cpu().numpy().flatten()

            # mask training items
            scores[list(items["train"])] = -np.inf

            # top-K
            topk = np.argpartition(scores, -k)[-k:]
            topk = topk[np.argsort(-scores[topk])]

            hits = len(items["val"] & set(topk))
            recalls.append(hits / len(items["val"]))
    return float(np.mean(recalls))


def ndcg_at_k(model, user_pos, k=10, device="cpu"):
    """
    Compute NDCG@K across users.
    """
    model.eval()
    ndcgs = []
    with torch.no_grad():
        for u, items in user_pos.items():
            u_tensor = torch.tensor([u], device=device)
            scores = model.score_all_items(u_tensor).cpu().numpy().flatten()

            scores[list(items["train"])] = -np.inf

            topk = np.argpartition(scores, -k)[-k:]
            topk = topk[np.argsort(-scores[topk])]

            dcg = 0.0
            for rank, i in enumerate(topk, start=1):
                if i in items["val"]:
                    dcg += 1.0 / np.log2(rank + 1)
            idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(k, len(items["val"])) + 1))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))
