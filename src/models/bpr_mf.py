import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        """Dot product score for user u and item i"""
        return (self.user_emb(u) * self.item_emb(i)).sum(-1)

    def score_all_items(self, u):
        """Scores of user u against all items (for ranking/eval)."""
        return self.user_emb(u) @ self.item_emb.weight.T


def bpr_loss(pos_scores, neg_scores, reg_lambda=1e-4, model=None):
    """
    pos_scores: [B]
    neg_scores: [B, K]
    """
    diff = pos_scores.unsqueeze(1) - neg_scores
    loss = -F.logsigmoid(diff).mean()

    if reg_lambda and model:
        loss += reg_lambda * (
            model.user_emb.weight.norm(2).pow(2) +
            model.item_emb.weight.norm(2).pow(2)
        ) / 2.0

    return loss
