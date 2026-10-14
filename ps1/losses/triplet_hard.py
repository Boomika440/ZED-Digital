"""
Batch-Hard Triplet Loss with Hard-Negative Mining.
Reference: Hermans et al., 'In Defense of the Triplet Loss for Person Re-Identification', 2017.

In low inter-class variance, trivial negative pairs (which easily satisfy the margin)
contribute zero gradient. Batch-Hard mining forces the network to find the absolute
closest negative (someone wearing identical uniform) and pushes them outside the margin boundary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Euclidean distance matrix between x and y.
    ||x - y||^2 = ||x||^2 - 2<x, y> + ||y||^2
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # For numerical stability
    return dist


class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Loss.
    For each anchor a_i in batch:
      hardest_positive = max_{p in P_i} D(a_i, p)
      hardest_negative = min_{n in N_i} D(a_i, n)
      loss_i = relu(hardest_positive - hardest_negative + margin)
    """
    def __init__(self, margin: float = 0.3, smooth: bool = False):
        super().__init__()
        self.margin = margin
        self.smooth = smooth
        if not smooth:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized feature vectors.
            targets: [B] identity labels.
        """
        n = embeddings.size(0)
        dist_mat = euclidean_dist(embeddings, embeddings)

        # Boolean masks for positive and negative pairs
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())

        # For each anchor, extract the hardest positive (max distance among same class)
        dist_ap = []
        for i in range(n):
            pos_dists = dist_mat[i][is_pos[i]]
            if len(pos_dists) > 0:
                dist_ap.append(pos_dists.max().unsqueeze(0))
            else:
                dist_ap.append(torch.tensor([0.0], device=embeddings.device))
        dist_ap = torch.cat(dist_ap)

        # For each anchor, extract the hardest negative (min distance among different classes)
        dist_an = []
        for i in range(n):
            neg_dists = dist_mat[i][is_neg[i]]
            if len(neg_dists) > 0:
                dist_an.append(neg_dists.min().unsqueeze(0))
            else:
                dist_an.append(torch.tensor([0.0], device=embeddings.device))
        dist_an = torch.cat(dist_an)

        # Compute margin ranking loss
        y = torch.ones_like(dist_an)
        if self.smooth:
            # Soft-margin triplet formulation: log(1 + exp(d_ap - d_an))
            diff = dist_ap - dist_an
            loss = F.softplus(diff).mean()
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
