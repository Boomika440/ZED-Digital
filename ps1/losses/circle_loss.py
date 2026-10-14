"""
Circle Loss implementation for Deep Metric Learning.
Reference: Sun et al., 'Circle Loss: A Unified Perspective of Pair Similarity Optimization', CVPR 2020.
Enforces flexible, self-paced convergence by weighting similarities based on optimization status.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """
    Circle Loss for pairwise metric learning.
    Dynamically assigns weights to similarity scores:
      alpha_p = clamp(O_p - s_p, min=0)
      alpha_n = clamp(s_n - O_n, min=0)
    """
    def __init__(self, m: float = 0.25, gamma: float = 64.0):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.o_p = 1.0 + m
        self.o_n = -m
        self.delta_p = 1.0 - m
        self.delta_n = m

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized feature vectors.
            targets: [B] identity labels.
        """
        # Cosine similarity matrix: [B, B]
        similarity_mat = torch.matmul(embeddings, embeddings.t())

        n = embeddings.size(0)
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t()).float()
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t()).float()

        # Remove diagonal self-similarity from positive mask
        eye_mask = torch.eye(n, device=embeddings.device).bool()
        is_pos.masked_fill_(eye_mask, 0)

        # Dynamic weights
        alpha_p = torch.clamp(self.o_p - similarity_mat.detach(), min=0.0)
        alpha_n = torch.clamp(similarity_mat.detach() - self.o_n, min=0.0)

        logit_p = -self.gamma * alpha_p * (similarity_mat - self.delta_p)
        logit_n = self.gamma * alpha_n * (similarity_mat - self.delta_n)

        # Exclude non-positive/negative elements with large negative numbers for logsumexp
        logit_p.masked_fill_(is_pos == 0, -1e9)
        logit_n.masked_fill_(is_neg == 0, -1e9)

        loss_p = torch.logsumexp(logit_p, dim=1)
        loss_n = torch.logsumexp(logit_n, dim=1)

        loss = F.softplus(loss_p + loss_n).mean()
        return loss
