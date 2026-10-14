"""
Embedding Head and BNNeck for Metric Learning.
Decouples metric distance optimization (pre-BN) from angular classification hyperplane (post-BN).
Includes Generalized-Mean (GeM) Pooling and L2 Normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (Radenovic et al., TPAMI 2018).
    f(x) = (1/|x| * sum(x^p))^(1/p)
    When p=1 -> AvgPool; when p->infinity -> MaxPool.
    Learns to emphasize prominent discriminative activations.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_val = self.p.clamp(min=1.0)
        x_clamped = x.clamp(min=self.eps)
        pooled = F.adaptive_avg_pool2d(x_clamped.pow(p_val), (1, 1)).pow(1.0 / p_val)
        return pooled.squeeze(-1).squeeze(-1)


class BNNeckHead(nn.Module):
    """
    BNNeck embedding architecture.
    Provides two representations:
    1. feat_metric (pre-BN): Used for Triplet Loss & Cosine Similarity search.
    2. feat_class (post-BN): Used for Angular Margin / ArcFace classification.
    """
    def __init__(self, in_features: int = 512, embedding_dim: int = 512, dropout_rate: float = 0.1):
        super().__init__()
        self.gem_pool = GeMPooling(p=3.0)

        # Optional linear projection if embedding_dim != in_features
        if embedding_dim != in_features:
            self.fc = nn.Linear(in_features, embedding_dim, bias=False)
        else:
            self.fc = nn.Identity()

        # Batch Normalization Neck
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # No learnable bias in BN neck

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, feature_map: torch.Tensor):
        """
        Args:
            feature_map: [B, C, H, W]
        Returns:
            feat_metric: [B, embedding_dim] L2-normalized feature for metric retrieval
            feat_class: [B, embedding_dim] Post-BN feature for ArcFace classification
        """
        pooled = self.gem_pool(feature_map)  # [B, C]
        proj = self.fc(pooled)              # [B, embedding_dim]

        feat_class = self.bottleneck(proj)  # Post-BN for classification
        feat_class = self.dropout(feat_class)

        # Pre-BN normalized for metric distance
        feat_metric = F.normalize(proj, p=2, dim=1)

        return feat_metric, feat_class
