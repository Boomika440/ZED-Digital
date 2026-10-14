"""
Unified End-to-End Open-Set Person Re-Identification Network.
Fuses compact backbone, fine-grained part attention, and BNNeck embedding formulation.
Designed for real-time edge execution with single-backbone efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .backbone import build_backbone
    from .part_attention import FineGrainedPartAttention
    from .embedding_head import BNNeckHead
except (ImportError, ValueError):
    try:
        from models.backbone import build_backbone
        from models.part_attention import FineGrainedPartAttention
        from models.embedding_head import BNNeckHead
    except ImportError:
        from ps1.models.backbone import build_backbone
        from ps1.models.part_attention import FineGrainedPartAttention
        from ps1.models.embedding_head import BNNeckHead


class OpenSetReIDNet(nn.Module):
    """
    Open-Set Person Re-ID Architecture under Low Inter-Class Variance.
    """
    def __init__(
        self,
        backbone_name: str = "osnet_compact",
        embedding_dim: int = 512,
        use_part_attention: bool = True,
        num_parts: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.use_part_attention = use_part_attention
        self.embedding_dim = embedding_dim

        # 1. Single Compact Backbone
        self.backbone = build_backbone(backbone_name=backbone_name, feature_dim=embedding_dim)

        # 2. Fine-Grained Part Attention
        if self.use_part_attention:
            self.part_attention = FineGrainedPartAttention(
                in_channels=embedding_dim,
                num_parts=num_parts,
                part_dim=embedding_dim // num_parts
            )
            # BN neck for fused features
            self.fusion_bn = nn.BatchNorm1d(embedding_dim)
        else:
            self.part_attention = None

        # 3. BNNeck Embedding Head
        self.head = BNNeckHead(
            in_features=embedding_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast forward pass for real-time inference / evaluation.
        Returns strictly L2-normalized embedding vector of shape [B, embedding_dim].
        """
        feature_map = self.backbone(x)
        feat_metric, _ = self.head(feature_map)

        if self.use_part_attention and self.part_attention is not None:
            fused_parts, _ = self.part_attention(feature_map)
            fused_metric = F.normalize(fused_parts, p=2, dim=1)
            # Combine global context with localized discriminative cues
            final_embedding = F.normalize(feat_metric + 0.6 * fused_metric, p=2, dim=1)
            return final_embedding

        return feat_metric

    def forward(self, x: torch.Tensor):
        """
        Full training forward pass returning:
        - final_metric_embedding: [B, D] L2 normalized for Triplet / Cosine loss
        - feat_class: [B, D] for ArcFace classification
        - part_features: list of part tensors for auxiliary fine-grained loss (if enabled)
        """
        feature_map = self.backbone(x)
        feat_metric, feat_class = self.head(feature_map)

        part_features = []
        if self.use_part_attention and self.part_attention is not None:
            fused_parts, part_features = self.part_attention(feature_map)
            fused_metric = F.normalize(fused_parts, p=2, dim=1)
            final_metric = F.normalize(feat_metric + 0.6 * fused_metric, p=2, dim=1)
            # For classification, fuse pre-BN representations and pass through BN
            fused_class = self.fusion_bn(feat_class + 0.6 * fused_parts)
            return final_metric, fused_class, part_features

        return feat_metric, feat_class, part_features


def build_model(config: dict) -> OpenSetReIDNet:
    model_cfg = config.get("model", {})
    return OpenSetReIDNet(
        backbone_name=model_cfg.get("backbone", "osnet_compact"),
        embedding_dim=model_cfg.get("embedding_dim", 512),
        use_part_attention=model_cfg.get("use_part_attention", True),
        num_parts=model_cfg.get("num_parts", 4),
        dropout_rate=model_cfg.get("dropout_rate", 0.1)
    )
