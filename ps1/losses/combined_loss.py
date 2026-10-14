"""
Unified Loss Manager combining Margin Loss, Batch-Hard Triplet, and Fine-Grained Part Regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .arcface import SubCenterArcFace, StandardArcFace, LinearSoftmaxClassifier
    from .triplet_hard import BatchHardTripletLoss
    from .circle_loss import CircleLoss
except (ImportError, ValueError):
    try:
        from losses.arcface import SubCenterArcFace, StandardArcFace, LinearSoftmaxClassifier
        from losses.triplet_hard import BatchHardTripletLoss
        from losses.circle_loss import CircleLoss
    except ImportError:
        from ps1.losses.arcface import SubCenterArcFace, StandardArcFace, LinearSoftmaxClassifier
        from ps1.losses.triplet_hard import BatchHardTripletLoss
        from ps1.losses.circle_loss import CircleLoss


class CombinedReIDLoss(nn.Module):
    """
    Unified loss framework:
      L_total = lambda_margin * L_margin + lambda_triplet * L_triplet + lambda_part * L_part
    """
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 512,
        margin_loss_type: str = "subcenter_arcface",
        margin: float = 0.35,
        scale: float = 30.0,
        sub_centers: int = 3,
        use_hard_triplet: bool = True,
        triplet_margin: float = 0.3,
        lambda_margin: float = 1.0,
        lambda_triplet: float = 1.0,
        lambda_part: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.margin_loss_type = margin_loss_type
        self.use_hard_triplet = use_hard_triplet
        self.lambda_margin = lambda_margin
        self.lambda_triplet = lambda_triplet
        self.lambda_part = lambda_part

        # 1. Margin / Classification Head
        if margin_loss_type == "subcenter_arcface":
            self.classifier = SubCenterArcFace(
                in_features=feature_dim,
                num_classes=num_classes,
                sub_centers=sub_centers,
                scale=scale,
                margin=margin
            )
        elif margin_loss_type == "arcface":
            self.classifier = StandardArcFace(
                in_features=feature_dim,
                num_classes=num_classes,
                scale=scale,
                margin=margin
            )
        elif margin_loss_type == "softmax":
            self.classifier = LinearSoftmaxClassifier(
                in_features=feature_dim,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown margin loss type: {margin_loss_type}")

        self.ce_loss = nn.CrossEntropyLoss()

        # 2. Metric Learning Head
        if use_hard_triplet:
            self.triplet_loss = BatchHardTripletLoss(margin=triplet_margin)
        else:
            self.triplet_loss = None

    def forward(
        self,
        metric_feats: torch.Tensor,
        class_feats: torch.Tensor,
        part_feats: list,
        targets: torch.Tensor
    ):
        """
        Args:
            metric_feats: [B, D] L2 normalized pre-BN embeddings.
            class_feats: [B, D] Post-BN embeddings.
            part_feats: list of [B, part_dim] tensors.
            targets: [B] Identity labels.
        Returns:
            total_loss: scalar tensor.
            loss_dict: dictionary of float values for monitoring.
        """
        loss_dict = {}

        # 1. Classification / Margin Loss
        logits = self.classifier(class_feats, targets)
        loss_margin = self.ce_loss(logits, targets)
        loss_dict["loss_margin"] = loss_margin.item()
        total_loss = self.lambda_margin * loss_margin

        # 2. Batch-Hard Triplet Loss
        if self.use_hard_triplet and self.triplet_loss is not None:
            loss_triplet = self.triplet_loss(metric_feats, targets)
            loss_dict["loss_triplet"] = loss_triplet.item()
            total_loss = total_loss + self.lambda_triplet * loss_triplet

        # 3. Fine-Grained Part Regularization
        # Computes batch-hard triplet on each localized part to ensure each part is independently discriminative
        if part_feats is not None and len(part_feats) > 0 and self.lambda_part > 0 and self.use_hard_triplet:
            part_loss = 0.0
            for p_idx, p_feat in enumerate(part_feats):
                norm_p = F.normalize(p_feat, p=2, dim=1)
                part_loss += self.triplet_loss(norm_p, targets)
            part_loss = part_loss / len(part_feats)
            loss_dict["loss_part"] = part_loss.item()
            total_loss = total_loss + self.lambda_part * part_loss

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
