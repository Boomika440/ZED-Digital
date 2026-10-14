"""
Margin-Based Decision-Boundary Loss Formulations.
Includes Sub-Center ArcFace, Standard ArcFace, and CosFace.

Under low inter-class appearance variance (identical uniform clothing),
standard softmax yields weak boundaries that easily admit unenrolled impostors.
Sub-Center ArcFace enforces an angular margin 'm' on a hypersphere, while allowing
intra-class multi-modality (e.g. front vs rear camera angles) via multiple sub-centers per identity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCenterArcFace(nn.Module):
    """
    Sub-Center ArcFace (Deng et al., ECCV 2020).
    Assigns K sub-centers per identity. Each sample is supervised against its nearest sub-center.
    Formula:
        cos(theta + m) for target class's nearest sub-center
        s * cos(theta) for other classes
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        sub_centers: int = 3,
        scale: float = 30.0,
        margin: float = 0.35,
        easy_margin: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.sub_centers = sub_centers
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Weight shape: [num_classes * sub_centers, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, in_features] feature vectors (post-BN or normalized).
            labels: [B] ground truth class IDs.
        Returns:
            logits: [B, num_classes] scaled angular logits ready for CrossEntropyLoss.
        """
        # Normalize weights and features to unit hypersphere
        norm_weights = F.normalize(self.weight, p=2, dim=1)
        norm_features = F.normalize(features, p=2, dim=1)

        # Cosine similarity to all sub-centers: [B, num_classes * sub_centers]
        cosine_all = F.linear(norm_features, norm_weights)

        # Reshape to [B, num_classes, sub_centers] and pick maximum similarity per class
        cosine_all = cosine_all.view(-1, self.num_classes, self.sub_centers)
        cosine, _ = torch.max(cosine_all, dim=2)  # [B, num_classes]

        # Calculate cos(theta + m) using trigonometric identity
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for ground truth labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply angular margin penalty to target class only
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        return output


class StandardArcFace(nn.Module):
    """
    Standard ArcFace (Additive Angular Margin Loss, Deng et al., CVPR 2019).
    """
    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        norm_weights = F.normalize(self.weight, p=2, dim=1)
        norm_features = F.normalize(features, p=2, dim=1)
        cosine = F.linear(norm_features, norm_weights).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        return output


class LinearSoftmaxClassifier(nn.Module):
    """
    Standard Linear Classifier without margin (Used for Baseline Ablation).
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes, bias=False)

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        return self.linear(features)
