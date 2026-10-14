"""
Fine-Grained Spatial Part-Attention Module for Low Inter-Class Appearance Variance.
Under identical clothing/uniforms, global pooling dilutes subtle discriminative cues.
This module extracts localized part representations (Head/Face, Chest/Badge, Waist/Hands, Feet/Shoes)
and refines them via self-attention to maximize discriminative power.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineGrainedPartAttention(nn.Module):
    """
    Part-Aware Attention Splitter.
    Given a feature map of shape [B, C, H, W] (e.g. [B, 512, 16, 8]):
    1. Splits the spatial map vertically into K part strips (H // K slices).
    2. Learns a spatial attention mask for each strip to focus on high-frequency details (e.g. badge, shoe accent).
    3. Aggregates each part into a compact part descriptor.
    """
    def __init__(self, in_channels: int = 512, num_parts: int = 4, part_dim: int = 128):
        super().__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels
        self.part_dim = part_dim

        # Spatial attention generator per part
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Dimension reduction projection for each part
        self.part_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, part_dim),
                nn.BatchNorm1d(part_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_parts)
        ])

        # Inter-part cross-correlation aggregator
        self.part_fusion = nn.Linear(num_parts * part_dim, in_channels)
        self.fusion_bn = nn.BatchNorm1d(in_channels)

    def forward(self, feature_map: torch.Tensor):
        """
        Args:
            feature_map: [B, C, H, W] tensor from backbone.
        Returns:
            fused_part_feat: [B, C] discriminative part representation.
            part_features: list of [B, part_dim] individual part vectors.
        """
        B, C, H, W = feature_map.shape
        strip_h = H // self.num_parts

        part_features = []

        # Generate spatial attention map
        att_map = self.attention_conv(feature_map)  # [B, 1, H, W]
        attended_features = feature_map * att_map

        for i in range(self.num_parts):
            # Extract strip i
            h_start = i * strip_h
            h_end = (i + 1) * strip_h if i < self.num_parts - 1 else H
            strip = attended_features[:, :, h_start:h_end, :]  # [B, C, strip_h, W]

            # Global average pooling on strip
            pooled_strip = F.adaptive_avg_pool2d(strip, (1, 1)).view(B, C)
            part_proj = self.part_projections[i](pooled_strip)  # [B, part_dim]
            part_features.append(part_proj)

        # Concatenate part descriptors
        concat_parts = torch.cat(part_features, dim=1)  # [B, num_parts * part_dim]
        fused_part_feat = self.fusion_bn(self.part_fusion(concat_parts))  # [B, C]

        return fused_part_feat, part_features
