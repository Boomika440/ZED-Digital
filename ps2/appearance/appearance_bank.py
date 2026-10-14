"""
Multi-Template Appearance Bank with Occlusion Quality-Filtering.
Maintains a dynamic gallery of clean, uncorrupted historical appearance descriptors
for each active and lost track, preventing template corruption during dense crowd mutual occlusions.
"""

from typing import List, Dict, Optional
import numpy as np


class AppearanceRecord:
    """Stores a single feature template with quality score."""
    def __init__(self, feature: np.ndarray, score: float, overlap_ratio: float, frame_id: int):
        self.feature = feature / max(float(np.linalg.norm(feature)), 1e-8)
        self.score = score
        self.overlap_ratio = overlap_ratio
        self.frame_id = frame_id
        # Quality index: higher when detection is sharp and unoccluded
        self.quality = score * max(0.0, 1.0 - overlap_ratio)


class TrackAppearanceBank:
    """
    Quality-weighted appearance gallery per track.
    Ensures that occluded frames do NOT corrupt the identity representation.
    """
    def __init__(self, max_templates: int = 15, min_quality_thresh: float = 0.40):
        self.max_templates = max_templates
        self.min_quality_thresh = min_quality_thresh
        self.records: List[AppearanceRecord] = []

    def add(self, feature: np.ndarray, score: float, overlap_ratio: float, frame_id: int):
        """Adds template only if quality is above threshold."""
        rec = AppearanceRecord(feature, score, overlap_ratio, frame_id)
        if rec.quality >= self.min_quality_thresh:
            self.records.append(rec)
            if len(self.records) > self.max_templates:
                # Remove oldest or lowest quality record
                self.records.pop(0)

    def get_consensus_feature(self) -> Optional[np.ndarray]:
        """Computes quality-weighted average appearance vector."""
        if len(self.records) == 0:
            return None

        weights = np.array([r.quality for r in self.records], dtype=np.float32)
        total_w = np.sum(weights)
        if total_w < 1e-6:
            return self.records[-1].feature.copy()

        weights /= total_w
        weighted_feat = np.zeros_like(self.records[0].feature)
        for r, w in zip(self.records, weights):
            weighted_feat += w * r.feature

        norm = np.linalg.norm(weighted_feat)
        return (weighted_feat / max(norm, 1e-8)).astype(np.float32)

    def min_distance(self, query_feature: np.ndarray) -> float:
        """Computes minimum cosine distance across stored clean templates."""
        if len(self.records) == 0:
            return 1.0

        q = query_feature / max(float(np.linalg.norm(query_feature)), 1e-8)
        feats = np.array([r.feature for r in self.records], dtype=np.float32)
        sims = np.dot(feats, q)
        max_sim = float(np.max(sims))
        return float(np.clip(1.0 - max_sim, 0.0, 2.0))
