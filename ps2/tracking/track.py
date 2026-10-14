"""
Track Lifecycle and State Management for Dense Crowd Tracking.
Manages state transitions: Tentative -> Confirmed -> Lost -> Deleted.
Includes:
- Occlusion-aware appearance template protection.
- Quality-weighted feature Exponential Moving Average (EMA).
- Trajectory trail recording and spatial motion history.
"""

from enum import Enum
from typing import Optional, List
import numpy as np

try:
    from .bboxes import xyah_to_xyxy, xyxy_to_xyah
except (ImportError, ValueError):
    try:
        from tracking.bboxes import xyah_to_xyxy, xyxy_to_xyah
    except ImportError:
        from ps2.tracking.bboxes import xyah_to_xyxy, xyxy_to_xyah


class TrackState(Enum):
    Tentative = 1   # Newly initiated, awaiting confirmation (min_hits_to_confirm)
    Confirmed = 2   # Actively tracked, high-confidence confirmed identity
    Lost = 3        # Temporarily lost/occluded, held in Re-Association Bank (up to max_lost_time)
    Deleted = 4     # Inactive beyond max_lost_time, marked for permanent purging


class Track:
    """
    Individual Track entity with lifecycle management, appearance bank, and occlusion state.
    """
    _count = 0

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        score: float,
        feature: Optional[np.ndarray] = None,
        min_hits_to_confirm: int = 3,
        max_lost_time: int = 60,
        ema_alpha: float = 0.90
    ):
        Track._count += 1
        self.track_id = Track._count
        self.state = TrackState.Tentative

        self.mean = mean.copy().astype(np.float32)
        self.covariance = covariance.copy().astype(np.float32)

        self.score = float(score)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.hit_streak = 1

        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_lost_time = max_lost_time
        self.ema_alpha = ema_alpha

        # Occlusion flags
        self.is_occluded = False
        self.overlap_ratio = 0.0

        # Appearance features
        self.features: List[np.ndarray] = []
        self.smooth_feature: Optional[np.ndarray] = None
        self.last_clean_feature: Optional[np.ndarray] = None

        if feature is not None:
            norm_feat = feature / max(float(np.linalg.norm(feature)), 1e-8)
            self.features.append(norm_feat)
            self.smooth_feature = norm_feat.copy()
            self.last_clean_feature = norm_feat.copy()

        # Trajectory history: list of [xc, yc]
        self.trajectory: List[np.ndarray] = []
        self._record_center()

    @classmethod
    def reset_counter(cls):
        """Resets ID counter for new evaluation sequences."""
        cls._count = 0

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.Lost

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted

    @property
    def to_xyxy(self) -> np.ndarray:
        """Returns current bounding box as [x1, y1, x2, y2]."""
        return xyah_to_xyxy(self.mean[:4])

    @property
    def to_xyah(self) -> np.ndarray:
        """Returns [xc, yc, a, h]."""
        return self.mean[:4].copy()

    def _record_center(self):
        xc, yc = float(self.mean[0]), float(self.mean[1])
        self.trajectory.append(np.array([xc, yc], dtype=np.float32))
        if len(self.trajectory) > 60:
            self.trajectory.pop(0)

    def mark_occlusion(self, is_occluded: bool, overlap_ratio: float = 0.0):
        """Updates occlusion status."""
        self.is_occluded = is_occluded
        self.overlap_ratio = float(overlap_ratio)

    def predict(self, kf):
        """Advances track state using Kalman Filter."""
        self.mean, self.covariance = kf.predict(
            self.mean,
            self.covariance,
            is_occluded=self.is_occluded or self.is_lost
        )
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self._record_center()

    def update(
        self,
        kf,
        measurement_xyah: np.ndarray,
        score: float,
        feature: Optional[np.ndarray] = None
    ):
        """Updates track with matched detection."""
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement_xyah)
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        self.hit_streak += 1

        # State transition: Tentative -> Confirmed
        if self.state == TrackState.Tentative and self.hits >= self.min_hits_to_confirm:
            self.state = TrackState.Confirmed
        elif self.state == TrackState.Lost:
            # Re-associated! Return to Confirmed state
            self.state = TrackState.Confirmed

        # Appearance update with Occlusion Quality-Weighting
        if feature is not None:
            norm_feat = feature / max(float(np.linalg.norm(feature)), 1e-8)

            # CRITICAL RULE: If track is occluded, DO NOT pollute template with overlapping clothing!
            if not self.is_occluded and self.overlap_ratio < 0.25:
                if self.smooth_feature is None:
                    self.smooth_feature = norm_feat.copy()
                else:
                    self.smooth_feature = self.ema_alpha * self.smooth_feature + (1.0 - self.ema_alpha) * norm_feat
                    self.smooth_feature /= max(float(np.linalg.norm(self.smooth_feature)), 1e-8)

                self.features.append(norm_feat)
                if len(self.features) > 20:
                    self.features.pop(0)
                self.last_clean_feature = self.smooth_feature.copy()

        self._record_center()

    def mark_missed(self):
        """Handles missed detection in current frame."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Confirmed:
            if self.time_since_update > 1:
                self.state = TrackState.Lost

        if self.time_since_update > self.max_lost_time:
            self.state = TrackState.Deleted
