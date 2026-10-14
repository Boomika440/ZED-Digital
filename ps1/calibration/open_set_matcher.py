"""
Open-Set Matching Engine with Unknown Identity Rejection.
Supports:
1. Dynamic Gallery Registration (No assumption of fixed identity count).
2. Extreme Value Theory (EVT) and S-Norm Calibrated Score Computation.
3. Dual-Criteria Decision Boundary: Thresholding + Ambiguity Margin Rejection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    from .evt_weibull import EVTWeibullCalibrator
    from .score_normalizer import SNormNormalizer, PlattCalibrator
except (ImportError, ValueError):
    try:
        from calibration.evt_weibull import EVTWeibullCalibrator
        from calibration.score_normalizer import SNormNormalizer, PlattCalibrator
    except ImportError:
        from ps1.calibration.evt_weibull import EVTWeibullCalibrator
        from ps1.calibration.score_normalizer import SNormNormalizer, PlattCalibrator


class OpenSetReIDMatcher:
    """
    Open-Set Matcher that verifies returning individuals against a dynamic gallery,
    minimizing False-Accept Rate (FAR) under low inter-class variance.
    """
    def __init__(
        self,
        calibration_method: str = "evt_weibull",
        threshold: float = 0.72,
        ambiguity_margin: float = 0.08,
        evt_tail_size: int = 20
    ):
        self.calibration_method = calibration_method
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin

        # Dynamic gallery storage (supports adding/removing identities dynamically)
        self.gallery_features: List[np.ndarray] = []
        self.gallery_pids: List[int] = []
        self.gallery_cams: List[int] = []

        # Calibrators
        self.evt_calibrator = EVTWeibullCalibrator(tail_size=evt_tail_size)
        self.snorm_normalizer = SNormNormalizer()
        self.platt_calibrator = PlattCalibrator()

        self.is_calibrated = False

    def enroll_gallery(self, features: np.ndarray, pids: np.ndarray, camids: Optional[np.ndarray] = None):
        """
        Enrolls identities into the gallery database.
        Accepts arbitrary batch sizes and dynamically expands.
        """
        assert features.shape[0] == len(pids), "Feature count must match PID count"
        # Normalize features to unit length
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norm_features = features / np.clip(norms, 1e-8, None)

        self.gallery_features = list(norm_features)
        self.gallery_pids = list(pids)
        self.gallery_cams = list(camids) if camids is not None else [1] * len(pids)

    def add_identity(self, feature: np.ndarray, pid: int, camid: int = 1):
        """
        Dynamically enlists an individual without fixed gallery constraints.
        """
        norm_feat = feature / max(float(np.linalg.norm(feature)), 1e-8)
        self.gallery_features.append(norm_feat)
        self.gallery_pids.append(pid)
        self.gallery_cams.append(camid)

    def calibrate(self):
        """
        Fits calibration statistics against current gallery distribution.
        """
        if len(self.gallery_features) < 2:
            return

        g_feats = np.array(self.gallery_features)
        g_pids = np.array(self.gallery_pids)

        if self.calibration_method == "evt_weibull":
            self.evt_calibrator.fit(g_feats, g_pids)
        elif self.calibration_method == "s_norm":
            self.snorm_normalizer.fit_cohort(g_feats)

        self.is_calibrated = True

    def match_single(self, query_feature: np.ndarray) -> Dict:
        """
        Matches a single query embedding against the dynamic gallery.
        Returns:
            {
               "is_accepted": bool,
               "matched_pid": int or None (-1 if rejected),
               "confidence": float,
               "top1_score": float,
               "top2_score": float,
               "rejection_reason": str or None
            }
        """
        assert len(self.gallery_features) > 0, "Gallery is empty!"

        q = query_feature / max(float(np.linalg.norm(query_feature)), 1e-8)
        G = np.array(self.gallery_features)  # [N, D]

        # 1. Raw Cosine Similarity
        raw_sims = np.dot(G, q)  # [N]

        # Rank matches
        sorted_indices = np.argsort(-raw_sims)
        top1_idx = sorted_indices[0]
        top1_raw = float(raw_sims[top1_idx])
        top1_pid = int(self.gallery_pids[top1_idx])

        top2_raw = float(raw_sims[sorted_indices[1]]) if len(sorted_indices) > 1 else -1.0
        margin = top1_raw - top2_raw if top2_raw != -1.0 else 1.0

        # 2. Score Calibration into Confidence
        if self.calibration_method == "evt_weibull" and self.is_calibrated:
            top1_dist = float(np.clip(1.0 - top1_raw, 0.0, 2.0))
            confidence = self.evt_calibrator.compute_match_probability(top1_dist, top1_pid)
        elif self.calibration_method == "s_norm" and self.is_calibrated:
            s_norm_scores = self.snorm_normalizer.normalize(q[np.newaxis, :], G, raw_sims[np.newaxis, :])
            confidence = float(s_norm_scores[0, top1_idx])
        else:
            confidence = top1_raw

        # 3. Decision Boundary: Dual-Threshold Rejection
        # Check 1: Score confidence threshold
        if confidence < self.threshold:
            return {
                "is_accepted": False,
                "matched_pid": -1,
                "confidence": confidence,
                "top1_score": top1_raw,
                "top2_score": top2_raw,
                "rejection_reason": f"Confidence ({confidence:.3f}) below threshold ({self.threshold:.3f})"
            }

        # Check 2: Ambiguity margin check (rejects uniform collisions where query matches 2 people equally)
        if self.ambiguity_margin > 0 and margin < self.ambiguity_margin:
            return {
                "is_accepted": False,
                "matched_pid": -1,
                "confidence": confidence,
                "top1_score": top1_raw,
                "top2_score": top2_raw,
                "rejection_reason": f"Margin ({margin:.3f}) below ambiguity threshold ({self.ambiguity_margin:.3f})"
            }

        return {
            "is_accepted": True,
            "matched_pid": top1_pid,
            "confidence": confidence,
            "top1_score": top1_raw,
            "top2_score": top2_raw,
            "rejection_reason": None
        }

    def match_batch(self, query_features: np.ndarray) -> List[Dict]:
        """
        Batch inference for evaluating entire probe sets.
        """
        results = []
        for i in range(query_features.shape[0]):
            res = self.match_single(query_features[i])
            results.append(res)
        return results
