"""
Occlusion-Aware Re-Association Multi-Object Tracker (OAR-Tracker).
Core Causal Online Multi-Object Tracking Engine for Dense Gate Surveillance.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

try:
    from .bboxes import xyxy_to_xyah, compute_inter_track_overlap
    from .kalman_filter import KalmanBoxTracker
    from .track import Track, TrackState
    from .data_association import HierarchicalDataAssociator
except (ImportError, ValueError):
    try:
        from tracking.bboxes import xyxy_to_xyah, compute_inter_track_overlap
        from tracking.kalman_filter import KalmanBoxTracker
        from tracking.track import Track, TrackState
        from tracking.data_association import HierarchicalDataAssociator
    except ImportError:
        from ps2.tracking.bboxes import xyxy_to_xyah, compute_inter_track_overlap
        from ps2.tracking.kalman_filter import KalmanBoxTracker
        from ps2.tracking.track import Track, TrackState
        from ps2.tracking.data_association import HierarchicalDataAssociator


class OcclusionAwareReIDTracker:
    """
    Online Causal Multi-Object Tracker engineered to maintain stable identities
    through mutual occlusion and short disappearances in crowded choke-points.
    """
    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        tracker_cfg = cfg.get("tracker", {})
        kalman_cfg = cfg.get("kalman", {})

        # Detection Thresholds
        self.track_high_thresh = tracker_cfg.get("track_high_thresh", 0.60)
        self.track_low_thresh = tracker_cfg.get("track_low_thresh", 0.15)
        self.new_track_thresh = tracker_cfg.get("new_track_thresh", 0.70)

        # Association Parameters
        self.match_thresh_stage1 = tracker_cfg.get("match_thresh_stage1", 0.75)
        self.match_thresh_stage2 = tracker_cfg.get("match_thresh_stage2", 0.55)
        self.reassociation_thresh = tracker_cfg.get("reassociation_thresh", 0.35)
        self.lambda_appearance = tracker_cfg.get("lambda_appearance", 0.40)
        self.use_appearance = tracker_cfg.get("use_appearance", True)

        # Lifecycle Management
        self.min_hits_to_confirm = tracker_cfg.get("min_hits_to_confirm", 3)
        self.max_lost_time = tracker_cfg.get("max_lost_time", 60)
        self.ema_alpha = tracker_cfg.get("appearance_ema_alpha", 0.90)

        # Occlusion Parameters
        self.overlap_thresh = tracker_cfg.get("inter_track_overlap_thresh", 0.35)
        self.freeze_appearance_on_occlusion = tracker_cfg.get("freeze_appearance_on_occlusion", True)

        # Initialize Kalman Filter
        self.kf = KalmanBoxTracker(
            std_weight_position=kalman_cfg.get("std_weight_position", 0.05),
            std_weight_velocity=kalman_cfg.get("std_weight_velocity", 0.00625),
            velocity_damping_factor=kalman_cfg.get("velocity_damping_factor", 0.95)
        )

        # Initialize Associator
        self.associator = HierarchicalDataAssociator(
            match_thresh_stage1=self.match_thresh_stage1,
            match_thresh_stage2=self.match_thresh_stage2,
            reassociation_thresh=self.reassociation_thresh,
            lambda_appearance=self.lambda_appearance,
            use_appearance=self.use_appearance,
            use_mahalanobis_gate=kalman_cfg.get("use_mahalanobis_gate", True),
            chi2_threshold=kalman_cfg.get("mahalanobis_threshold", 9.488)
        )

        # Active Track Banks
        self.tracks: List[Track] = []
        self.frame_id = 0

    def reset(self):
        """Resets tracker state and track IDs."""
        self.tracks.clear()
        self.frame_id = 0
        Track.reset_counter()

    def update(
        self,
        dets_xyxy: np.ndarray,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Processes a single causal video frame.
        Args:
            dets_xyxy: [N, 4] bounding boxes in [x1, y1, x2, y2]
            scores: [N] detection confidence scores
            features: [N, D] appearance feature embeddings (optional)
        Returns:
            active_tracks: List of confirmed active tracks visible in current frame
        """
        self.frame_id += 1

        if len(dets_xyxy) == 0:
            dets_xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)
            if features is not None:
                features = np.empty((0, features.shape[-1]), dtype=np.float32)

        # -------------------------------------------------------------
        # 1. Kalman Prediction for all existing tracks
        # -------------------------------------------------------------
        for trk in self.tracks:
            trk.predict(self.kf)

        # -------------------------------------------------------------
        # 2. Inter-Track Occlusion Detection (Flagging mutual overlap)
        # -------------------------------------------------------------
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed]
        if len(confirmed_tracks) > 1:
            boxes = np.array([t.to_xyxy for t in confirmed_tracks], dtype=np.float32)
            overlap_ratios = compute_inter_track_overlap(boxes)
            for t, ratio in zip(confirmed_tracks, overlap_ratios):
                is_occ = ratio > self.overlap_thresh
                t.mark_occlusion(is_occ, ratio)
        elif len(confirmed_tracks) == 1:
            confirmed_tracks[0].mark_occlusion(False, 0.0)

        # -------------------------------------------------------------
        # 3. Partition Detections into High and Low Score Pools
        # -------------------------------------------------------------
        high_mask = scores >= self.track_high_thresh
        low_mask = (scores >= self.track_low_thresh) & (~high_mask)

        dets_high_boxes = dets_xyxy[high_mask]
        dets_high_scores = scores[high_mask]
        dets_high_feats = features[high_mask] if features is not None else None

        dets_low_boxes = dets_xyxy[low_mask]
        dets_low_scores = scores[low_mask]

        # -------------------------------------------------------------
        # 4. Partition Existing Tracks into Lifecycle Pools
        # -------------------------------------------------------------
        active_tracks = [t for t in self.tracks if t.is_confirmed and not t.is_lost]
        lost_tracks = [t for t in self.tracks if t.is_lost]
        tentative_tracks = [t for t in self.tracks if t.is_tentative]

        # -------------------------------------------------------------
        # 5. Multi-Stage Hierarchical Association
        # -------------------------------------------------------------
        assoc_result = self.associator.associate(
            active_tracks=active_tracks,
            lost_tracks=lost_tracks,
            tentative_tracks=tentative_tracks,
            dets_high_boxes=dets_high_boxes,
            dets_high_scores=dets_high_scores,
            dets_high_feats=dets_high_feats,
            dets_low_boxes=dets_low_boxes,
            dets_low_scores=dets_low_scores,
            kf=self.kf
        )

        # -------------------------------------------------------------
        # 6. Apply Association Updates to Tracks
        # -------------------------------------------------------------
        # Update Stage 1 Matches (Active Tracks + High-Confidence Detections)
        for trk_idx, det_idx in assoc_result["matches_stage1"]:
            trk = active_tracks[trk_idx]
            box_xyxy = dets_high_boxes[det_idx]
            meas_xyah = xyxy_to_xyah(box_xyxy[None, :])[0]
            feat = dets_high_feats[det_idx] if dets_high_feats is not None else None
            trk.update(self.kf, meas_xyah, dets_high_scores[det_idx], feat)

        # Update Stage 2 Matches (Active Tracks + Low-Confidence Occluded Detections)
        for trk_idx, det_low_idx in assoc_result["matches_stage2"]:
            trk = active_tracks[trk_idx]
            box_xyxy = dets_low_boxes[det_low_idx]
            meas_xyah = xyxy_to_xyah(box_xyxy[None, :])[0]
            # No feature update for low confidence detections (avoid noise)
            trk.update(self.kf, meas_xyah, dets_low_scores[det_low_idx], None)

        # Update Stage 3 Matches (Re-Associated Lost Tracks!)
        for lost_idx, det_idx in assoc_result["reassoc_matches"]:
            trk = lost_tracks[lost_idx]
            box_xyxy = dets_high_boxes[det_idx]
            meas_xyah = xyxy_to_xyah(box_xyxy[None, :])[0]
            feat = dets_high_feats[det_idx] if dets_high_feats is not None else None
            # Successfully re-associated to prior identity!
            trk.update(self.kf, meas_xyah, dets_high_scores[det_idx], feat)

        # Update Stage 4 Matches (Tentative Tracks)
        for tent_idx, det_idx in assoc_result["tentative_matches"]:
            trk = tentative_tracks[tent_idx]
            box_xyxy = dets_high_boxes[det_idx]
            meas_xyah = xyxy_to_xyah(box_xyxy[None, :])[0]
            feat = dets_high_feats[det_idx] if dets_high_feats is not None else None
            trk.update(self.kf, meas_xyah, dets_high_scores[det_idx], feat)

        # Handle Missed Tracks
        for trk_idx in assoc_result["unmatched_active_tracks"]:
            active_tracks[trk_idx].mark_missed()

        for lost_idx in assoc_result["unmatched_lost_tracks"]:
            lost_tracks[lost_idx].mark_missed()

        for tent_idx in assoc_result["unmatched_tentative_tracks"]:
            tentative_tracks[tent_idx].mark_missed()

        # -------------------------------------------------------------
        # 7. Spawn New Tentative Tracks from Unmatched High Detections
        # -------------------------------------------------------------
        for det_idx in assoc_result["unmatched_dets_high"]:
            score = dets_high_scores[det_idx]
            if score >= self.new_track_thresh:
                box_xyxy = dets_high_boxes[det_idx]
                meas_xyah = xyxy_to_xyah(box_xyxy[None, :])[0]
                mean, cov = self.kf.initiate(meas_xyah)
                feat = dets_high_feats[det_idx] if dets_high_feats is not None else None

                new_track = Track(
                    mean=mean,
                    covariance=cov,
                    score=score,
                    feature=feat,
                    min_hits_to_confirm=self.min_hits_to_confirm,
                    max_lost_time=self.max_lost_time,
                    ema_alpha=self.ema_alpha
                )
                self.tracks.append(new_track)

        # -------------------------------------------------------------
        # 8. Filter Out Deleted Tracks
        # -------------------------------------------------------------
        self.tracks = [t for t in self.tracks if not t.is_deleted]

        # -------------------------------------------------------------
        # 9. Return Active Confirmed Tracks for Output / Evaluation
        # -------------------------------------------------------------
        output_tracks = [t for t in self.tracks if t.is_confirmed and t.time_since_update == 0]
        return output_tracks
