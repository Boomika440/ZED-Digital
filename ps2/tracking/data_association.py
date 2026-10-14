"""
Hierarchical Data Association Engine for Dense Occlusion Tracking.
Implements:
1. Appearance-Motion Cost Matrix Fusion with Mahalanobis Motion Gating.
2. Two-Stage Detection Score Association (ByteTrack-style occlusion recovery).
3. Lost Track Global Re-Association Bank (Identity-Preserving Recovery).
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .bboxes import box_ious, xyxy_to_xyah
    from .track import Track, TrackState
except (ImportError, ValueError):
    try:
        from tracking.bboxes import box_ious, xyxy_to_xyah
        from tracking.track import Track, TrackState
    except ImportError:
        from ps2.tracking.bboxes import box_ious, xyxy_to_xyah
        from ps2.tracking.track import Track, TrackState


def compute_appearance_cost_matrix(tracks: List[Track], detection_features: np.ndarray) -> np.ndarray:
    """
    Computes cosine distance matrix between track smooth features and detection features:
    C_app(i, j) = 1.0 - (f_i . f_j) / (||f_i|| * ||f_j||)
    Returns: [num_tracks, num_detections] matrix in [0.0, 2.0]
    """
    num_tracks = len(tracks)
    num_dets = len(detection_features)
    if num_tracks == 0 or num_dets == 0:
        return np.zeros((num_tracks, num_dets), dtype=np.float32)

    cost_matrix = np.ones((num_tracks, num_dets), dtype=np.float32)
    det_feats = np.asarray(detection_features, dtype=np.float32)
    # L2 normalize detections
    det_norms = np.linalg.norm(det_feats, axis=1, keepdims=True)
    det_feats = det_feats / np.maximum(det_norms, 1e-8)

    for i, trk in enumerate(tracks):
        ref_feat = trk.last_clean_feature if trk.last_clean_feature is not None else trk.smooth_feature
        if ref_feat is not None:
            # Cosine similarity
            sims = np.dot(det_feats, ref_feat)
            cost_matrix[i] = 1.0 - sims
        else:
            cost_matrix[i] = 1.0  # Neutral cost when appearance is missing

    return np.clip(cost_matrix, 0.0, 2.0).astype(np.float32)


def compute_motion_cost_matrix(tracks: List[Track], detection_boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Computes IoU-based motion cost matrix:
    C_motion(i, j) = 1.0 - IoU(box_i, box_j)
    """
    num_tracks = len(tracks)
    num_dets = len(detection_boxes_xyxy)
    if num_tracks == 0 or num_dets == 0:
        return np.zeros((num_tracks, num_dets), dtype=np.float32)

    track_boxes = np.array([t.to_xyxy for t in tracks], dtype=np.float32)
    ious = box_ious(track_boxes, detection_boxes_xyxy)
    return (1.0 - ious).astype(np.float32)


def fuse_motion_and_appearance(
    motion_cost: np.ndarray,
    app_cost: np.ndarray,
    lambda_app: float = 0.40,
    iou_gate_thresh: float = 0.05
) -> np.ndarray:
    """
    Fuses appearance and motion costs with loose spatial gating.
    If IoU is 0 (spatially separated), penalizes cost to prevent distant identity drift.
    """
    fused = lambda_app * app_cost + (1.0 - lambda_app) * motion_cost

    # Spatial sanity gate: If box overlap is 0 (motion_cost >= 1.0) and appearance is not extremely close (< 0.20),
    # penalize heavily to avoid jumping across the scene in single frame
    spatial_disconnect = (motion_cost >= (1.0 - iou_gate_thresh)) & (app_cost > 0.20)
    fused[spatial_disconnect] = 1e5

    return fused.astype(np.float32)


def apply_mahalanobis_gating(
    cost_matrix: np.ndarray,
    tracks: List[Track],
    detection_boxes_xyxy: np.ndarray,
    kf,
    chi2_threshold: float = 9.488
) -> np.ndarray:
    """
    Applies Chi-Square Mahalanobis distance gating to cost matrix.
    Pairs exceeding chi2_threshold are assigned infinity cost.
    """
    if cost_matrix.size == 0:
        return cost_matrix

    meas_xyah = xyxy_to_xyah(detection_boxes_xyxy)
    gated_cost = cost_matrix.copy()

    for i, trk in enumerate(tracks):
        dists_sq = kf.compute_mahalanobis_distance(trk.mean, trk.covariance, meas_xyah)
        gate_mask = dists_sq > chi2_threshold
        gated_cost[i, gate_mask] = 1e5

    return gated_cost


def linear_assignment(cost_matrix: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves linear sum assignment problem via Hungarian algorithm.
    Returns:
        matches: [K, 2] indices of (track_idx, det_idx)
        unmatched_tracks: list of track indices
        unmatched_detections: list of detection indices
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int)
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(cost_matrix.shape[0]))
    unmatched_detections = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= threshold:
            matches.append([r, c])
            unmatched_tracks.remove(r)
            unmatched_detections.remove(c)

    matches_arr = np.array(matches, dtype=int).reshape(-1, 2)
    return matches_arr, np.array(list(unmatched_tracks), dtype=int), np.array(list(unmatched_detections), dtype=int)


class HierarchicalDataAssociator:
    """
    Full multi-stage data association pipeline handling mutual occlusions and re-identifications.
    """
    def __init__(
        self,
        match_thresh_stage1: float = 0.75,
        match_thresh_stage2: float = 0.55,
        reassociation_thresh: float = 0.35,
        lambda_appearance: float = 0.40,
        use_appearance: bool = True,
        use_mahalanobis_gate: bool = True,
        chi2_threshold: float = 9.488
    ):
        self.match_thresh_stage1 = match_thresh_stage1
        self.match_thresh_stage2 = match_thresh_stage2
        self.reassociation_thresh = reassociation_thresh
        self.lambda_appearance = lambda_appearance
        self.use_appearance = use_appearance
        self.use_mahalanobis_gate = use_mahalanobis_gate
        self.chi2_threshold = chi2_threshold

    def associate(
        self,
        active_tracks: List[Track],
        lost_tracks: List[Track],
        tentative_tracks: List[Track],
        dets_high_boxes: np.ndarray,
        dets_high_scores: np.ndarray,
        dets_high_feats: Optional[np.ndarray],
        dets_low_boxes: np.ndarray,
        dets_low_scores: np.ndarray,
        kf
    ) -> dict:
        """
        Executes hierarchical 4-stage association:
        1. Stage 1: Active Confirmed Tracks vs. High-Score Detections (Appearance + Motion)
        2. Stage 2: Remaining Active Tracks vs. Low-Score Detections (Motion IoU for occluded targets)
        3. Stage 3: Remaining High-Score Detections vs. Lost Track Bank (Re-Association)
        4. Stage 4: Remaining High-Score Detections vs. Tentative Tracks (Confirmation)
        """
        # =========================================================================
        # STAGE 1: Confirmed Active Tracks vs. High-Confidence Detections
        # =========================================================================
        num_active = len(active_tracks)
        num_dets_high = len(dets_high_boxes)

        if num_active > 0 and num_dets_high > 0:
            motion_cost = compute_motion_cost_matrix(active_tracks, dets_high_boxes)
            if self.use_appearance and dets_high_feats is not None:
                app_cost = compute_appearance_cost_matrix(active_tracks, dets_high_feats)
                cost_s1 = fuse_motion_and_appearance(motion_cost, app_cost, self.lambda_appearance)
            else:
                cost_s1 = motion_cost

            if self.use_mahalanobis_gate:
                cost_s1 = apply_mahalanobis_gating(cost_s1, active_tracks, dets_high_boxes, kf, self.chi2_threshold)

            matches_s1, unconfirmed_tracks_s1, unconfirmed_dets_high_s1 = linear_assignment(
                cost_s1, self.match_thresh_stage1
            )
        else:
            matches_s1 = np.empty((0, 2), dtype=int)
            unconfirmed_tracks_s1 = np.arange(num_active, dtype=int)
            unconfirmed_dets_high_s1 = np.arange(num_dets_high, dtype=int)

        # =========================================================================
        # STAGE 2: Remaining Active Tracks vs. Low-Confidence Detections (Occlusion Recovery)
        # =========================================================================
        remain_active_tracks = [active_tracks[i] for i in unconfirmed_tracks_s1]
        num_remain_active = len(remain_active_tracks)
        num_dets_low = len(dets_low_boxes)

        if num_remain_active > 0 and num_dets_low > 0:
            # Low-score detections arise from partial occlusions; use pure IoU motion cost
            cost_s2 = compute_motion_cost_matrix(remain_active_tracks, dets_low_boxes)
            matches_s2_local, unconfirmed_tracks_s2_local, _ = linear_assignment(
                cost_s2, self.match_thresh_stage2
            )
            # Map back to original active track indices
            matches_s2 = []
            for trk_local_idx, det_low_idx in matches_s2_local:
                original_trk_idx = unconfirmed_tracks_s1[trk_local_idx]
                matches_s2.append([original_trk_idx, det_low_idx])
            matches_s2 = np.array(matches_s2, dtype=int).reshape(-1, 2)
            unmatched_active_tracks = [unconfirmed_tracks_s1[i] for i in unconfirmed_tracks_s2_local]
        else:
            matches_s2 = np.empty((0, 2), dtype=int)
            unmatched_active_tracks = list(unconfirmed_tracks_s1)

        # =========================================================================
        # STAGE 3: Remaining High-Score Detections vs. Lost Track Bank (Re-Association)
        # =========================================================================
        remain_dets_high_boxes = dets_high_boxes[unconfirmed_dets_high_s1]
        remain_dets_high_feats = dets_high_feats[unconfirmed_dets_high_s1] if dets_high_feats is not None else None
        num_remain_dets = len(remain_dets_high_boxes)
        num_lost = len(lost_tracks)

        reassoc_matches = []
        unmatched_dets_high_after_reassoc = list(unconfirmed_dets_high_s1)
        unmatched_lost_tracks = list(range(num_lost))

        if num_lost > 0 and num_remain_dets > 0 and self.use_appearance and remain_dets_high_feats is not None:
            # Appearance distance between lost tracks and reappearing detections
            app_cost_lost = compute_appearance_cost_matrix(lost_tracks, remain_dets_high_feats)
            motion_cost_lost = compute_motion_cost_matrix(lost_tracks, remain_dets_high_boxes)
            # Weight appearance heavily during re-association after disappearance
            cost_s3 = 0.70 * app_cost_lost + 0.30 * motion_cost_lost

            matches_s3_local, un_lost_local, un_dets_s3_local = linear_assignment(
                cost_s3, self.reassociation_thresh
            )
            for lost_local_idx, det_remain_idx in matches_s3_local:
                original_det_high_idx = unconfirmed_dets_high_s1[det_remain_idx]
                reassoc_matches.append([lost_local_idx, original_det_high_idx])
                unmatched_dets_high_after_reassoc.remove(original_det_high_idx)

            reassoc_matches = np.array(reassoc_matches, dtype=int).reshape(-1, 2)
            unmatched_lost_tracks = [lost_local_idx for lost_local_idx in un_lost_local]
        else:
            reassoc_matches = np.empty((0, 2), dtype=int)

        # =========================================================================
        # STAGE 4: Remaining High-Score Detections vs. Tentative Tracks
        # =========================================================================
        num_tentative = len(tentative_tracks)
        still_unmatched_det_boxes = dets_high_boxes[unmatched_dets_high_after_reassoc]
        tentative_matches = []
        final_unmatched_dets_high = list(unmatched_dets_high_after_reassoc)
        unmatched_tentative_tracks = list(range(num_tentative))

        if num_tentative > 0 and len(still_unmatched_det_boxes) > 0:
            cost_s4 = compute_motion_cost_matrix(tentative_tracks, still_unmatched_det_boxes)
            matches_s4_local, un_tent_local, _ = linear_assignment(cost_s4, self.match_thresh_stage2)

            for tent_local_idx, det_sub_idx in matches_s4_local:
                orig_det_idx = unmatched_dets_high_after_reassoc[det_sub_idx]
                tentative_matches.append([tent_local_idx, orig_det_idx])
                final_unmatched_dets_high.remove(orig_det_idx)

            tentative_matches = np.array(tentative_matches, dtype=int).reshape(-1, 2)
            unmatched_tentative_tracks = list(un_tent_local)
        else:
            tentative_matches = np.empty((0, 2), dtype=int)

        return {
            "matches_stage1": matches_s1,                         # (active_idx, det_high_idx)
            "matches_stage2": matches_s2,                         # (active_idx, det_low_idx)
            "reassoc_matches": reassoc_matches,                   # (lost_idx, det_high_idx)
            "tentative_matches": tentative_matches,               # (tent_idx, det_high_idx)
            "unmatched_active_tracks": unmatched_active_tracks,   # active indices needing miss handling
            "unmatched_lost_tracks": unmatched_lost_tracks,       # lost indices
            "unmatched_tentative_tracks": unmatched_tentative_tracks, # tentative indices
            "unmatched_dets_high": final_unmatched_dets_high      # detections to spawn new tracks
        }
