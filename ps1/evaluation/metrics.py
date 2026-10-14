"""
Open-Set Person Re-Identification Evaluation Metrics.
Implements:
1. Detection and Identification Rate (DIR) at fixed False-Accept Rates (FAR = 0.1%, 1%, 5%, 10%).
2. ROC Curve computation (DIR vs FAR trade-off).
3. Area Under Curve (AUROC / DIR-FAR AUC).
4. Fixed-Recall Thresholding (FAR at fixed TAR = 90%, 95%).
5. Closed-set CMC (Rank-1, Rank-5, Rank-10) and mAP for known queries.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_roc_and_dir(
    known_scores: np.ndarray,
    known_correct_mask: np.ndarray,
    unknown_scores: np.ndarray,
    target_fars: List[float] = [0.001, 0.01, 0.05, 0.10],
    num_thresholds: int = 2000
) -> Dict:
    """
    Computes Open-Set ROC curve and DIR @ fixed FARs.
    Args:
        known_scores: [N_known] Confidence/similarity scores of top-1 matches for known probes.
        known_correct_mask: [N_known] Boolean array, True if top-1 predicted PID matches ground truth PID.
        unknown_scores: [N_unknown] Confidence/similarity scores of top-1 matches for unknown/impostor probes.
        target_fars: List of target FAR values to evaluate DIR.
    Returns:
        Dictionary containing FARs, TARs, DIRs, AUC, and DIR at target FARs.
    """
    n_known = len(known_scores)
    n_unknown = len(unknown_scores)

    assert n_known > 0, "Known probe count must be > 0"
    assert n_unknown > 0, "Unknown probe count must be > 0"

    min_val = min(float(np.min(known_scores)), float(np.min(unknown_scores)))
    max_val = max(float(np.max(known_scores)), float(np.max(unknown_scores)))
    thresholds = np.linspace(min_val - 1e-4, max_val + 1e-4, num_thresholds)

    far_list = []
    tar_list = []
    dir_list = []

    for t in thresholds:
        # False Accept Rate: proportion of impostors above threshold t
        fa_count = np.sum(unknown_scores >= t)
        far = fa_count / n_unknown

        # True Accept Rate (Verification Recall): proportion of known queries above threshold t
        ta_count = np.sum(known_scores >= t)
        tar = ta_count / n_known

        # Detection and Identification Rate: accepted above t AND correctly identified
        dir_count = np.sum((known_scores >= t) & known_correct_mask)
        dir_rate = dir_count / n_known

        far_list.append(far)
        tar_list.append(tar)
        dir_list.append(dir_rate)

    far_arr = np.array(far_list)
    tar_arr = np.array(tar_list)
    dir_arr = np.array(dir_list)

    # Sort by increasing FAR for monotonic curve and interpolation
    sort_idx = np.argsort(far_arr)
    far_sorted = far_arr[sort_idx]
    dir_sorted = dir_arr[sort_idx]
    tar_sorted = tar_arr[sort_idx]
    thresh_sorted = thresholds[sort_idx]

    # Calculate DIR @ Target FARs via interpolation
    dir_at_far = {}
    threshold_at_far = {}
    for target_far in target_fars:
        idx = np.searchsorted(far_sorted, target_far)
        if idx < len(dir_sorted):
            dir_val = float(dir_sorted[idx])
            thresh_val = float(thresh_sorted[idx])
        else:
            dir_val = float(dir_sorted[-1])
            thresh_val = float(thresh_sorted[-1])
        dir_at_far[f"DIR@FAR={target_far:.1%}"] = dir_val
        threshold_at_far[f"Thresh@FAR={target_far:.1%}"] = thresh_val

    # Fixed Recall Thresholding (FAR at TAR = 90% and TAR = 95%)
    tar_rev_idx = np.argsort(tar_sorted)
    far_at_tar = {}
    for target_tar in [0.90, 0.95]:
        t_idx = np.searchsorted(tar_sorted[tar_rev_idx], target_tar)
        if t_idx < len(far_sorted):
            far_at_tar[f"FAR@TAR={int(target_tar*100)}%"] = float(far_sorted[tar_rev_idx][t_idx])
        else:
            far_at_tar[f"FAR@TAR={int(target_tar*100)}%"] = float(far_sorted[tar_rev_idx][-1])

    # Area Under ROC (DIR vs FAR)
    auc = float(np.trapezoid(dir_sorted, far_sorted))

    return {
        "far": far_sorted,
        "tar": tar_sorted,
        "dir": dir_sorted,
        "thresholds": thresh_sorted,
        "auc": auc,
        "dir_at_far": dir_at_far,
        "threshold_at_far": threshold_at_far,
        "far_at_tar": far_at_tar
    }


def compute_cmc_and_map(dist_matrix: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray, topk=(1, 5, 10)) -> Dict:
    """
    Computes standard closed-set CMC (Cumulative Matching Characteristics) and mean Average Precision.
    Args:
        dist_matrix: [N_q, N_g] distance matrix (smaller = closer).
        q_pids: [N_q] query labels.
        g_pids: [N_g] gallery labels.
    """
    m, n = dist_matrix.shape
    indices = np.argsort(dist_matrix, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

    cmc = np.zeros(len(topk), dtype=float)
    for i, k in enumerate(topk):
        cmc[i] = np.mean(np.any(matches[:, :k], axis=1))

    # Compute mAP
    all_ap = []
    for i in range(m):
        row_matches = matches[i]
        if not np.any(row_matches):
            all_ap.append(0.0)
            continue
        d_recall = 1.0 / np.sum(row_matches)
        precision = np.cumsum(row_matches) / np.arange(1, n + 1)
        ap = np.sum(precision[row_matches] * d_recall)
        all_ap.append(ap)

    map_score = float(np.mean(all_ap))

    return {
        "Rank-1": float(cmc[0]),
        "Rank-5": float(cmc[1]) if len(topk) > 1 else 0.0,
        "Rank-10": float(cmc[2]) if len(topk) > 2 else 0.0,
        "mAP": map_score
    }
