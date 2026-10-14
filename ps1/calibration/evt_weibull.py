"""
Extreme Value Theory (EVT) Weibull Distribution Score Calibration.
References:
- Scheirer et al., 'Toward Open Set Recognition', IEEE TPAMI 2013.
- Bendale & Boult, 'Towards Open Set Deep Networks' (OpenMax), CVPR 2016.

Fits a Weibull distribution to the lower-tail non-match distance distribution of each enrolled identity.
Converts unbounded/uncalibrated cosine distance into a calibrated statistical probability
that the query belongs to the extreme match tail, rejecting unenrolled impostors.
"""

import numpy as np
from scipy.stats import weibull_min


class EVTWeibullCalibrator:
    """
    Calibrates distance scores using Extreme Value Theory.
    For each enrolled gallery identity:
    1. Collect distances to all non-matching enrolled samples.
    2. Extract the tail of size 'tail_size' (the closest impostors / hard negatives).
    3. Fit a Weibull distribution: parameters (shape, loc, scale).
    4. At inference time, compute CDF(d) = 1 - exp(- (d / scale)^shape).
    """
    def __init__(self, tail_size: int = 20):
        self.tail_size = tail_size
        self.models = {}  # pid -> (shape, loc, scale)

    def fit(self, gallery_features: np.ndarray, gallery_pids: np.ndarray):
        """
        Fits Weibull parameters for each gallery identity using enrolled training/gallery features.
        Args:
            gallery_features: [N, D] L2-normalized feature embeddings.
            gallery_pids: [N] identity labels.
        """
        unique_pids = np.unique(gallery_pids)
        # Cosine distance matrix: d = 1 - cos(theta) in [0, 2]
        similarity_mat = np.dot(gallery_features, gallery_features.T)
        dist_mat = np.clip(1.0 - similarity_mat, 0.0, 2.0)

        for pid in unique_pids:
            # Mask of all non-match samples
            non_match_mask = (gallery_pids != pid)
            pid_indices = np.where(gallery_pids == pid)[0]

            if len(pid_indices) == 0 or np.sum(non_match_mask) == 0:
                continue

            # Non-match distances to this identity
            non_match_dists = dist_mat[pid_indices][:, non_match_mask].flatten()
            sorted_dists = np.sort(non_match_dists)

            # Take the smallest non-match distances (closest hard impostor tail)
            tail = sorted_dists[:min(self.tail_size, len(sorted_dists))]

            if len(tail) >= 5 and np.std(tail) > 1e-5:
                try:
                    # Fit Weibull distribution to lower tail
                    shape, loc, scale = weibull_min.fit(tail, floc=0)
                    self.models[pid] = (shape, loc, scale)
                except Exception:
                    self.models[pid] = (2.0, 0.0, float(np.mean(tail) + 1e-4))
            else:
                mean_val = float(np.mean(tail)) if len(tail) > 0 else 0.5
                self.models[pid] = (2.0, 0.0, mean_val + 1e-4)

        return self

    def compute_match_probability(self, query_dist: float, pid: int) -> float:
        """
        Computes calibrated probability that query is a true match given its distance to gallery identity 'pid'.
        If distance is far inside the non-match tail, match probability drops to 0.
        """
        if pid not in self.models:
            # Fallback to sigmoid scaling of distance
            return float(1.0 / (1.0 + np.exp((query_dist - 0.4) * 10.0)))

        shape, loc, scale = self.models[pid]
        # Cumulative probability under Weibull
        p_non_match = weibull_min.cdf(query_dist, shape, loc=loc, scale=scale)
        p_match = float(np.clip(1.0 - p_non_match, 0.0, 1.0))
        return p_match
