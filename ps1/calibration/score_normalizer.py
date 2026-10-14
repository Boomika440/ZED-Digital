"""
Gallery Score Normalization and Confidence Calibration.
Includes:
1. S-Norm (Symmetric Score Normalization) against cohort background to alleviate uniform hubness.
2. Temperature / Platt Logistic Calibration for converting raw cosine into calibrated confidences.
"""

import numpy as np


class SNormNormalizer:
    """
    Symmetric Score Normalization (S-Norm).
    Normalizes similarity between query q and gallery g by computing their z-scores
    against a background cohort C:
        S_norm(q, g) = 0.5 * [(S(q, g) - mu_q) / sigma_q + (S(q, g) - mu_g) / sigma_g]
    """
    def __init__(self, cohort_k: int = 15):
        self.cohort_k = cohort_k
        self.cohort_embeddings = None

    def fit_cohort(self, cohort_embeddings: np.ndarray):
        """
        Registers cohort embeddings (e.g. from gallery or training set).
        """
        self.cohort_embeddings = cohort_embeddings

    def normalize(self, query_feats: np.ndarray, gallery_feats: np.ndarray, raw_scores: np.ndarray) -> np.ndarray:
        """
        Args:
            query_feats: [M, D] L2 normalized query embeddings.
            gallery_feats: [N, D] L2 normalized gallery embeddings.
            raw_scores: [M, N] cosine similarity matrix.
        Returns:
            normalized_scores: [M, N] S-Norm normalized similarity matrix.
        """
        if self.cohort_embeddings is None:
            return raw_scores

        # Query vs Cohort scores: [M, C]
        q_cohort = np.dot(query_feats, self.cohort_embeddings.T)
        # Gallery vs Cohort scores: [N, C]
        g_cohort = np.dot(gallery_feats, self.cohort_embeddings.T)

        # Compute query statistics over cohort
        mu_q = np.mean(q_cohort, axis=1, keepdims=True)  # [M, 1]
        sigma_q = np.std(q_cohort, axis=1, keepdims=True) + 1e-6

        # Compute gallery statistics over cohort
        mu_g = np.mean(g_cohort, axis=1, keepdims=True).T  # [1, N]
        sigma_g = np.std(g_cohort, axis=1, keepdims=True).T + 1e-6

        # Symmetric normalization
        z_q = (raw_scores - mu_q) / sigma_q
        z_g = (raw_scores - mu_g) / sigma_g
        s_norm = 0.5 * (z_q + z_g)

        return s_norm


class PlattCalibrator:
    """
    Platt Scaling (Logistic Sigmoid Calibration).
    Maps similarity score s into calibrated probability P(Match | s) in [0, 1].
    """
    def __init__(self, temperature: float = 0.08, bias: float = 0.70):
        self.temperature = temperature
        self.bias = bias

    def fit(self, match_scores: np.ndarray, non_match_scores: np.ndarray):
        """
        Fits temperature and bias using validation pairs via maximum likelihood.
        """
        from scipy.optimize import minimize
        all_scores = np.concatenate([match_scores, non_match_scores])
        all_labels = np.concatenate([np.ones_like(match_scores), np.zeros_like(non_match_scores)])

        def nll(params):
            t, b = params
            logits = (all_scores - b) / max(t, 1e-4)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
            loss = -np.mean(all_labels * np.log(probs) + (1.0 - all_labels) * np.log(1.0 - probs))
            return loss

        res = minimize(nll, [self.temperature, self.bias], bounds=[(0.01, 1.0), (0.1, 0.99)])
        if res.success:
            self.temperature, self.bias = res.x

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        logits = (scores - self.bias) / self.temperature
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
