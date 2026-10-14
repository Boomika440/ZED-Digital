"""
Adaptive Kalman Filter for Bounding Box Motion Modeling and State Estimation.
State representation: [xc, yc, a, h, v_xc, v_yc, v_a, v_h]^T
Includes:
- Height-scaled process and measurement covariance.
- Velocity damping under prolonged occlusion to prevent runaway linear extrapolation drift.
- Mahalanobis distance gating (Chi-square distribution test).
"""

import numpy as np
import scipy.linalg


class KalmanBoxTracker:
    """
    Kalman filter tracking bounding box geometry with velocity damping and motion uncertainty.
    """
    # 4 degrees of freedom Chi-Square distribution 95% confidence threshold
    CHI2_THRESHOLD_95 = 9.488

    def __init__(
        self,
        std_weight_position: float = 0.05,
        std_weight_velocity: float = 0.00625,
        velocity_damping_factor: float = 0.95
    ):
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self.velocity_damping_factor = velocity_damping_factor

        # State transition matrix F: 8x8 constant velocity model
        self._motion_mat = np.eye(8, 8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0  # pos = pos + vel * dt (dt = 1 frame)

        # Measurement matrix H: 4x8 mapping state to [xc, yc, a, h]
        self._update_mat = np.eye(4, 8, dtype=np.float32)

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Initializes track state from an unassociated detection [xc, yc, a, h].
        Returns:
            mean: [8] initial state vector
            covariance: [8, 8] initial covariance matrix
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)

        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray, is_occluded: bool = False) -> tuple:
        """
        Advances the state vector and covariance by one time step.
        If is_occluded=True, applies velocity damping to mitigate runaway extrapolation.
        """
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        # Apply velocity damping during occlusion
        curr_motion_mat = self._motion_mat.copy()
        if is_occluded:
            # Damp velocity transfer
            for i in range(4):
                curr_motion_mat[i, i + 4] *= self.velocity_damping_factor
            # Damp velocity state
            mean[4:] *= self.velocity_damping_factor

        pred_mean = np.dot(curr_motion_mat, mean)
        pred_cov = np.linalg.multi_dot([curr_motion_mat, covariance, curr_motion_mat.T]) + motion_cov

        # Clamp aspect ratio to plausible human bounding box boundaries (e.g. 0.15 to 0.85)
        pred_mean[2] = np.clip(pred_mean[2], 0.15, 0.90)
        pred_mean[3] = np.maximum(10.0, pred_mean[3])

        return pred_mean.astype(np.float32), pred_cov.astype(np.float32)

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Projects state distribution to measurement space [xc, yc, a, h].
        Returns:
            projected_mean: [4]
            innovation_cov: [4, 4]
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        meas_cov = np.diag(np.square(std)).astype(np.float32)

        proj_mean = np.dot(self._update_mat, mean)
        proj_cov = np.linalg.multi_dot([self._update_mat, covariance, self._update_mat.T]) + meas_cov
        return proj_mean.astype(np.float32), proj_cov.astype(np.float32)

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """
        Updates state and covariance with a matched detection [xc, yc, a, h].
        """
        proj_mean, proj_cov = self.project(mean, covariance)

        # Kalman gain computation via Cholesky decomposition for numerical stability
        chol_factor, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T

        innovation = measurement - proj_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_cov = covariance - np.linalg.multi_dot([kalman_gain, proj_cov, kalman_gain.T])

        new_mean[2] = np.clip(new_mean[2], 0.15, 0.90)
        new_mean[3] = np.maximum(10.0, new_mean[3])

        return new_mean.astype(np.float32), new_cov.astype(np.float32)

    def compute_mahalanobis_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray
    ) -> np.ndarray:
        """
        Computes Mahalanobis distance squared between track prediction and candidate measurements:
        d_M^2 = (z - Hx)^T S^-1 (z - Hx)
        Args:
            mean: [8] state vector
            covariance: [8, 8] covariance
            measurements: [M, 4] array of candidate measurements
        Returns:
            dists_sq: [M] array of squared Mahalanobis distances
        """
        if len(measurements) == 0:
            return np.empty(0, dtype=np.float32)

        proj_mean, proj_cov = self.project(mean, covariance)
        diff = measurements - proj_mean[None, :]

        try:
            chol_factor, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
            inv_diff = scipy.linalg.cho_solve((chol_factor, lower), diff.T, check_finite=False).T
            dists_sq = np.sum(diff * inv_diff, axis=1)
        except Exception:
            # Fallback to pseudo-inverse
            inv_cov = np.linalg.pinv(proj_cov)
            dists_sq = np.sum(np.dot(diff, inv_cov) * diff, axis=1)

        return np.maximum(0.0, dists_sq).astype(np.float32)
