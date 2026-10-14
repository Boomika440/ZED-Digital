#include "kalman_filter.hpp"
#include <cmath>
#include <algorithm>

namespace ZedDigital {

KalmanBoxFilter::KalmanBoxFilter(float std_weight_position, float std_weight_velocity, float velocity_damping)
    : m_std_pos(std_weight_position),
      m_std_vel(std_weight_velocity),
      m_damping(velocity_damping)
{
}

void KalmanBoxFilter::initiate(
    const std::array<float, 4>& measurement,
    std::array<float, 8>& mean,
    std::array<std::array<float, 8>, 8>& cov
) {
    for (int i = 0; i < 4; ++i) {
        mean[i] = measurement[i];
        mean[i + 4] = 0.0f;
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            cov[i][j] = 0.0f;
        }
    }

    float h = measurement[3];
    float std_pos_val = 2.0f * m_std_pos * h;
    float std_vel_val = 10.0f * m_std_vel * h;

    cov[0][0] = std_pos_val * std_pos_val;
    cov[1][1] = std_pos_val * std_pos_val;
    cov[2][2] = 1e-4f;
    cov[3][3] = std_pos_val * std_pos_val;

    cov[4][4] = std_vel_val * std_vel_val;
    cov[5][5] = std_vel_val * std_vel_val;
    cov[6][6] = 1e-10f;
    cov[7][7] = std_vel_val * std_vel_val;
}

void KalmanBoxFilter::predict(
    std::array<float, 8>& mean,
    std::array<std::array<float, 8>, 8>& cov,
    bool is_occluded
) {
    float h = mean[3];
    float std_pos_val = m_std_pos * h;
    float std_vel_val = m_std_vel * h;

    float dt_pos = is_occluded ? m_damping : 1.0f;
    if (is_occluded) {
        for (int i = 4; i < 8; ++i) {
            mean[i] *= m_damping;
        }
    }

    // x = x + dt * vx
    for (int i = 0; i < 4; ++i) {
        mean[i] += mean[i + 4] * dt_pos;
    }

    // P' = F * P * F^T + Q
    std::array<std::array<float, 8>, 8> new_cov = cov;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            new_cov[i][j] += cov[i + 4][j] * dt_pos;
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            new_cov[i][j] += new_cov[i][j + 4] * dt_pos;
        }
    }

    // Add Q
    new_cov[0][0] += std_pos_val * std_pos_val;
    new_cov[1][1] += std_pos_val * std_pos_val;
    new_cov[2][2] += 1e-4f;
    new_cov[3][3] += std_pos_val * std_pos_val;

    new_cov[4][4] += std_vel_val * std_vel_val;
    new_cov[5][5] += std_vel_val * std_vel_val;
    new_cov[6][6] += 1e-10f;
    new_cov[7][7] += std_vel_val * std_vel_val;

    cov = new_cov;
    mean[2] = std::clamp(mean[2], 0.15f, 0.90f);
    mean[3] = std::max(10.0f, mean[3]);
}

void KalmanBoxFilter::project(
    const std::array<float, 8>& mean,
    const std::array<std::array<float, 8>, 8>& cov,
    std::array<float, 4>& proj_mean,
    std::array<std::array<float, 4>, 4>& proj_cov
) {
    for (int i = 0; i < 4; ++i) {
        proj_mean[i] = mean[i];
    }

    float h = mean[3];
    float std_r = m_std_pos * h;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            proj_cov[i][j] = cov[i][j];
        }
    }

    proj_cov[0][0] += std_r * std_r;
    proj_cov[1][1] += std_r * std_r;
    proj_cov[2][2] += 1e-2f;
    proj_cov[3][3] += std_r * std_r;
}

void KalmanBoxFilter::update(
    std::array<float, 8>& mean,
    std::array<std::array<float, 8>, 8>& cov,
    const std::array<float, 4>& measurement
) {
    std::array<float, 4> proj_mean;
    std::array<std::array<float, 4>, 4> S;
    project(mean, cov, proj_mean, S);

    // Simplified diagonal gain update for fast bounded latency
    std::array<float, 8> K;
    for (int i = 0; i < 4; ++i) {
        float var_s = std::max(1e-4f, S[i][i]);
        K[i] = cov[i][i] / var_s;
        K[i + 4] = cov[i + 4][i] / var_s;
    }

    for (int i = 0; i < 4; ++i) {
        float diff = measurement[i] - proj_mean[i];
        mean[i] += K[i] * diff;
        mean[i + 4] += K[i + 4] * diff;

        cov[i][i] *= (1.0f - K[i]);
        cov[i + 4][i + 4] *= (1.0f - K[i + 4] * 0.5f);
    }

    mean[2] = std::clamp(mean[2], 0.15f, 0.90f);
    mean[3] = std::max(10.0f, mean[3]);
}

float KalmanBoxFilter::computeMahalanobis(
    const std::array<float, 8>& mean,
    const std::array<std::array<float, 8>, 8>& cov,
    const std::array<float, 4>& measurement
) {
    std::array<float, 4> proj_mean;
    std::array<std::array<float, 4>, 4> S;
    project(mean, cov, proj_mean, S);

    float dist_sq = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float diff = measurement[i] - proj_mean[i];
        dist_sq += (diff * diff) / std::max(1e-4f, S[i][i]);
    }
    return dist_sq;
}

} // namespace ZedDigital
