#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <vector>
#include <array>
#include "types.hpp"

namespace ZedDigital {

class KalmanBoxFilter {
public:
    KalmanBoxFilter(
        float std_weight_position = 0.05f,
        float std_weight_velocity = 0.00625f,
        float velocity_damping = 0.95f
    );

    // Initialize track state from measurement [xc, yc, a, h]
    void initiate(
        const std::array<float, 4>& measurement,
        std::array<float, 8>& mean,
        std::array<std::array<float, 8>, 8>& cov
    );

    // Advance state by 1 step
    void predict(
        std::array<float, 8>& mean,
        std::array<std::array<float, 8>, 8>& cov,
        bool is_occluded = false
    );

    // Update state with measurement [xc, yc, a, h]
    void update(
        std::array<float, 8>& mean,
        std::array<std::array<float, 8>, 8>& cov,
        const std::array<float, 4>& measurement
    );

    // Project state to measurement space
    void project(
        const std::array<float, 8>& mean,
        const std::array<std::array<float, 8>, 8>& cov,
        std::array<float, 4>& proj_mean,
        std::array<std::array<float, 4>, 4>& proj_cov
    );

    // Compute squared Mahalanobis distance
    float computeMahalanobis(
        const std::array<float, 8>& mean,
        const std::array<std::array<float, 8>, 8>& cov,
        const std::array<float, 4>& measurement
    );

private:
    float m_std_pos;
    float m_std_vel;
    float m_damping;
};

} // namespace ZedDigital

#endif // KALMAN_FILTER_HPP
