#ifndef TRACK_HPP
#define TRACK_HPP

#include <vector>
#include <array>
#include "types.hpp"
#include "kalman_filter.hpp"

namespace ZedDigital {

class CppTrack {
public:
    static int s_count;
    static void resetCounter() { s_count = 0; }

    CppTrack(
        const std::array<float, 8>& mean,
        const std::array<std::array<float, 8>, 8>& cov,
        float score,
        const std::vector<float>& feature = {},
        int min_hits = 3,
        int max_lost = 60,
        float ema_alpha = 0.90f
    );

    int track_id;
    TrackState state;
    std::array<float, 8> mean;
    std::array<std::array<float, 8>, 8> cov;

    float score;
    int hits;
    int age;
    int time_since_update;
    int hit_streak;

    int min_hits_to_confirm;
    int max_lost_time;
    float ema_alpha;

    bool is_occluded;
    float overlap_ratio;

    std::vector<float> smooth_feature;
    std::vector<float> last_clean_feature;

    BoundingBox getBBox() const;
    std::array<float, 4> getMeasurement() const;

    void predict(KalmanBoxFilter& kf);
    void update(
        KalmanBoxFilter& kf,
        const std::array<float, 4>& measurement,
        float new_score,
        const std::vector<float>& new_feature = {}
    );
    void markMissed();
};

} // namespace ZedDigital

#endif // TRACK_HPP
