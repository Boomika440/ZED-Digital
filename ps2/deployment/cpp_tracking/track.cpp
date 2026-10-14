#include "track.hpp"
#include <cmath>
#include <algorithm>

namespace ZedDigital {

int CppTrack::s_count = 0;

CppTrack::CppTrack(
    const std::array<float, 8>& initial_mean,
    const std::array<std::array<float, 8>, 8>& initial_cov,
    float initial_score,
    const std::vector<float>& feature,
    int min_hits,
    int max_lost,
    float ema_alpha_val
)
    : state(TrackState::Tentative),
      mean(initial_mean),
      cov(initial_cov),
      score(initial_score),
      hits(1),
      age(1),
      time_since_update(0),
      hit_streak(1),
      min_hits_to_confirm(min_hits),
      max_lost_time(max_lost),
      ema_alpha(ema_alpha_val),
      is_occluded(false),
      overlap_ratio(0.0f)
{
    track_id = ++s_count;
    if (!feature.empty()) {
        smooth_feature = feature;
        last_clean_feature = feature;
    }
}

BoundingBox CppTrack::getBBox() const {
    float xc = mean[0];
    float yc = mean[1];
    float a = mean[2];
    float h = mean[3];
    float w = a * h;

    BoundingBox box;
    box.x1 = xc - w * 0.5f;
    box.y1 = yc - h * 0.5f;
    box.x2 = xc + w * 0.5f;
    box.y2 = yc + h * 0.5f;
    return box;
}

std::array<float, 4> CppTrack::getMeasurement() const {
    return {mean[0], mean[1], mean[2], mean[3]};
}

void CppTrack::predict(KalmanBoxFilter& kf) {
    kf.predict(mean, cov, is_occluded || (state == TrackState::Lost));
    age++;
    if (time_since_update > 0) {
        hit_streak = 0;
    }
    time_since_update++;
}

void CppTrack::update(
    KalmanBoxFilter& kf,
    const std::array<float, 4>& measurement,
    float new_score,
    const std::vector<float>& new_feature
) {
    kf.update(mean, cov, measurement);
    score = new_score;
    hits++;
    time_since_update = 0;
    hit_streak++;

    if (state == TrackState::Tentative && hits >= min_hits_to_confirm) {
        state = TrackState::Confirmed;
    } else if (state == TrackState::Lost) {
        state = TrackState::Confirmed; // Re-associated
    }

    if (!new_feature.empty() && !is_occluded && overlap_ratio < 0.25f) {
        if (smooth_feature.empty()) {
            smooth_feature = new_feature;
        } else {
            for (size_t i = 0; i < smooth_feature.size() && i < new_feature.size(); ++i) {
                smooth_feature[i] = ema_alpha * smooth_feature[i] + (1.0f - ema_alpha) * new_feature[i];
            }
        }
        last_clean_feature = smooth_feature;
    }
}

void CppTrack::markMissed() {
    if (state == TrackState::Tentative) {
        state = TrackState::Deleted;
    } else if (state == TrackState::Confirmed) {
        if (time_since_update > 1) {
            state = TrackState::Lost;
        }
    }

    if (time_since_update > max_lost_time) {
        state = TrackState::Deleted;
    }
}

} // namespace ZedDigital
