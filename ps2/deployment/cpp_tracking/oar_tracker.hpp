#ifndef OAR_TRACKER_HPP
#define OAR_TRACKER_HPP

#include <vector>
#include "types.hpp"
#include "kalman_filter.hpp"
#include "track.hpp"
#include "hungarian.hpp"

namespace ZedDigital {

class OcclusionAwareTracker {
public:
    OcclusionAwareTracker(
        float track_high_thresh = 0.60f,
        float track_low_thresh = 0.15f,
        float new_track_thresh = 0.70f,
        float match_thresh_s1 = 0.75f,
        float match_thresh_s2 = 0.55f,
        float reassoc_thresh = 0.35f,
        float lambda_app = 0.40f,
        int min_hits = 3,
        int max_lost = 60
    );

    void reset();
    std::vector<CppTrack> update(const std::vector<Detection>& detections);

private:
    float cosineDistance(const std::vector<float>& a, const std::vector<float>& b);
    void detectInterTrackOcclusions(std::vector<CppTrack>& active_tracks);

    float m_track_high_thresh;
    float m_track_low_thresh;
    float m_new_track_thresh;
    float m_match_thresh_s1;
    float m_match_thresh_s2;
    float m_reassoc_thresh;
    float m_lambda_app;
    int m_min_hits;
    int m_max_lost;

    KalmanBoxFilter m_kf;
    std::vector<CppTrack> m_tracks;
};

} // namespace ZedDigital

#endif // OAR_TRACKER_HPP
