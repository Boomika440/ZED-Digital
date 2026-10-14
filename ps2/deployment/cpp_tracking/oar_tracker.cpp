#include "oar_tracker.hpp"
#include <unordered_set>
#include <cmath>

namespace ZedDigital {

OcclusionAwareTracker::OcclusionAwareTracker(
    float track_high_thresh,
    float track_low_thresh,
    float new_track_thresh,
    float match_thresh_s1,
    float match_thresh_s2,
    float reassoc_thresh,
    float lambda_app,
    int min_hits,
    int max_lost
)
    : m_track_high_thresh(track_high_thresh),
      m_track_low_thresh(track_low_thresh),
      m_new_track_thresh(new_track_thresh),
      m_match_thresh_s1(match_thresh_s1),
      m_match_thresh_s2(match_thresh_s2),
      m_reassoc_thresh(reassoc_thresh),
      m_lambda_app(lambda_app),
      m_min_hits(min_hits),
      m_max_lost(max_lost)
{
}

void OcclusionAwareTracker::reset() {
    m_tracks.clear();
    CppTrack::resetCounter();
}

float OcclusionAwareTracker::cosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 1.0f;
    }
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-6f) return 1.0f;
    float sim = dot / denom;
    return std::clamp(1.0f - sim, 0.0f, 2.0f);
}

void OcclusionAwareTracker::detectInterTrackOcclusions(std::vector<CppTrack>& active_tracks) {
    const size_t n = active_tracks.size();
    for (size_t i = 0; i < n; ++i) {
        float max_overlap = 0.0f;
        BoundingBox box_i = active_tracks[i].getBBox();
        float area_i = std::max(1e-4f, box_i.area());

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            BoundingBox box_j = active_tracks[j].getBBox();
            float ix1 = std::max(box_i.x1, box_j.x1);
            float iy1 = std::max(box_i.y1, box_j.y1);
            float ix2 = std::min(box_i.x2, box_j.x2);
            float iy2 = std::min(box_i.y2, box_j.y2);
            if (ix2 > ix1 && iy2 > iy1) {
                float iarea = (ix2 - ix1) * (iy2 - iy1);
                float overlap = iarea / area_i;
                if (overlap > max_overlap) max_overlap = overlap;
            }
        }
        active_tracks[i].is_occluded = (max_overlap > 0.35f);
        active_tracks[i].overlap_ratio = max_overlap;
    }
}

std::vector<CppTrack> OcclusionAwareTracker::update(const std::vector<Detection>& detections) {
    // 1. Predict
    for (auto& trk : m_tracks) {
        trk.predict(m_kf);
    }

    // 2. Detect mutual occlusion
    detectInterTrackOcclusions(m_tracks);

    // 3. Partition detections
    std::vector<int> dets_high_indices;
    std::vector<int> dets_low_indices;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].score >= m_track_high_thresh) {
            dets_high_indices.push_back(static_cast<int>(i));
        } else if (detections[i].score >= m_track_low_thresh) {
            dets_low_indices.push_back(static_cast<int>(i));
        }
    }

    // Partition tracks
    std::vector<int> active_track_indices;
    std::vector<int> lost_track_indices;
    std::vector<int> tentative_track_indices;

    for (size_t i = 0; i < m_tracks.size(); ++i) {
        if (m_tracks[i].state == TrackState::Confirmed) {
            active_track_indices.push_back(static_cast<int>(i));
        } else if (m_tracks[i].state == TrackState::Lost) {
            lost_track_indices.push_back(static_cast<int>(i));
        } else if (m_tracks[i].state == TrackState::Tentative) {
            tentative_track_indices.push_back(static_cast<int>(i));
        }
    }

    // -------------------------------------------------------------
    // STAGE 1: Active Tracks vs. High-Score Detections
    // -------------------------------------------------------------
    std::unordered_set<int> matched_active;
    std::unordered_set<int> matched_dets_high;

    if (!active_track_indices.empty() && !dets_high_indices.empty()) {
        std::vector<std::vector<float>> cost_s1(
            active_track_indices.size(),
            std::vector<float>(dets_high_indices.size(), 0.0f)
        );

        for (size_t i = 0; i < active_track_indices.size(); ++i) {
            int t_idx = active_track_indices[i];
            BoundingBox t_box = m_tracks[t_idx].getBBox();

            for (size_t j = 0; j < dets_high_indices.size(); ++j) {
                int d_idx = dets_high_indices[j];
                float iou = BoundingBox::iou(t_box, detections[d_idx].box);
                float motion_cost = 1.0f - iou;
                float app_cost = 1.0f;

                if (!detections[d_idx].feature.empty()) {
                    const auto& ref_feat = !m_tracks[t_idx].last_clean_feature.empty() ?
                                           m_tracks[t_idx].last_clean_feature : m_tracks[t_idx].smooth_feature;
                    app_cost = cosineDistance(ref_feat, detections[d_idx].feature);
                }

                float total_cost = m_lambda_app * app_cost + (1.0f - m_lambda_app) * motion_cost;
                if (iou <= 0.0f && app_cost > 0.20f) total_cost = 1e5f; // Spatial gate
                cost_s1[i][j] = total_cost;
            }
        }

        auto matches = HungarianAlgorithm::solve(cost_s1, m_match_thresh_s1);
        for (const auto& match : matches) {
            int t_idx = active_track_indices[match.first];
            int d_idx = dets_high_indices[match.second];
            const auto& det = detections[d_idx];
            m_tracks[t_idx].update(
                m_kf,
                {det.box.cx(), det.box.cy(), det.box.aspect_ratio(), det.box.height()},
                det.score,
                det.feature
            );
            matched_active.insert(t_idx);
            matched_dets_high.insert(d_idx);
        }
    }

    // -------------------------------------------------------------
    // STAGE 2: Remaining Active Tracks vs. Low-Score Detections (Occlusion Recovery)
    // -------------------------------------------------------------
    std::vector<int> unmatched_active;
    for (int t_idx : active_track_indices) {
        if (matched_active.find(t_idx) == matched_active.end()) {
            unmatched_active.push_back(t_idx);
        }
    }

    if (!unmatched_active.empty() && !dets_low_indices.empty()) {
        std::vector<std::vector<float>> cost_s2(
            unmatched_active.size(),
            std::vector<float>(dets_low_indices.size(), 0.0f)
        );

        for (size_t i = 0; i < unmatched_active.size(); ++i) {
            int t_idx = unmatched_active[i];
            BoundingBox t_box = m_tracks[t_idx].getBBox();
            for (size_t j = 0; j < dets_low_indices.size(); ++j) {
                int d_idx = dets_low_indices[j];
                cost_s2[i][j] = 1.0f - BoundingBox::iou(t_box, detections[d_idx].box);
            }
        }

        auto matches = HungarianAlgorithm::solve(cost_s2, m_match_thresh_s2);
        for (const auto& match : matches) {
            int t_idx = unmatched_active[match.first];
            int d_idx = dets_low_indices[match.second];
            const auto& det = detections[d_idx];
            m_tracks[t_idx].update(
                m_kf,
                {det.box.cx(), det.box.cy(), det.box.aspect_ratio(), det.box.height()},
                det.score,
                {}
            );
            matched_active.insert(t_idx);
        }
    }

    // Mark remaining active as missed
    for (int t_idx : active_track_indices) {
        if (matched_active.find(t_idx) == matched_active.end()) {
            m_tracks[t_idx].markMissed();
        }
    }

    // -------------------------------------------------------------
    // STAGE 3: Remaining High-Score Detections vs. Lost Track Bank (Re-Association)
    // -------------------------------------------------------------
    std::vector<int> remain_dets_high;
    for (int d_idx : dets_high_indices) {
        if (matched_dets_high.find(d_idx) == matched_dets_high.end()) {
            remain_dets_high.push_back(d_idx);
        }
    }

    std::unordered_set<int> matched_lost;
    if (!lost_track_indices.empty() && !remain_dets_high.empty()) {
        std::vector<std::vector<float>> cost_s3(
            lost_track_indices.size(),
            std::vector<float>(remain_dets_high.size(), 0.0f)
        );

        for (size_t i = 0; i < lost_track_indices.size(); ++i) {
            int l_idx = lost_track_indices[i];
            BoundingBox l_box = m_tracks[l_idx].getBBox();
            for (size_t j = 0; j < remain_dets_high.size(); ++j) {
                int d_idx = remain_dets_high[j];
                float app_dist = cosineDistance(m_tracks[l_idx].smooth_feature, detections[d_idx].feature);
                float motion_dist = 1.0f - BoundingBox::iou(l_box, detections[d_idx].box);
                cost_s3[i][j] = 0.70f * app_dist + 0.30f * motion_dist;
            }
        }

        auto matches = HungarianAlgorithm::solve(cost_s3, m_reassoc_thresh);
        for (const auto& match : matches) {
            int l_idx = lost_track_indices[match.first];
            int d_idx = remain_dets_high[match.second];
            const auto& det = detections[d_idx];
            m_tracks[l_idx].update(
                m_kf,
                {det.box.cx(), det.box.cy(), det.box.aspect_ratio(), det.box.height()},
                det.score,
                det.feature
            );
            matched_lost.insert(l_idx);
            matched_dets_high.insert(d_idx);
        }
    }

    for (int l_idx : lost_track_indices) {
        if (matched_lost.find(l_idx) == matched_lost.end()) {
            m_tracks[l_idx].markMissed();
        }
    }

    // -------------------------------------------------------------
    // STAGE 4: Spawn New Tracks from Remaining High Detections
    // -------------------------------------------------------------
    for (int d_idx : dets_high_indices) {
        if (matched_dets_high.find(d_idx) == matched_dets_high.end()) {
            const auto& det = detections[d_idx];
            if (det.score >= m_new_track_thresh) {
                std::array<float, 8> init_mean;
                std::array<std::array<float, 8>, 8> init_cov;
                m_kf.initiate(
                    {det.box.cx(), det.box.cy(), det.box.aspect_ratio(), det.box.height()},
                    init_mean,
                    init_cov
                );

                CppTrack new_trk(
                    init_mean,
                    init_cov,
                    det.score,
                    det.feature,
                    m_min_hits,
                    m_max_lost
                );
                m_tracks.push_back(new_trk);
            }
        }
    }

    // Purge deleted tracks
    m_tracks.erase(
        std::remove_if(m_tracks.begin(), m_tracks.end(), [](const CppTrack& t) {
            return t.state == TrackState::Deleted;
        }),
        m_tracks.end()
    );

    // Return confirmed active tracks
    std::vector<CppTrack> output;
    for (const auto& trk : m_tracks) {
        if (trk.state == TrackState::Confirmed && trk.time_since_update == 0) {
            output.push_back(trk);
        }
    }
    return output;
}

} // namespace ZedDigital
