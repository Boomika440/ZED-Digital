#include "oar_tracker.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

int main(int argc, char** argv) {
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Zed Digital - C++ Real-Time Multi-Object Tracker (OAR)" << std::endl;
    std::cout << "  Problem Statement 2: Dense Occlusion Gate Surveillance" << std::endl;
    std::cout << "===========================================================" << std::endl;

    const int num_pedestrians = 25;
    const int num_frames = 200;

    ZedDigital::OcclusionAwareTracker tracker(
        0.60f, // high score thresh
        0.15f, // low score thresh
        0.70f, // new track thresh
        0.75f, // stage 1 thresh
        0.55f, // stage 2 thresh
        0.35f, // reassoc thresh
        0.40f, // appearance weight
        3,     // min hits
        60     // max lost
    );

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_noise(-2.0f, 2.0f);
    std::uniform_real_distribution<float> dist_score(0.70f, 0.98f);
    std::uniform_real_distribution<float> dist_occ(0.0f, 1.0f);

    std::cout << "\n[C++ Tracker] Simulating 200 frames of dense choke-point gate traffic ("
              << num_pedestrians << " targets)..." << std::endl;

    std::vector<double> frame_latencies_ms;
    frame_latencies_ms.reserve(num_frames);

    int total_confirmed_tracks = 0;

    for (int frame = 1; frame <= num_frames; ++frame) {
        // Synthesize detections
        std::vector<ZedDigital::Detection> detections;
        for (int p = 0; p < num_pedestrians; ++p) {
            // Simulate 15% random drop due to occlusion
            if (dist_occ(rng) < 0.15f) continue;

            float cx = 400.0f + p * 45.0f + frame * 1.5f + dist_noise(rng);
            float cy = 300.0f + (p % 4) * 80.0f + dist_noise(rng);
            float w = 55.0f;
            float h = 150.0f;

            ZedDigital::Detection det;
            det.box.x1 = cx - w * 0.5f;
            det.box.y1 = cy - h * 0.5f;
            det.box.x2 = cx + w * 0.5f;
            det.box.y2 = cy + h * 0.5f;
            det.score = dist_score(rng);

            // Dummy 128-D embedding with identity seed
            det.feature.resize(128);
            std::mt19937 feat_rng(1000 + p);
            for (int k = 0; k < 128; ++k) {
                det.feature[k] = static_cast<float>(feat_rng() % 1000) / 1000.0f - 0.5f;
            }

            detections.push_back(det);
        }

        // Measure tracking update latency
        auto t0 = std::chrono::high_resolution_clock::now();
        auto active_tracks = tracker.update(detections);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        frame_latencies_ms.push_back(ms);
        total_confirmed_tracks += static_cast<int>(active_tracks.size());

        if (frame % 50 == 0 || frame == num_frames) {
            std::cout << "  Frame [" << frame << "/" << num_frames << "] -> "
                      << active_tracks.size() << " active tracks, latency: "
                      << ms << " ms" << std::endl;
        }
    }

    double sum_ms = 0.0;
    for (double v : frame_latencies_ms) sum_ms += v;
    double mean_ms = sum_ms / num_frames;
    double fps = 1000.0 / std::max(1e-4, mean_ms);

    std::cout << "\n===========================================================" << std::endl;
    std::cout << "  C++ REAL-TIME PERFORMANCE PROFILE SUMMARY" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Simulated Crowd Volume : " << num_pedestrians << " pedestrians / frame" << std::endl;
    std::cout << "  Average Frame Latency  : " << mean_ms << " ms" << std::endl;
    std::cout << "  Processing Throughput  : " << fps << " FPS" << std::endl;
    std::cout << "  Real-time Requirement  : " << (mean_ms < 15.0 ? "PASSED (Bounded Latency < 15ms)" : "NEEDS OPTIMIZATION") << std::endl;
    std::cout << "===========================================================" << std::endl;

    return 0;
}
