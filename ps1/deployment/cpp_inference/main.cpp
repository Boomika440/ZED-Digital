#include "reid_engine.hpp"
#include <iostream>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Zed Digital - Open-Set Re-ID Real-Time Edge Engine (C++)" << std::endl;
    std::cout << "  Problem Statement 1: Low Inter-Class Variance Deployment" << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::string model_path = (argc > 1) ? argv[1] : "./weights/open_set_reid.onnx";
    bool use_cuda = (argc > 2 && std::string(argv[2]) == "--cuda");

    // Initialize engine
    ZedDigital::OpenSetReIDEngine engine(
        0.72f, // Calibrated threshold for FAR = 1%
        0.08f, // Ambiguity margin
        512    // Feature dimension
    );

    std::cout << "\n[C++] Loading model from: " << model_path << std::endl;
    bool loaded = engine.loadModel(model_path, use_cuda);
    if (!loaded) {
        std::cout << "[C++] Notice: Proceeding in benchmark simulation mode." << std::endl;
    }

    // 1. Simulate Dynamic Gallery Enrollment
    std::cout << "\n[C++] Enrolling 50 employees into dynamic gallery database..." << std::endl;
    for (int pid = 1; pid <= 50; ++pid) {
        cv::Mat dummy_crop = cv::Mat(256, 128, CV_8UC3, cv::Scalar(40, 42, 60)); // Navy uniform base
        // Micro-attribute accent
        cv::circle(dummy_crop, cv::Point(64, 50), 6, cv::Scalar((pid * 25) % 255, (pid * 40) % 255, 200), -1);
        engine.enrollIdentity(pid, dummy_crop);
    }
    std::cout << "[C++] Enrolled gallery size: " << engine.getGallerySize() << " identities." << std::endl;

    // 2. Simulate Known Employee Return (Should Match)
    std::cout << "\n[C++] Testing Known Employee (Enrolled ID: 15 returning to camera)..." << std::endl;
    cv::Mat known_query = cv::Mat(256, 128, CV_8UC3, cv::Scalar(42, 44, 62));
    cv::circle(known_query, cv::Point(64, 50), 6, cv::Scalar((15 * 25) % 255, (15 * 40) % 255, 200), -1);

    auto result_known = engine.match(known_query);
    std::cout << "  --> Match Decision : " << (result_known.is_accepted ? "ACCEPTED" : "REJECTED") << std::endl;
    std::cout << "  --> Predicted PID  : " << result_known.matched_pid << std::endl;
    std::cout << "  --> Top-1 Cosine   : " << result_known.top1_score << std::endl;
    std::cout << "  --> Calibrated Conf: " << result_known.confidence << std::endl;
    std::cout << "  --> Engine Rationale: " << result_known.reason << std::endl;

    // 3. Simulate Unenrolled Unknown Impostor in Identical Uniform (Must Reject!)
    std::cout << "\n[C++] Testing Unknown Impostor (Unenrolled person in same uniform)..." << std::endl;
    cv::Mat unknown_query = cv::Mat(256, 128, CV_8UC3, cv::Scalar(40, 42, 60)); // Same navy uniform
    cv::circle(unknown_query, cv::Point(64, 50), 6, cv::Scalar(10, 10, 240), -1); // Different badge

    auto result_unknown = engine.match(unknown_query);
    std::cout << "  --> Match Decision : " << (result_unknown.is_accepted ? "ACCEPTED" : "REJECTED") << std::endl;
    std::cout << "  --> Predicted PID  : " << result_unknown.matched_pid << " (-1 = UNKNOWN)" << std::endl;
    std::cout << "  --> Top-1 Cosine   : " << result_unknown.top1_score << std::endl;
    std::cout << "  --> Calibrated Conf: " << result_unknown.confidence << std::endl;
    std::cout << "  --> Engine Rationale: " << result_unknown.reason << std::endl;

    // 4. Real-Time Latency Benchmark (100 Iterations)
    std::cout << "\n[C++] Profiling End-to-End Matching Latency (100 runs)..." << std::endl;
    const int num_runs = 100;
    std::vector<double> latencies_ms;
    latencies_ms.reserve(num_runs);

    for (int i = 0; i < num_runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = engine.match(known_query);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies_ms.push_back(ms);
    }

    double total_ms = 0.0;
    for (double lat : latencies_ms) total_ms += lat;
    double mean_ms = total_ms / num_runs;
    double fps = 1000.0 / mean_ms;

    std::cout << "===========================================================" << std::endl;
    std::cout << "  C++ REAL-TIME PERFORMANCE PROFILE" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Average Latency : " << mean_ms << " ms / person crop" << std::endl;
    std::cout << "  Throughput Rate : " << fps << " FPS" << std::endl;
    std::cout << "  Constraint Check: " << (mean_ms < 15.0 ? "PASSED (Real-Time < 15ms)" : "NEEDS GPU") << std::endl;
    std::cout << "===========================================================" << std::endl;

    return 0;
}
