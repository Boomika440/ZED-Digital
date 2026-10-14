#ifndef REID_ENGINE_HPP
#define REID_ENGINE_HPP

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

namespace ZedDigital {

struct MatchResult {
    bool is_accepted;
    int matched_pid;       // Enrolled identity ID, or -1 if rejected as unknown
    float confidence;      // Calibrated probability [0.0, 1.0]
    float top1_score;      // Raw cosine similarity
    float top2_score;      // Second highest match
    std::string reason;    // Status or rejection rationale
};

struct GalleryRecord {
    int pid;
    std::vector<float> embedding; // L2 normalized 512-D vector
    float weibull_shape;          // EVT shape parameter
    float weibull_scale;          // EVT scale parameter
};

class OpenSetReIDEngine {
public:
    OpenSetReIDEngine(
        float decision_threshold = 0.72f,
        float ambiguity_margin = 0.08f,
        int embedding_dim = 512
    );
    ~OpenSetReIDEngine();

    // Initialize model weights and runtime session
    bool loadModel(const std::string& model_path, bool use_cuda = false);

    // Preprocess BGR bounding box to NCHW normalized float buffer
    void preprocess(const cv::Mat& bgr_crop, std::vector<float>& input_tensor, int target_h = 256, int target_w = 128);

    // Extract L2-normalized 512-D discriminative embedding from person crop
    std::vector<float> extractEmbedding(const cv::Mat& bgr_crop);

    // Dynamic Gallery Management (No assumption of fixed identity count)
    void enrollIdentity(int pid, const cv::Mat& bgr_crop);
    void enrollEmbedding(int pid, const std::vector<float>& embedding);
    void clearGallery();
    size_t getGallerySize() const;

    // Match query crop against dynamic gallery with EVT calibration & ambiguity rejection
    MatchResult match(const cv::Mat& query_crop);

    // Set decision boundary parameters
    void setThreshold(float threshold) { m_threshold = threshold; }
    void setAmbiguityMargin(float margin) { m_ambiguity_margin = margin; }

private:
    float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
    void normalizeL2(std::vector<float>& vec);
    float computeEVTProbability(float distance, float shape, float scale);

    float m_threshold;
    float m_ambiguity_margin;
    int m_embedding_dim;
    std::vector<GalleryRecord> m_gallery;

    // Model runtime handles
    bool m_is_loaded;
    cv::dnn::Net m_dnn_net; // OpenCV DNN backend for cross-platform edge execution
};

} // namespace ZedDigital

#endif // REID_ENGINE_HPP
