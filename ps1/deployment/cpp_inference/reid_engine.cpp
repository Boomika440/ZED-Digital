#include "reid_engine.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace ZedDigital {

OpenSetReIDEngine::OpenSetReIDEngine(float decision_threshold, float ambiguity_margin, int embedding_dim)
    : m_threshold(decision_threshold),
      m_ambiguity_margin(ambiguity_margin),
      m_embedding_dim(embedding_dim),
      m_is_loaded(false)
{
}

OpenSetReIDEngine::~OpenSetReIDEngine() {
}

bool OpenSetReIDEngine::loadModel(const std::string& model_path, bool use_cuda) {
    try {
        m_dnn_net = cv::dnn::readNetFromONNX(model_path);
        if (m_dnn_net.empty()) {
            std::cerr << "[C++ Engine] Failed to parse ONNX model from: " << model_path << std::endl;
            return false;
        }

        if (use_cuda) {
            m_dnn_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_dnn_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "[C++ Engine] Configured CUDA backend for OpenCV DNN." << std::endl;
        } else {
            m_dnn_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_dnn_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "[C++ Engine] Configured optimized CPU backend." << std::endl;
        }

        m_is_loaded = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[C++ Engine] Exception loading ONNX: " << e.what() << std::endl;
        return false;
    }
}

void OpenSetReIDEngine::preprocess(const cv::Mat& bgr_crop, std::vector<float>& input_tensor, int target_h, int target_w) {
    cv::Mat resized, rgb;
    cv::resize(bgr_crop, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // ImageNet mean & std
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};

    input_tensor.resize(3 * target_h * target_w);
    int plane_size = target_h * target_w;

    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
            int idx = y * target_w + x;
            input_tensor[0 * plane_size + idx] = ((pixel[0] / 255.0f) - mean[0]) / std_dev[0];
            input_tensor[1 * plane_size + idx] = ((pixel[1] / 255.0f) - mean[1]) / std_dev[1];
            input_tensor[2 * plane_size + idx] = ((pixel[2] / 255.0f) - mean[2]) / std_dev[2];
        }
    }
}

std::vector<float> OpenSetReIDEngine::extractEmbedding(const cv::Mat& bgr_crop) {
    std::vector<float> embedding(m_embedding_dim, 0.0f);

    if (m_is_loaded) {
        // Construct 4D blob: [1, 3, 256, 128]
        cv::Mat blob = cv::dnn::blobFromImage(
            bgr_crop,
            1.0 / 255.0,
            cv::Size(128, 256),
            cv::Scalar(0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0),
            true, // swapRB
            false // crop
        );

        m_dnn_net.setInput(blob);
        cv::Mat out = m_dnn_net.forward();

        const float* out_ptr = out.ptr<float>();
        for (int i = 0; i < m_embedding_dim; ++i) {
            embedding[i] = out_ptr[i];
        }
    } else {
        // Fallback synthetic pseudo-embedding for testing without model weight binary
        for (int i = 0; i < m_embedding_dim; ++i) {
            embedding[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }

    normalizeL2(embedding);
    return embedding;
}

void OpenSetReIDEngine::enrollIdentity(int pid, const cv::Mat& bgr_crop) {
    std::vector<float> emb = extractEmbedding(bgr_crop);
    enrollEmbedding(pid, emb);
}

void OpenSetReIDEngine::enrollEmbedding(int pid, const std::vector<float>& embedding) {
    GalleryRecord rec;
    rec.pid = pid;
    rec.embedding = embedding;
    normalizeL2(rec.embedding);
    rec.weibull_shape = 2.1f;
    rec.weibull_scale = 0.35f;
    m_gallery.push_back(rec);
}

void OpenSetReIDEngine::clearGallery() {
    m_gallery.clear();
}

size_t OpenSetReIDEngine::getGallerySize() const {
    return m_gallery.size();
}

void OpenSetReIDEngine::normalizeL2(std::vector<float>& vec) {
    float sum_sq = 0.0f;
    for (float v : vec) sum_sq += v * v;
    float norm = std::sqrt(sum_sq);
    if (norm < 1e-8f) norm = 1e-8f;
    for (float& v : vec) v /= norm;
}

float OpenSetReIDEngine::cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

float OpenSetReIDEngine::computeEVTProbability(float distance, float shape, float scale) {
    // Cumulative Weibull CDF: F(d) = 1 - exp(- (d / scale)^shape)
    // Probability of true match = 1 - F(d) = exp(- (d / scale)^shape)
    if (distance <= 0.0f) return 1.0f;
    float val = std::pow(distance / scale, shape);
    return std::exp(-val);
}

MatchResult OpenSetReIDEngine::match(const cv::Mat& query_crop) {
    MatchResult result;
    result.is_accepted = false;
    result.matched_pid = -1;
    result.confidence = 0.0f;
    result.top1_score = -1.0f;
    result.top2_score = -1.0f;
    result.reason = "Unmatched";

    if (m_gallery.empty()) {
        result.reason = "Empty gallery database";
        return result;
    }

    std::vector<float> query_emb = extractEmbedding(query_crop);

    // Compute similarities against all enrolled identities
    std::vector<std::pair<float, int>> similarities; // (score, index)
    similarities.reserve(m_gallery.size());

    for (size_t i = 0; i < m_gallery.size(); ++i) {
        float sim = cosineSimilarity(query_emb, m_gallery[i].embedding);
        similarities.push_back({sim, static_cast<int>(i)});
    }

    // Sort descending
    std::sort(similarities.begin(), similarities.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    int top1_idx = similarities[0].second;
    float top1_sim = similarities[0].first;
    float top2_sim = (similarities.size() > 1) ? similarities[1].first : -1.0f;

    result.top1_score = top1_sim;
    result.top2_score = top2_sim;

    // EVT Confidence Calibration
    float dist = std::max(0.0f, 1.0f - top1_sim);
    float conf = computeEVTProbability(dist, m_gallery[top1_idx].weibull_shape, m_gallery[top1_idx].weibull_scale);
    result.confidence = conf;

    // Decision Boundary 1: Verification Confidence Threshold
    if (conf < m_threshold) {
        result.is_accepted = false;
        result.matched_pid = -1;
        result.reason = "Rejected: Confidence below threshold (EVT Impostor Rejection)";
        return result;
    }

    // Decision Boundary 2: Ambiguity Margin (Rejects uniform collisions)
    if (m_ambiguity_margin > 0.0f && top2_sim > 0.0f) {
        float margin = top1_sim - top2_sim;
        if (margin < m_ambiguity_margin) {
            result.is_accepted = false;
            result.matched_pid = -1;
            result.reason = "Rejected: Ambiguity collision (query equally close to multiple identities)";
            return result;
        }
    }

    // Accepted true match
    result.is_accepted = true;
    result.matched_pid = m_gallery[top1_idx].pid;
    result.reason = "Accepted: Verified enrolled identity";
    return result;
}

} // namespace ZedDigital
