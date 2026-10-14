#ifndef TYPES_HPP
#define TYPES_HPP

#include <vector>
#include <algorithm>
#include <cmath>

namespace ZedDigital {

struct BoundingBox {
    float x1;
    float y1;
    float x2;
    float y2;

    float width() const { return std::max(0.0f, x2 - x1); }
    float height() const { return std::max(0.0f, y2 - y1); }
    float area() const { return width() * height(); }
    float cx() const { return x1 + width() * 0.5f; }
    float cy() const { return y1 + height() * 0.5f; }
    float aspect_ratio() const { return width() / std::max(1e-4f, height()); }

    static float iou(const BoundingBox& a, const BoundingBox& b) {
        float ix1 = std::max(a.x1, b.x1);
        float iy1 = std::max(a.y1, b.y1);
        float ix2 = std::min(a.x2, b.x2);
        float iy2 = std::min(a.y2, b.y2);
        float iw = std::max(0.0f, ix2 - ix1);
        float ih = std::max(0.0f, iy2 - iy1);
        float iarea = iw * ih;
        float uarea = a.area() + b.area() - iarea;
        return (uarea > 0.0f) ? (iarea / uarea) : 0.0f;
    }
};

struct Detection {
    BoundingBox box;
    float score;
    std::vector<float> feature; // L2 normalized appearance embedding
};

enum class TrackState {
    Tentative = 1,
    Confirmed = 2,
    Lost = 3,
    Deleted = 4
};

} // namespace ZedDigital

#endif // TYPES_HPP
