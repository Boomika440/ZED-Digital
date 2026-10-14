#include "hungarian.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

namespace ZedDigital {

std::vector<std::pair<int, int>> HungarianAlgorithm::solve(
    const std::vector<std::vector<float>>& cost_matrix,
    float max_cost_threshold
) {
    std::vector<std::pair<int, int>> matches;
    if (cost_matrix.empty() || cost_matrix[0].empty()) {
        return matches;
    }

    const int n = static_cast<int>(cost_matrix.size());
    const int m = static_cast<int>(cost_matrix[0].size());
    const int dim = std::max(n, m);

    const float INF = 1e9f;
    std::vector<std::vector<float>> cost(dim + 1, std::vector<float>(dim + 1, 0.0f));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cost[i + 1][j + 1] = cost_matrix[i][j];
        }
    }

    std::vector<float> u(dim + 1, 0.0f), v(dim + 1, 0.0f);
    std::vector<int> p(dim + 1, 0), way(dim + 1, 0);

    for (int i = 1; i <= dim; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(dim + 1, INF);
        std::vector<char> used(dim + 1, false);

        do {
            used[j0] = true;
            int i0 = p[j0];
            float delta = INF;
            int j1 = 0;

            for (int j = 1; j <= dim; ++j) {
                if (!used[j]) {
                    float cur = cost[i0][j] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= dim; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    // Extract assignment
    for (int j = 1; j <= m; ++j) {
        int i = p[j] - 1;
        int col = j - 1;
        if (i >= 0 && i < n && col < m) {
            if (cost_matrix[i][col] <= max_cost_threshold) {
                matches.push_back({i, col});
            }
        }
    }

    return matches;
}

} // namespace ZedDigital
