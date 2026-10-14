#ifndef HUNGARIAN_HPP
#define HUNGARIAN_HPP

#include <vector>
#include <utility>

namespace ZedDigital {

class HungarianAlgorithm {
public:
    // Solves minimum cost bipartite matching.
    // Cost matrix: [num_rows x num_cols]
    // Returns: vector of matched pairs: (row, col)
    static std::vector<std::pair<int, int>> solve(
        const std::vector<std::vector<float>>& cost_matrix,
        float max_cost_threshold
    );
};

} // namespace ZedDigital

#endif // HUNGARIAN_HPP
