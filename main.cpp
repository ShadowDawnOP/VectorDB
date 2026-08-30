// main.cpp
// Day 3: KD-Tree implementation (standalone class, alongside VectorStore/BruteForce from Days 1-2)

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>

struct KDPoint {
    int id;
    std::string label;
    std::vector<float> values;
};

struct KDNode {
    KDPoint point;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;
    int splitDim;

    KDNode(KDPoint p, int dim) : point(std::move(p)), splitDim(dim) {}
};

class KDTree {
private:
    std::unique_ptr<KDNode> root;
    int dimensions = 0;

    static float squaredEuclidean(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    // Recursively build a balanced KD-Tree from a list of points.
    std::unique_ptr<KDNode> buildRecursive(std::vector<KDPoint>& points, int start, int end, int depth) {
        if (start >= end) return nullptr;

        int dim = depth % dimensions;

        // Sort the slice by this dimension's value, pick the median as the node.
        int mid = start + (end - start) / 2;
        std::nth_element(points.begin() + start, points.begin() + mid, points.begin() + end,
                          [dim](const KDPoint& a, const KDPoint& b) {
                              return a.values[dim] < b.values[dim];
                          });

        auto node = std::make_unique<KDNode>(points[mid], dim);
        node->left = buildRecursive(points, start, mid, depth + 1);
        node->right = buildRecursive(points, mid + 1, end, depth + 1);
        return node;
    }

    // Best-so-far tracking during search: (squared distance, point)
    struct BestMatch {
        float distSq;
        KDPoint point;
    };

    void searchRecursive(const KDNode* node, const std::vector<float>& query,
                          std::vector<BestMatch>& best, int k) const {
        if (!node) return;

        float distSq = squaredEuclidean(query, node->point.values);

        // Insert into best-list, keep it sorted, trim to size k
        best.push_back({distSq, node->point});
        std::sort(best.begin(), best.end(),
                  [](const BestMatch& a, const BestMatch& b) { return a.distSq < b.distSq; });
        if (best.size() > static_cast<size_t>(k)) best.resize(k);

        int dim = node->splitDim;
        float diff = query[dim] - node->point.values[dim];

        // Decide which side to search first (the side the query "belongs" to)
        const KDNode* nearSide = (diff < 0) ? node->left.get() : node->right.get();
        const KDNode* farSide  = (diff < 0) ? node->right.get() : node->left.get();

        searchRecursive(nearSide, query, best, k);

        // Pruning check: only explore the far side if it could possibly contain
        // something closer than our current worst "best" match.
        float worstBestDist = (best.size() < static_cast<size_t>(k))
                                   ? std::numeric_limits<float>::max()
                                   : best.back().distSq;
        if (diff * diff < worstBestDist) {
            searchRecursive(farSide, query, best, k);
        }
    }

public:
    void build(std::vector<KDPoint> points) {
        if (points.empty()) return;
        dimensions = static_cast<int>(points[0].values.size());
        root = buildRecursive(points, 0, static_cast<int>(points.size()), 0);
    }

    std::vector<KDPoint> kNearest(const std::vector<float>& query, int k) const {
        std::vector<BestMatch> best;
        searchRecursive(root.get(), query, best, k);

        std::vector<KDPoint> results;
        for (auto& b : best) results.push_back(b.point);
        return results;
    }
};

int main() {
    std::vector<KDPoint> points = {
        {0, "CS",     {0.9f, 0.8f, 0.1f, 0.1f}},
        {1, "Math",   {0.85f, 0.75f, 0.15f, 0.05f}},
        {2, "Food",   {0.1f, 0.1f, 0.9f, 0.8f}},
        {3, "Sports", {0.2f, 0.1f, 0.1f, 0.9f}},
        {4, "Art",    {0.3f, 0.6f, 0.4f, 0.2f}},
    };

    KDTree tree;
    tree.build(points);

    std::vector<float> query = {0.88f, 0.79f, 0.12f, 0.08f};
    auto results = tree.kNearest(query, 2);

    std::cout << "KD-Tree top " << results.size() << " nearest (by Euclidean distance):\n";
    for (const auto& r : results) {
        std::cout << "  [" << r.id << "] " << r.label << "\n";
    }

    return 0;
}