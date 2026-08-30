// main.cpp
// Days 1-3 merged: VectorItem/VectorStore (brute-force, cosine similarity)
//                   + KDTree (pruned k-NN, Euclidean distance)

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <limits>

// ============================================================
// Day 1-2: VectorItem, distance metrics, VectorStore (brute-force)
// ============================================================

struct VectorItem {
    int id;
    std::string label;
    std::vector<float> values;

    VectorItem(int id_, std::string label_, std::vector<float> values_)
        : id(id_), label(std::move(label_)), values(std::move(values_)) {}
};

float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float manhattanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += std::fabs(a[i] - b[i]);
    }
    return sum;
}

float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float magnitude(const std::vector<float>& a) {
    return std::sqrt(dotProduct(a, a));
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float magA = magnitude(a);
    float magB = magnitude(b);
    if (magA == 0.0f || magB == 0.0f) return 0.0f;
    return dotProduct(a, b) / (magA * magB);
}

struct SearchResult {
    int id;
    std::string label;
    float score;
};

class VectorStore {
private:
    std::vector<VectorItem> items;
    int nextId = 0;

public:
    int insert(const std::string& label, const std::vector<float>& values) {
        int id = nextId++;
        items.emplace_back(id, label, values);
        return id;
    }

    size_t size() const { return items.size(); }

    std::vector<SearchResult> bruteForceSearch(const std::vector<float>& query, int k) const {
        std::vector<SearchResult> results;
        results.reserve(items.size());

        for (const auto& item : items) {
            float score = cosineSimilarity(query, item.values);
            results.push_back({item.id, item.label, score});
        }

        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.score > b.score;
                  });

        if (results.size() > static_cast<size_t>(k)) {
            results.resize(k);
        }
        return results;
    }
};

// ============================================================
// Day 3: KDTree (Euclidean distance, pruned k-NN search)
// ============================================================

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

    std::unique_ptr<KDNode> buildRecursive(std::vector<KDPoint>& points, int start, int end, int depth) {
        if (start >= end) return nullptr;

        int dim = depth % dimensions;
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

    struct BestMatch {
        float distSq;
        KDPoint point;
    };

    void searchRecursive(const KDNode* node, const std::vector<float>& query,
                          std::vector<BestMatch>& best, int k) const {
        if (!node) return;

        float distSq = squaredEuclidean(query, node->point.values);

        best.push_back({distSq, node->point});
        std::sort(best.begin(), best.end(),
                  [](const BestMatch& a, const BestMatch& b) { return a.distSq < b.distSq; });
        if (best.size() > static_cast<size_t>(k)) best.resize(k);

        int dim = node->splitDim;
        float diff = query[dim] - node->point.values[dim];

        const KDNode* nearSide = (diff < 0) ? node->left.get() : node->right.get();
        const KDNode* farSide  = (diff < 0) ? node->right.get() : node->left.get();

        searchRecursive(nearSide, query, best, k);

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

// ============================================================
// main(): demo both structures on the same toy dataset
// ============================================================

int main() {
    // ---- Brute-force (VectorStore) ----
    VectorStore store;
    store.insert("CS",     {0.9f, 0.8f, 0.1f, 0.1f});
    store.insert("Math",   {0.85f, 0.75f, 0.15f, 0.05f});
    store.insert("Food",   {0.1f, 0.1f, 0.9f, 0.8f});
    store.insert("Sports", {0.2f, 0.1f, 0.1f, 0.9f});
    store.insert("Art",    {0.3f, 0.6f, 0.4f, 0.2f});

    std::vector<float> query = {0.88f, 0.79f, 0.12f, 0.08f};

    auto bruteResults = store.bruteForceSearch(query, 2);
    std::cout << "[Brute-Force / cosine] Top " << bruteResults.size() << " matches:\n";
    for (const auto& r : bruteResults) {
        std::cout << "  [" << r.id << "] " << r.label
                  << "  (cosine similarity: " << r.score << ")\n";
    }

    // ---- KD-Tree ----
    std::vector<KDPoint> points = {
        {0, "CS",     {0.9f, 0.8f, 0.1f, 0.1f}},
        {1, "Math",   {0.85f, 0.75f, 0.15f, 0.05f}},
        {2, "Food",   {0.1f, 0.1f, 0.9f, 0.8f}},
        {3, "Sports", {0.2f, 0.1f, 0.1f, 0.9f}},
        {4, "Art",    {0.3f, 0.6f, 0.4f, 0.2f}},
    };

    KDTree tree;
    tree.build(points);

    auto kdResults = tree.kNearest(query, 2);
    std::cout << "\n[KD-Tree / euclidean] Top " << kdResults.size() << " matches:\n";
    for (const auto& r : kdResults) {
        std::cout << "  [" << r.id << "] " << r.label << "\n";
    }

    return 0;
}