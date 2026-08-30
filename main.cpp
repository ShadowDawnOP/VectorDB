// main.cpp
// Days 1-6 merged:
//   - VectorItem/VectorStore (brute-force, cosine similarity)
//   - KDTree (pruned k-NN, Euclidean distance)
//   - HNSWIndex (multilayer graph: insert + search)
//   - Benchmark comparing all three on a synthetic dataset

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <limits>
#include <random>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <set>

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
// Day 5-6: HNSWIndex — insertion (Day 5) + search (Day 6)
// ============================================================

class HNSWIndex {
public:
    struct Node {
        int id;
        std::string label;
        std::vector<float> values;
        std::vector<std::vector<int>> neighbors; // neighbors[layer] = list of node ids
    };

    struct HNSWResult {
        int id;
        std::string label;
        float score; // cosine similarity, higher = better
    };

private:
    std::vector<Node> nodes;
    int entryPoint = -1;
    int maxLayer = -1;
    int M;                 // max connections per node per layer (2*M at layer 0)
    int efConstruction;    // candidate list size while building
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniform01;

    float distance(const std::vector<float>& a, const std::vector<float>& b) const {
        return 1.0f - cosineSimilarity(a, b);
    }

    int randomLevel() {
        double r = uniform01(rng);
        if (r <= 0.0) r = 1e-9;
        double levelLambda = 1.0 / std::log(1.0 * M);
        return static_cast<int>(-std::log(r) * levelLambda);
    }

    std::vector<std::pair<float,int>> searchLayer(const std::vector<float>& query,
                                                     std::vector<int> entryPoints,
                                                     int ef, int layer) const {
        std::unordered_set<int> visited;
        auto cmpMin = [](const std::pair<float,int>& a, const std::pair<float,int>& b) { return a.first > b.first; };
        auto cmpMax = [](const std::pair<float,int>& a, const std::pair<float,int>& b) { return a.first < b.first; };
        std::priority_queue<std::pair<float,int>, std::vector<std::pair<float,int>>, decltype(cmpMin)> candidates(cmpMin);
        std::priority_queue<std::pair<float,int>, std::vector<std::pair<float,int>>, decltype(cmpMax)> best(cmpMax);

        for (int ep : entryPoints) {
            float d = distance(query, nodes[ep].values);
            candidates.push({d, ep});
            best.push({d, ep});
            visited.insert(ep);
        }

        while (!candidates.empty()) {
            auto curPair = candidates.top();
            float curDist = curPair.first;
            int curId = curPair.second;
            candidates.pop();

            if (!best.empty() && curDist > best.top().first && best.size() >= static_cast<size_t>(ef)) {
                break;
            }

            if (static_cast<size_t>(curId) >= nodes.size() || layer >= (int)nodes[curId].neighbors.size()) continue;

            for (int neighborId : nodes[curId].neighbors[layer]) {
                if (visited.count(neighborId)) continue;
                visited.insert(neighborId);
                float d = distance(query, nodes[neighborId].values);

                if (best.size() < static_cast<size_t>(ef) || d < best.top().first) {
                    candidates.push({d, neighborId});
                    best.push({d, neighborId});
                    if (best.size() > static_cast<size_t>(ef)) best.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> result;
        while (!best.empty()) { result.push_back(best.top()); best.pop(); }
        std::reverse(result.begin(), result.end());
        return result;
    }

    std::vector<int> selectNeighborsSimple(std::vector<std::pair<float,int>> candidates, int maxM) const {
        std::sort(candidates.begin(), candidates.end());
        std::vector<int> result;
        for (int i = 0; i < std::min((int)candidates.size(), maxM); i++) {
            result.push_back(candidates[i].second);
        }
        return result;
    }

public:
    HNSWIndex(int M_ = 16, int efConstruction_ = 200, unsigned seed = 42)
        : M(M_), efConstruction(efConstruction_), rng(seed), uniform01(0.0f, 1.0f) {}

    void insert(const std::string& label, const std::vector<float>& values) {
        int id = static_cast<int>(nodes.size());
        int level = randomLevel();

        Node node;
        node.id = id;
        node.label = label;
        node.values = values;
        node.neighbors.resize(level + 1);
        nodes.push_back(node);

        if (entryPoint == -1) {
            entryPoint = id;
            maxLayer = level;
            return;
        }

        int curEntry = entryPoint;

        for (int lc = maxLayer; lc > level; lc--) {
            auto results = searchLayer(values, {curEntry}, 1, lc);
            if (!results.empty()) curEntry = results[0].second;
        }

        for (int lc = std::min(level, maxLayer); lc >= 0; lc--) {
            auto candidates = searchLayer(values, {curEntry}, efConstruction, lc);
            int maxConn = (lc == 0) ? (2 * M) : M;
            auto selected = selectNeighborsSimple(candidates, maxConn);

            nodes[id].neighbors[lc] = selected;
            for (int neighborId : selected) {
                nodes[neighborId].neighbors[lc].push_back(id);

                if ((int)nodes[neighborId].neighbors[lc].size() > maxConn) {
                    std::vector<std::pair<float,int>> nCandidates;
                    for (int nb : nodes[neighborId].neighbors[lc]) {
                        nCandidates.push_back({distance(nodes[neighborId].values, nodes[nb].values), nb});
                    }
                    nodes[neighborId].neighbors[lc] = selectNeighborsSimple(nCandidates, maxConn);
                }
            }

            if (!candidates.empty()) curEntry = candidates[0].second;
        }

        if (level > maxLayer) {
            maxLayer = level;
            entryPoint = id;
        }
    }

    // Day 6: k-NN search. ef controls recall/speed tradeoff at query time (ef >= k recommended).
    std::vector<HNSWResult> search(const std::vector<float>& query, int k, int ef) const {
        if (entryPoint == -1) return {};

        int curEntry = entryPoint;

        // Descend greedily from top layer down to layer 1 (ef=1, cheap)
        for (int lc = maxLayer; lc > 0; lc--) {
            auto results = searchLayer(query, {curEntry}, 1, lc);
            if (!results.empty()) curEntry = results[0].second;
        }

        // Final thorough search at layer 0, using the user-supplied ef
        int effectiveEf = std::max(ef, k);
        auto candidates = searchLayer(query, {curEntry}, effectiveEf, 0);

        std::vector<HNSWResult> results;
        for (int i = 0; i < std::min((int)candidates.size(), k); i++) {
            int id = candidates[i].second;
            float dist = candidates[i].first;
            results.push_back({id, nodes[id].label, 1.0f - dist});
        }
        return results;
    }

    size_t size() const { return nodes.size(); }
    int getMaxLayer() const { return maxLayer; }
    int getEntryPoint() const { return entryPoint; }

    void printGraph() const {
        for (const auto& node : nodes) {
            std::cout << "[" << node.id << "] " << node.label << " (top layer " << node.neighbors.size()-1 << ")\n";
            for (size_t l = 0; l < node.neighbors.size(); l++) {
                std::cout << "    L" << l << ": ";
                for (int nb : node.neighbors[l]) std::cout << nb << " ";
                std::cout << "\n";
            }
        }
    }
};

// ============================================================
// Day 6: Benchmark — brute-force vs KD-Tree vs HNSW, larger synthetic dataset
// ============================================================

std::vector<float> randomVector(std::mt19937& rng, int dim) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (int i = 0; i < dim; i++) v[i] = dist(rng);
    return v;
}

int main() {
    const int N = 5000;      // number of vectors
    const int DIM = 64;      // dimensions per vector
    const int K = 10;        // top-K to retrieve

    std::mt19937 rng(123);

    // Generate synthetic dataset
    std::vector<std::vector<float>> dataset;
    dataset.reserve(N);
    for (int i = 0; i < N; i++) dataset.push_back(randomVector(rng, DIM));

    // Build all three structures on the same data
    VectorStore store;
    std::vector<KDPoint> kdPoints;
    HNSWIndex hnsw(/*M=*/16, /*efConstruction=*/100, /*seed=*/42);

    for (int i = 0; i < N; i++) {
        std::string label = "vec" + std::to_string(i);
        store.insert(label, dataset[i]);
        kdPoints.push_back({i, label, dataset[i]});
        hnsw.insert(label, dataset[i]);
    }

    KDTree kdtree;
    kdtree.build(kdPoints);

    // Query vector: a fresh random vector, not one already in the dataset
    std::vector<float> query = randomVector(rng, DIM);

    // ---- Brute-force (ground truth) ----
    auto t0 = std::chrono::high_resolution_clock::now();
    auto bruteResults = store.bruteForceSearch(query, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double bruteMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- KD-Tree ----
    auto t2 = std::chrono::high_resolution_clock::now();
    auto kdResults = kdtree.kNearest(query, K);
    auto t3 = std::chrono::high_resolution_clock::now();
    double kdMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ---- HNSW ----
    auto t4 = std::chrono::high_resolution_clock::now();
    auto hnswResults = hnsw.search(query, K, /*ef=*/50);
    auto t5 = std::chrono::high_resolution_clock::now();
    double hnswMs = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // ---- Recall: what fraction of brute-force's true top-K did HNSW find? ----
    std::set<int> groundTruthIds;
    for (const auto& r : bruteResults) groundTruthIds.insert(r.id);

    int hits = 0;
    for (const auto& r : hnswResults) {
        if (groundTruthIds.count(r.id)) hits++;
    }
    double recall = (double)hits / (double)K;

    // ---- Report ----
    std::cout << "Dataset: " << N << " vectors, " << DIM << " dimensions, K=" << K << "\n\n";

    std::cout << "[Brute-Force] " << bruteMs << " ms\n";
    std::cout << "  Top 3 IDs: ";
    for (int i = 0; i < 3 && i < (int)bruteResults.size(); i++) std::cout << bruteResults[i].id << " ";
    std::cout << "\n\n";

    std::cout << "[KD-Tree]    " << kdMs << " ms\n";
    std::cout << "  Top 3 IDs: ";
    for (int i = 0; i < 3 && i < (int)kdResults.size(); i++) std::cout << kdResults[i].id << " ";
    std::cout << "\n\n";

    std::cout << "[HNSW]       " << hnswMs << " ms\n";
    std::cout << "  Top 3 IDs: ";
    for (int i = 0; i < 3 && i < (int)hnswResults.size(); i++) std::cout << hnswResults[i].id << " ";
    std::cout << "\n";
    std::cout << "  Recall@" << K << " vs brute-force ground truth: " << (recall * 100.0) << "%\n\n";

    std::cout << "Speedup (brute-force / HNSW): " << (bruteMs / hnswMs) << "x\n";

    return 0;
}