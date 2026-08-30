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
#include <random>
#include <queue>
#include <unordered_set>

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

// ============================================================
// Day 5: HNSWIndex — insertion / multilayer graph construction
// ============================================================

class HNSWIndex {
public:
    struct Node {
        int id;
        std::string label;
        std::vector<float> values;
        std::vector<std::vector<int>> neighbors; // neighbors[layer] = list of node ids
    };

private:
    std::vector<Node> nodes;
    int entryPoint = -1;
    int maxLayer = -1;
    int M;                 // max connections per node per layer (2*M at layer 0)
    int efConstruction;    // candidate list size while building
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniform01;

    // Distance = 1 - cosine similarity, so smaller = more similar (standard "distance" convention)
    float distance(const std::vector<float>& a, const std::vector<float>& b) const {
        return 1.0f - cosineSimilarity(a, b);
    }

    // Random layer assignment, exponentially decaying (mirrors skip-list promotion)
    int randomLevel() {
        double r = uniform01(rng);
        if (r <= 0.0) r = 1e-9;  // guard against log(0)
        double levelLambda = 1.0 / std::log(1.0 * M);
        return static_cast<int>(-std::log(r) * levelLambda);
    }

    // Greedy search on ONE layer. Returns up to `ef` closest nodes found, ascending distance.
    // Reused by both insert() (ef=efConstruction) and, on Day 6, search() (ef=user-supplied).
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

            // Stop condition: current candidate is worse than our worst "best" AND we have enough results
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
                    if (best.size() > static_cast<size_t>(ef)) best.pop();  // evict worst
                }
            }
        }

        std::vector<std::pair<float,int>> result;
        while (!best.empty()) { result.push_back(best.top()); best.pop(); }
        std::reverse(result.begin(), result.end());  // was max-heap order, flip to ascending
        return result;
    }

    // Simple neighbor selection heuristic: just keep the closest `maxM` candidates.
    // (There's a smarter "heuristic" version in the original HNSW paper that favors diverse
    // directions, not just closest — worth mentioning in interviews, see Q7 below.)
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

        // First node ever inserted: just becomes the entry point, nothing to connect to.
        if (entryPoint == -1) {
            entryPoint = id;
            maxLayer = level;
            return;
        }

        int curEntry = entryPoint;

        // Phase 1: descend from top layer to (level+1), cheap greedy 1-NN search each time,
        // just to find a good starting point for the real work below.
        for (int lc = maxLayer; lc > level; lc--) {
            auto results = searchLayer(values, {curEntry}, 1, lc);
            if (!results.empty()) curEntry = results[0].second;
        }

        // Phase 2: from min(level, maxLayer) down to 0, find real candidates and wire up edges.
        for (int lc = std::min(level, maxLayer); lc >= 0; lc--) {
            auto candidates = searchLayer(values, {curEntry}, efConstruction, lc);
            int maxConn = (lc == 0) ? (2 * M) : M;
            auto selected = selectNeighborsSimple(candidates, maxConn);

            // Bidirectional connections: new node <-> each selected neighbor
            nodes[id].neighbors[lc] = selected;
            for (int neighborId : selected) {
                nodes[neighborId].neighbors[lc].push_back(id);

                // If that neighbor is now over capacity, prune it back down to its closest maxConn
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

        // New node reaches higher than anything seen before -> it's the new entry point.
        if (level > maxLayer) {
            maxLayer = level;
            entryPoint = id;
        }
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

        // ---- HNSW ----
    HNSWIndex index(/*M=*/4, /*efConstruction=*/50, /*seed=*/42);

    index.insert("CS",      {0.9f, 0.8f, 0.1f, 0.1f});
    index.insert("Math",    {0.85f, 0.75f, 0.15f, 0.05f});
    index.insert("Food",    {0.1f, 0.1f, 0.9f, 0.8f});
    index.insert("Sports",  {0.2f, 0.1f, 0.1f, 0.9f});
    index.insert("Art",     {0.3f, 0.6f, 0.4f, 0.2f});
    index.insert("Physics", {0.8f, 0.85f, 0.05f, 0.1f});

    std::cout << "\n[HNSW] Inserted " << index.size() << " nodes.\n";
    std::cout << "Entry point: " << index.getEntryPoint() << ", max layer: " << index.getMaxLayer() << "\n\n";
    index.printGraph();

    return 0;
}