// main.cpp
// Day 2: Distance metrics + Brute-Force search

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>

struct VectorItem {
    int id;
    std::string label;
    std::vector<float> values;

    VectorItem(int id_, std::string label_, std::vector<float> values_)
        : id(id_), label(std::move(label_)), values(std::move(values_)) {}
};

// ---------- Distance Metrics ----------

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

// Returns a value in [-1, 1]. 1 = identical direction, 0 = orthogonal, -1 = opposite.
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float magA = magnitude(a);
    float magB = magnitude(b);
    if (magA == 0.0f || magB == 0.0f) return 0.0f;  // avoid divide-by-zero
    return dotProduct(a, b) / (magA * magB);
}

// ---------- Search result struct ----------

struct SearchResult {
    int id;
    std::string label;
    float score;  // similarity (higher = better) for cosine
};

// ---------- VectorStore with Brute-Force search ----------

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

    // Brute-force top-K search using cosine similarity.
    // O(N * d) — checks every stored vector.
    std::vector<SearchResult> bruteForceSearch(const std::vector<float>& query, int k) const {
        std::vector<SearchResult> results;
        results.reserve(items.size());

        for (const auto& item : items) {
            float score = cosineSimilarity(query, item.values);
            results.push_back({item.id, item.label, score});
        }

        // Sort descending by score (higher cosine similarity = more similar)
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

int main() {
    VectorStore store;

    store.insert("CS",     {0.9f, 0.8f, 0.1f, 0.1f});
    store.insert("Math",   {0.85f, 0.75f, 0.15f, 0.05f});
    store.insert("Food",   {0.1f, 0.1f, 0.9f, 0.8f});
    store.insert("Sports", {0.2f, 0.1f, 0.1f, 0.9f});

    std::vector<float> query = {0.88f, 0.79f, 0.12f, 0.08f};  // should match "CS"/"Math"

    auto results = store.bruteForceSearch(query, 2);

    std::cout << "Top " << results.size() << " matches:\n";
    for (const auto& r : results) {
        std::cout << "  [" << r.id << "] " << r.label
                  << "  (cosine similarity: " << r.score << ")\n";
    }

    return 0;
}