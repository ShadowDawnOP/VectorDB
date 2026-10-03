// VectorStore.h
// Day 11 refactor: the in-memory vector storage + brute-force search engine,
// split out of main.cpp into its own header.

#ifndef VECTORSTORE_H
#define VECTORSTORE_H

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>

struct VectorItem {
    int id;
    std::string label;
    std::string text;
    std::vector<float> values;
    VectorItem(int id_, std::string label_, std::string text_, std::vector<float> values_)
        : id(id_), label(std::move(label_)), text(std::move(text_)), values(std::move(values_)) {}
};

inline float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) sum += a[i] * b[i];
    return sum;
}

inline float magnitude(const std::vector<float>& a) {
    return std::sqrt(dotProduct(a, a));
}

inline float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float magA = magnitude(a), magB = magnitude(b);
    if (magA == 0.0f || magB == 0.0f) return 0.0f;
    return dotProduct(a, b) / (magA * magB);
}

struct SearchResult {
    int id;
    std::string label;
    std::string text;
    float score;
};

class VectorStore {
private:
    std::vector<VectorItem> items;
    int nextId = 0;

public:
    int insert(const std::string& label, const std::string& text, const std::vector<float>& values) {
        if (values.empty()) {
            throw std::invalid_argument("Cannot insert an empty vector");
        }
        int id = nextId++;
        items.emplace_back(id, label, text, values);
        return id;
    }

    size_t size() const { return items.size(); }

    bool remove(int id) {
        auto it = std::remove_if(items.begin(), items.end(),
                                  [id](const VectorItem& v) { return v.id == id; });
        bool found = (it != items.end());
        items.erase(it, items.end());
        return found;
    }

    std::vector<SearchResult> bruteForceSearch(const std::vector<float>& query, int k) const {
        if (k <= 0) {
            throw std::invalid_argument("k must be a positive integer");
        }
        std::vector<SearchResult> results;
        results.reserve(items.size());
        for (const auto& item : items) {
            if (item.values.size() != query.size()) {
                // Skip (don't crash) items with mismatched dimensions — defends against
                // a store that somehow ends up with mixed-dimension vectors.
                continue;
            }
            results.push_back({item.id, item.label, item.text, cosineSimilarity(query, item.values)});
        }
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
        if (results.size() > static_cast<size_t>(k)) {
            results.resize(k);
        }
        return results;
    }
};

#endif // VECTORSTORE_H