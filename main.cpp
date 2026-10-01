// main.cpp
// Day 7: Days 1-2 engine (VectorStore) wrapped in a REST API using cpp-httplib + nlohmann/json
// (KDTree/HNSWIndex from Days 3-6 intentionally left out today for clarity — see Interview Q7
//  below for exactly how you'd wire them back in as a follow-up exercise.)

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
#include <mutex>
#include "httplib.h"
#include "json.hpp"
using json = nlohmann::json;

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

float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Dimension mismatch");
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) sum += a[i] * b[i];
    return sum;
}
float magnitude(const std::vector<float>& a) { return std::sqrt(dotProduct(a, a)); }
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float magA = magnitude(a), magB = magnitude(b);
    if (magA == 0.0f || magB == 0.0f) return 0.0f;
    return dotProduct(a, b) / (magA * magB);
}

struct SearchResult { int id; std::string label; float score; };

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
    bool remove(int id) {
        auto it = std::remove_if(items.begin(), items.end(),
                                  [id](const VectorItem& v) { return v.id == id; });
        bool found = (it != items.end());
        items.erase(it, items.end());
        return found;
    }
    std::vector<SearchResult> bruteForceSearch(const std::vector<float>& query, int k) const {
        std::vector<SearchResult> results;
        results.reserve(items.size());
        for (const auto& item : items) {
            results.push_back({item.id, item.label, cosineSimilarity(query, item.values)});
        }
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
        if (results.size() > static_cast<size_t>(k)) results.resize(k);
        return results;
    }
};

// ============================================================
// REST API wiring (Day 7)
// ============================================================

int main() {
    VectorStore store;
    std::mutex storeMutex;  // protects store across concurrent HTTP requests

    httplib::Server svr;

    // Allow the frontend (Day 8) to call this from a browser
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });

    // GET /health — simple liveness check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // POST /insert  { "label": "...", "values": [0.1, 0.2, ...] }
    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string label = body.at("label").get<std::string>();
            std::vector<float> values = body.at("values").get<std::vector<float>>();

            std::lock_guard<std::mutex> lock(storeMutex);
            int id = store.insert(label, values);

            json response = {{"id", id}, {"label", label}, {"status", "inserted"}};
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });

    // POST /search  { "values": [0.1, 0.2, ...], "k": 5 }
    svr.Post("/search", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::vector<float> query = body.at("values").get<std::vector<float>>();
            int k = body.value("k", 5);

            std::lock_guard<std::mutex> lock(storeMutex);
            auto results = store.bruteForceSearch(query, k);

            json response = json::array();
            for (const auto& r : results) {
                response.push_back({{"id", r.id}, {"label", r.label}, {"score", r.score}});
            }
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });

    // DELETE /delete/:id
    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        int id = std::stoi(req.matches[1]);

        std::lock_guard<std::mutex> lock(storeMutex);
        bool found = store.remove(id);

        json response = {{"id", id}, {"deleted", found}};
        res.status = found ? 200 : 404;
        res.set_content(response.dump(), "application/json");
    });

    // GET /count — how many vectors are currently stored
    svr.Get("/count", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(storeMutex);
        json response = {{"count", store.size()}};
        res.set_content(response.dump(), "application/json");
    });

    std::cout << "Server starting on http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);

    return 0;
}