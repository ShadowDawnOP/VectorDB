// main.cpp
// Day 9: Day 7's REST API extended with real embeddings via Ollama (local, free)

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
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
// Day 9: Ollama embedding client
// ============================================================

// Calls Ollama's local /api/embed endpoint, returns the embedding vector for `text`.
// Throws std::runtime_error with a clear message if Ollama isn't running or errors out.
std::vector<float> getEmbedding(const std::string& text) {
    httplib::Client cli("http://localhost:11434");
    cli.set_connection_timeout(5);  // seconds — fail fast if Ollama isn't running
    cli.set_read_timeout(30);       // embedding can take a moment on first call (model loading)

    json requestBody = {
        {"model", "nomic-embed-text"},
        {"input", text}
    };

    auto res = cli.Post("/api/embed", requestBody.dump(), "application/json");

    if (!res) {
        throw std::runtime_error("Could not reach Ollama at localhost:11434 — is it running? Try: ollama serve");
    }
    if (res->status != 200) {
        throw std::runtime_error("Ollama returned status " + std::to_string(res->status) + ": " + res->body);
    }

    json responseBody = json::parse(res->body);
    // Ollama's /api/embed response shape: { "embeddings": [[0.1, 0.2, ...]] }
    std::vector<float> embedding = responseBody.at("embeddings")[0].get<std::vector<float>>();
    return embedding;
}

// ============================================================
// REST API wiring
// ============================================================

int main() {
    VectorStore store;
    std::mutex storeMutex;

    httplib::Server svr;

    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // Day 7: insert with pre-computed values (still works, e.g. for testing)
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

    // Day 9: NEW — insert raw text, server computes the real embedding via Ollama
    // { "label": "...", "text": "A sentence to embed" }
    svr.Post("/embed-insert", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string label = body.at("label").get<std::string>();
            std::string text = body.at("text").get<std::string>();

            std::vector<float> embedding = getEmbedding(text);  // calls Ollama

            std::lock_guard<std::mutex> lock(storeMutex);
            int id = store.insert(label, embedding);

            json response = {{"id", id}, {"label", label}, {"dimensions", embedding.size()}, {"status", "inserted"}};
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });

    // Day 9: NEW — search with raw text, server embeds the query then searches
    // { "text": "query text", "k": 5 }
    svr.Post("/embed-search", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string text = body.at("text").get<std::string>();
            int k = body.value("k", 5);

            std::vector<float> queryEmbedding = getEmbedding(text);  // calls Ollama

            std::lock_guard<std::mutex> lock(storeMutex);
            auto results = store.bruteForceSearch(queryEmbedding, k);

            json response = json::array();
            for (const auto& r : results) {
                response.push_back({{"id", r.id}, {"label", r.label}, {"score", r.score}});
            }
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        int id = std::stoi(req.matches[1]);
        std::lock_guard<std::mutex> lock(storeMutex);
        bool found = store.remove(id);
        json response = {{"id", id}, {"deleted", found}};
        res.status = found ? 200 : 404;
        res.set_content(response.dump(), "application/json");
    });

    svr.Get("/count", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(storeMutex);
        json response = {{"count", store.size()}};
        res.set_content(response.dump(), "application/json");
    });

    std::cout << "Server starting on http://localhost:8080\n";
    std::cout << "Make sure Ollama is running (ollama serve) with nomic-embed-text pulled.\n";
    svr.listen("0.0.0.0", 8080);

    return 0;
}