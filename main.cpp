// main.cpp
// Day 10: Day 9's embedding engine extended into a full RAG pipeline
// (retrieval via embeddings + generation via a local Ollama chat model)

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <mutex>
#include <sstream>

#include "httplib.h"
#include "json.hpp"
using json = nlohmann::json;

// ============================================================
// Day 1-2 + Day 9: VectorItem now also stores the original text
// (needed for RAG — search gives us WHICH chunk matched, but we need
//  the chunk's actual content to hand to the LLM as context)
// ============================================================

struct VectorItem {
    int id;
    std::string label;
    std::string text;        // NEW (Day 10): the original source text this vector represents
    std::vector<float> values;
    VectorItem(int id_, std::string label_, std::string text_, std::vector<float> values_)
        : id(id_), label(std::move(label_)), text(std::move(text_)), values(std::move(values_)) {}
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

struct SearchResult { int id; std::string label; std::string text; float score; };

class VectorStore {
private:
    std::vector<VectorItem> items;
    int nextId = 0;
public:
    int insert(const std::string& label, const std::string& text, const std::vector<float>& values) {
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
        std::vector<SearchResult> results;
        results.reserve(items.size());
        for (const auto& item : items) {
            results.push_back({item.id, item.label, item.text, cosineSimilarity(query, item.values)});
        }
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
        if (results.size() > static_cast<size_t>(k)) results.resize(k);
        return results;
    }
};

// ============================================================
// Day 9: Ollama embedding client (unchanged)
// ============================================================

std::vector<float> getEmbedding(const std::string& text) {
    httplib::Client cli("http://localhost:11434");
    cli.set_connection_timeout(5);
    cli.set_read_timeout(30);

    json requestBody = {{"model", "nomic-embed-text"}, {"input", text}};
    auto res = cli.Post("/api/embed", requestBody.dump(), "application/json");

    if (!res) {
        throw std::runtime_error("Could not reach Ollama at localhost:11434 — is it running? Try: ollama serve");
    }
    if (res->status != 200) {
        throw std::runtime_error("Ollama returned status " + std::to_string(res->status) + ": " + res->body);
    }

    json responseBody = json::parse(res->body);
    return responseBody.at("embeddings")[0].get<std::vector<float>>();
}

// ============================================================
// Day 10: Ollama generation client — the "G" in RAG
// ============================================================

// Calls Ollama's /api/generate endpoint with a chat model, returns the generated answer text.
// `prompt` should already contain the retrieved context + the user's question, assembled by the caller.
std::string generateAnswer(const std::string& prompt) {
    httplib::Client cli("http://localhost:11434");
    cli.set_connection_timeout(5);
    cli.set_read_timeout(120);  // generation is slower than embedding — give it real time, especially on CPU

    json requestBody = {
        {"model", "llama3.2"},   // small, free, runs on CPU — swap for any chat model you've pulled
        {"prompt", prompt},
        {"stream", false}        // simpler to handle one full response than a streamed one, for now
    };

    auto res = cli.Post("/api/generate", requestBody.dump(), "application/json");

    if (!res) {
        throw std::runtime_error("Could not reach Ollama at localhost:11434 — is it running? Try: ollama serve");
    }
    if (res->status != 200) {
        throw std::runtime_error("Ollama returned status " + std::to_string(res->status) + ": " + res->body);
    }

    json responseBody = json::parse(res->body);
    return responseBody.at("response").get<std::string>();
}

// Builds the actual prompt sent to the LLM: retrieved context chunks + instructions + the question.
// This is the "prompt engineering" piece of RAG — how you frame this matters a lot for answer quality.
std::string buildRagPrompt(const std::vector<SearchResult>& retrievedChunks, const std::string& question) {
    std::ostringstream prompt;
    prompt << "You are a helpful assistant. Answer the question using ONLY the context below. "
           << "If the context doesn't contain the answer, say you don't know — do not make something up.\n\n";

    prompt << "Context:\n";
    for (const auto& chunk : retrievedChunks) {
        prompt << "- (" << chunk.label << ") " << chunk.text << "\n";
    }

    prompt << "\nQuestion: " << question << "\nAnswer:";
    return prompt.str();
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

    // Insert raw text; server embeds it via Ollama and stores both the vector AND the original text
    svr.Post("/embed-insert", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string label = body.at("label").get<std::string>();
            std::string text = body.at("text").get<std::string>();

            std::vector<float> embedding = getEmbedding(text);

            std::lock_guard<std::mutex> lock(storeMutex);
            int id = store.insert(label, text, embedding);

            json response = {{"id", id}, {"label", label}, {"dimensions", embedding.size()}, {"status", "inserted"}};
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/embed-search", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string text = body.at("text").get<std::string>();
            int k = body.value("k", 5);

            std::vector<float> queryEmbedding = getEmbedding(text);

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

    // Day 10: NEW — the full RAG endpoint. { "question": "...", "k": 3 }
    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string question = body.at("question").get<std::string>();
            int k = body.value("k", 3);

            // Cheap check FIRST: if nothing's stored, don't bother calling Ollama at all
            {
                std::lock_guard<std::mutex> lock(storeMutex);
                if (store.size() == 0) {
                    json response = {{"answer", "No documents have been inserted yet — nothing to search."}, {"sources", json::array()}};
                    res.set_content(response.dump(), "application/json");
                    return;
                }
            }

            // 1. RETRIEVE: embed the question, search for the most relevant stored chunks
            std::vector<float> questionEmbedding = getEmbedding(question);

            std::vector<SearchResult> retrieved;
            {
                std::lock_guard<std::mutex> lock(storeMutex);
                retrieved = store.bruteForceSearch(questionEmbedding, k);
            }

            // 2. AUGMENT: build a prompt combining the retrieved context with the question
            std::string prompt = buildRagPrompt(retrieved, question);

            // 3. GENERATE: send the augmented prompt to the LLM
            std::string answer = generateAnswer(prompt);

            // Return both the answer AND which sources were used — critical for trust/verification
            json sources = json::array();
            for (const auto& r : retrieved) {
                sources.push_back({{"id", r.id}, {"label", r.label}, {"score", r.score}});
            }

            json response = {{"answer", answer}, {"sources", sources}};
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
    std::cout << "Requires Ollama running with nomic-embed-text (embeddings) and llama3.2 (generation).\n";
    svr.listen("0.0.0.0", 8080);

    return 0;
}