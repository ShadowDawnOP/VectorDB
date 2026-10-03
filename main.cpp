// main.cpp
// Day 11: refactored into VectorStore.h + OllamaClient.h + this file (REST API wiring only).
// Behavior is identical to Day 10 — this is a structure/organization pass, not a new feature.

#include <iostream>
#include <string>
#include <mutex>
#include <sstream>

#include "httplib.h"
#include "json.hpp"
#include "VectorStore.h"
#include "OllamaClient.h"

using json = nlohmann::json;

// Builds the prompt sent to the LLM: retrieved context chunks + instructions + the question.
// Kept in main.cpp (not OllamaClient.h) since it's application-specific logic, not a generic
// "talk to Ollama" concern — OllamaClient.h stays reusable for any project, this prompt format is ours.
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
        } catch (const std::runtime_error& e) {
            // Ollama unreachable / bad Ollama response — not the client's fault
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        } catch (const std::exception& e) {
            // Malformed JSON, missing fields, invalid values — the client's fault
            res.status = 400;
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
        } catch (const std::runtime_error& e) {
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string question = body.at("question").get<std::string>();
            int k = body.value("k", 3);

            {
                std::lock_guard<std::mutex> lock(storeMutex);
                if (store.size() == 0) {
                    json response = {{"answer", "No documents have been inserted yet — nothing to search."}, {"sources", json::array()}};
                    res.set_content(response.dump(), "application/json");
                    return;
                }
            }

            std::vector<float> questionEmbedding = getEmbedding(question);

            std::vector<SearchResult> retrieved;
            {
                std::lock_guard<std::mutex> lock(storeMutex);
                retrieved = store.bruteForceSearch(questionEmbedding, k);
            }

            std::string prompt = buildRagPrompt(retrieved, question);
            std::string answer = generateAnswer(prompt);

            json sources = json::array();
            for (const auto& r : retrieved) {
                sources.push_back({{"id", r.id}, {"label", r.label}, {"score", r.score}});
            }

            json response = {{"answer", answer}, {"sources", sources}};
            res.set_content(response.dump(), "application/json");
        } catch (const std::runtime_error& e) {
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
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
    svr.listen("0.0.0.0", 8080);

    return 0;
}