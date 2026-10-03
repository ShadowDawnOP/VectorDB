// OllamaClient.h
// Day 11 refactor: everything that talks to Ollama (embeddings + generation),
// split out of main.cpp into its own header.

#ifndef OLLAMACLIENT_H
#define OLLAMACLIENT_H

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

inline std::vector<float> getEmbedding(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("Cannot embed empty text");
    }

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
    if (!responseBody.contains("embeddings") || responseBody["embeddings"].empty()) {
        throw std::runtime_error("Ollama response missing expected 'embeddings' field — is nomic-embed-text pulled?");
    }
    return responseBody.at("embeddings")[0].get<std::vector<float>>();
}

inline std::string generateAnswer(const std::string& prompt) {
    httplib::Client cli("http://localhost:11434");
    cli.set_connection_timeout(5);
    cli.set_read_timeout(120);

    json requestBody = {
        {"model", "llama3.2"},
        {"prompt", prompt},
        {"stream", false}
    };

    auto res = cli.Post("/api/generate", requestBody.dump(), "application/json");

    if (!res) {
        throw std::runtime_error("Could not reach Ollama at localhost:11434 — is it running? Try: ollama serve");
    }
    if (res->status != 200) {
        throw std::runtime_error("Ollama returned status " + std::to_string(res->status) + ": " + res->body);
    }

    json responseBody = json::parse(res->body);
    if (!responseBody.contains("response")) {
        throw std::runtime_error("Ollama response missing expected 'response' field — is llama3.2 pulled?");
    }
    return responseBody.at("response").get<std::string>();
}

#endif // OLLAMACLIENT_H