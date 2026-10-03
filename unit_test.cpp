#include <iostream>
#include "VectorStore.h"

int testsRun = 0, testsPassed = 0;

void check(bool condition, const std::string& description) {
    testsRun++;
    if (condition) {
        testsPassed++;
        std::cout << "  PASS: " << description << "\n";
    } else {
        std::cout << "  FAIL: " << description << "\n";
    }
}

int main() {
    std::cout << "VectorStore unit tests:\n";

    {
        VectorStore store;
        store.insert("a", "text a", {1.0f, 0.0f});
        store.insert("b", "text b", {0.0f, 1.0f});
        auto results = store.bruteForceSearch({1.0f, 0.0f}, 1);
        check(results.size() == 1 && results[0].label == "a", "basic search finds the correct closest match");
    }

    {
        VectorStore store;
        bool threw = false;
        try { store.insert("bad", "text", {}); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "inserting an empty vector throws invalid_argument");
    }

    {
        VectorStore store;
        store.insert("a", "text", {1.0f, 0.0f});
        bool threw = false;
        try { store.bruteForceSearch({1.0f, 0.0f}, 0); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "search with k=0 throws invalid_argument");
    }

    {
        VectorStore store;
        store.insert("a", "text", {1.0f, 0.0f});
        bool threw = false;
        try { store.bruteForceSearch({1.0f, 0.0f}, -5); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "search with negative k throws invalid_argument");
    }

    {
        VectorStore store;
        store.insert("a", "text", {1.0f, 0.0f});
        store.insert("b", "text", {0.0f, 1.0f});
        auto results = store.bruteForceSearch({1.0f, 0.0f}, 1000);
        check(results.size() == 2, "k larger than store size returns all available items, doesn't crash");
    }

    {
        VectorStore store;
        store.insert("2d", "text", {1.0f, 0.0f});
        store.insert("3d", "text", {1.0f, 0.0f, 0.0f});  // different dimension, inserted anyway
        bool threw = false;
        std::vector<SearchResult> results;
        try { results = store.bruteForceSearch({1.0f, 0.0f}, 5); }
        catch (...) { threw = true; }
        check(!threw, "search doesn't crash when store has mixed-dimension vectors");
        check(results.size() == 1 && results[0].label == "2d", "mismatched-dimension item is correctly skipped, not included");
    }

    {
        VectorStore store;
        auto results = store.bruteForceSearch({1.0f, 0.0f}, 5);
        check(results.empty(), "searching an empty store returns empty results without crashing");
    }

    {
        VectorStore store;
        store.insert("a", "text", {1.0f});
        bool found = store.remove(999);
        check(!found, "deleting a nonexistent id returns false");
        check(store.size() == 1, "store size unchanged after failed delete");
    }

    {
        float sim = cosineSimilarity({0.0f, 0.0f}, {1.0f, 0.0f});
        check(!std::isnan(sim) && sim == 0.0f, "cosine similarity with a zero vector returns 0, not NaN");
    }

    std::cout << "\n" << testsPassed << "/" << testsRun << " tests passed\n";
    return (testsPassed == testsRun) ? 0 : 1;
}