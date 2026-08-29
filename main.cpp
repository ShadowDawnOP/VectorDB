#include <iostream>
#include <vector>
#include <string>
#include <cmath>

struct VectorItem {
    int id;
    std::string label;      
    std::vector<float> values;

    // Constructor
    VectorItem(int id_, std::string label_, std::vector<float> values_)
        : id(id_), label(std::move(label_)), values(std::move(values_)) {}
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

    // Get total count
    size_t size() const {
        return items.size();
    }

    // Print all stored items (debug helper)
    void printAll() const {
        for (const auto& item : items) {
            std::cout << "[" << item.id << "] " << item.label << " -> (";
            for (size_t i = 0; i < item.values.size(); i++) {
                std::cout << item.values[i];
                if (i != item.values.size() - 1) std::cout << ", ";
            }
            std::cout << ")\n";
        }
    }
};

int main() {
    VectorStore store;

    store.insert("CS", {0.9f, 0.8f, 0.1f, 0.1f});
    store.insert("Food", {0.1f, 0.1f, 0.9f, 0.8f});
    store.insert("Sports", {0.2f, 0.1f, 0.1f, 0.9f});

    std::cout << "Stored " << store.size() << " vectors:\n";
    store.printAll();

    return 0;
}