#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace layer_forge {

enum class DataType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    // Add others as needed
};

class Tensor {
public:
    Tensor(const std::vector<uint64_t>& shape, DataType type, void* data = nullptr)
        : shape(shape), type(type), data(data) {}

    const std::vector<uint64_t>& get_shape() const { return shape; }
    DataType get_type() const { return type; }
    void* get_data() const { return data; }

    size_t nelements() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    // TODO: Add methods for element access, math ops, etc.

private:
    std::vector<uint64_t> shape;
    DataType type;
    void* data; // Pointer to memory-mapped or allocated data
};

} // namespace layer_forge
