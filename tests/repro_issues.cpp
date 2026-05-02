#include "gguf_parser.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cassert>

using namespace layer_forge;

void create_fake_gguf_huge_string(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f.write("GGUF", 4);
    uint32_t version = 3;
    f.write(reinterpret_cast<char*>(&version), 4);
    uint64_t tensor_count = 0;
    f.write(reinterpret_cast<char*>(&tensor_count), 8);
    uint64_t metadata_count = 1;
    f.write(reinterpret_cast<char*>(&metadata_count), 8);

    // Metadata key
    uint64_t key_len = 4;
    f.write(reinterpret_cast<char*>(&key_len), 8);
    f.write("test", 4);

    // Metadata value type (String = 8)
    uint32_t type = 8;
    f.write(reinterpret_cast<char*>(&type), 4);

    // Maliciously large string length (e.g., 2GB)
    uint64_t huge_len = 2ULL * 1024 * 1024 * 1024;
    f.write(reinterpret_cast<char*>(&huge_len), 8);
    f.close();
}

void create_fake_gguf_nested_array(const std::string& path, int depth) {
    std::ofstream f(path, std::ios::binary);
    f.write("GGUF", 4);
    uint32_t version = 3;
    f.write(reinterpret_cast<char*>(&version), 4);
    uint64_t tensor_count = 0;
    f.write(reinterpret_cast<char*>(&tensor_count), 8);
    uint64_t metadata_count = 1;
    f.write(reinterpret_cast<char*>(&metadata_count), 8);

    // Metadata key
    uint64_t key_len = 4;
    f.write(reinterpret_cast<char*>(&key_len), 8);
    f.write("test", 4);

    // Metadata value type (Array = 9)
    uint32_t type = 9;
    f.write(reinterpret_cast<char*>(&type), 4);

    // Nested arrays
    for (int i = 0; i < depth; ++i) {
        uint32_t sub_type = 9;
        f.write(reinterpret_cast<char*>(&sub_type), 4);
        uint64_t len = 1;
        f.write(reinterpret_cast<char*>(&len), 8);
    }
    // End with a simple type
    uint32_t final_type = 0; // UINT8
    f.write(reinterpret_cast<char*>(&final_type), 4);
    uint64_t final_len = 1;
    f.write(reinterpret_cast<char*>(&final_len), 8);
    uint8_t val = 42;
    f.write(reinterpret_cast<char*>(&val), 1);
    f.close();
}

void create_fake_gguf_many_dims(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f.write("GGUF", 4);
    uint32_t version = 3;
    f.write(reinterpret_cast<char*>(&version), 4);
    uint64_t tensor_count = 1;
    f.write(reinterpret_cast<char*>(&tensor_count), 8);
    uint64_t metadata_count = 0;
    f.write(reinterpret_cast<char*>(&metadata_count), 8);

    // Tensor info
    uint64_t name_len = 4;
    f.write(reinterpret_cast<char*>(&name_len), 8);
    f.write("test", 4);
    
    uint32_t n_dims = 100; // Maliciously many dims
    f.write(reinterpret_cast<char*>(&n_dims), 4);
    f.close();
}

int main() {
    std::cout << "Testing Issue 1: Huge String..." << std::endl;
    create_fake_gguf_huge_string("huge_string.gguf");
    try {
        GGUFParser parser("huge_string.gguf");
        parser.parse();
        std::cerr << "FAIL: Issue 1 not caught!" << std::endl;
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "SUCCESS: Caught expected error: " << e.what() << std::endl;
    }

    std::cout << "\nTesting Issue 2: Nested Array..." << std::endl;
    create_fake_gguf_nested_array("nested_array.gguf", 20); // Limit is 16
    try {
        GGUFParser parser("nested_array.gguf");
        parser.parse();
        std::cerr << "FAIL: Issue 2 not caught!" << std::endl;
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "SUCCESS: Caught expected error: " << e.what() << std::endl;
    }

    std::cout << "\nTesting Issue 8: Many Dimensions..." << std::endl;
    create_fake_gguf_many_dims("many_dims.gguf");
    try {
        GGUFParser parser("many_dims.gguf");
        parser.parse();
        std::cerr << "FAIL: Issue 8 not caught!" << std::endl;
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "SUCCESS: Caught expected error: " << e.what() << std::endl;
    }

    std::cout << "\nTesting as_uint64 helper..." << std::endl;
    GGUFValue v32; v32.type = GGUFType::UINT32; v32.data = (uint32_t)42;
    assert(as_uint64(v32) == 42);
    GGUFValue v64; v64.type = GGUFType::UINT64; v64.data = (uint64_t)123456789012345ULL;
    assert(as_uint64(v64) == 123456789012345ULL);
    GGUFValue vi8; vi8.type = GGUFType::INT8; vi8.data = (int8_t)-1;
    assert(as_uint64(vi8) == (uint64_t)-1);
    std::cout << "SUCCESS: as_uint64 helper passed!" << std::endl;

    std::cout << "\nAll GGUF parser security and helper tests passed!" << std::endl;
    return 0;
}
