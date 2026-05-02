#include "gguf_parser.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace layer_forge {

GGUFParser::GGUFParser(const std::string& path) : file_path(path) {
    file.open(path, std::ios::binary);
}

GGUFParser::~GGUFParser() {
    if (file.is_open()) {
        file.close();
    }
}

bool GGUFParser::parse() {
    if (!file.is_open()) {
        std::cerr << "Failed to open GGUF file: " << file_path << std::endl;
        return false;
    }

    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        std::cerr << "Invalid GGUF magic" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 2 && version != 3) {
        std::cerr << "Unsupported GGUF version: " << version << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    file.read(reinterpret_cast<char*>(&metadata_count), sizeof(metadata_count));

    // Parse Metadata
    for (uint64_t i = 0; i < metadata_count; ++i) {
        std::string key = read_string();
        uint32_t type_val;
        file.read(reinterpret_cast<char*>(&type_val), sizeof(type_val));
        GGUFType type = static_cast<GGUFType>(type_val);
        metadata[key] = read_value(type);
    }

    // Parse Tensor Info
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GGUFTensorInfo info;
        info.name = read_string();
        file.read(reinterpret_cast<char*>(&info.n_dims), sizeof(info.n_dims));
        if (info.n_dims > 10) {
            throw std::runtime_error("Too many dimensions for tensor " + info.name + ": " + std::to_string(info.n_dims));
        }
        info.dims.resize(info.n_dims);
        file.read(reinterpret_cast<char*>(info.dims.data()), info.n_dims * sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&info.type), sizeof(info.type));
        file.read(reinterpret_cast<char*>(&info.offset), sizeof(info.offset));
        tensors.push_back(info);
    }

    // Find the start of the data section
    uint64_t alignment = 32; // Default GGUF alignment
    if (metadata.count("general.alignment")) {
        alignment = std::get<uint32_t>(metadata["general.alignment"].data);
    }

    uint64_t current_pos = file.tellg();
    uint64_t padded_pos = (current_pos + alignment - 1) / alignment * alignment;
    
    // Adjust offsets to be absolute file positions
    for (auto& info : tensors) {
        info.offset += padded_pos;
    }
    
    return true;
}

std::string GGUFParser::read_string() {
    uint64_t len;
    if (!file.read(reinterpret_cast<char*>(&len), sizeof(len))) {
        throw std::runtime_error("Failed to read string length");
    }
    if (len > 1024 * 1024) { // 1MB limit
        throw std::runtime_error("String too long: " + std::to_string(len));
    }
    std::string str(len, '\0');
    if (len > 0) {
        if (!file.read(&str[0], len)) {
            throw std::runtime_error("Failed to read string content");
        }
    }
    return str;
}

GGUFValue GGUFParser::read_value(GGUFType type, int depth) {
    if (depth > 16) {
        throw std::runtime_error("GGUF value recursion depth exceeded");
    }
    GGUFValue val;
    val.type = type;
    switch (type) {
        case GGUFType::UINT8: { uint8_t v; file.read(reinterpret_cast<char*>(&v), 1); val.data = v; break; }
        case GGUFType::INT8: { int8_t v; file.read(reinterpret_cast<char*>(&v), 1); val.data = v; break; }
        case GGUFType::UINT16: { uint16_t v; file.read(reinterpret_cast<char*>(&v), 2); val.data = v; break; }
        case GGUFType::INT16: { int16_t v; file.read(reinterpret_cast<char*>(&v), 2); val.data = v; break; }
        case GGUFType::UINT32: { uint32_t v; file.read(reinterpret_cast<char*>(&v), 4); val.data = v; break; }
        case GGUFType::INT32: { int32_t v; file.read(reinterpret_cast<char*>(&v), 4); val.data = v; break; }
        case GGUFType::FLOAT32: { float v; file.read(reinterpret_cast<char*>(&v), 4); val.data = v; break; }
        case GGUFType::BOOL: { bool v; file.read(reinterpret_cast<char*>(&v), 1); val.data = v; break; }
        case GGUFType::STRING: { val.data = read_string(); break; }
        case GGUFType::UINT64: { uint64_t v; file.read(reinterpret_cast<char*>(&v), 8); val.data = v; break; }
        case GGUFType::INT64: { int64_t v; file.read(reinterpret_cast<char*>(&v), 8); val.data = v; break; }
        case GGUFType::FLOAT64: { double v; file.read(reinterpret_cast<char*>(&v), 8); val.data = v; break; }
        case GGUFType::ARRAY: {
            uint32_t sub_type_val;
            if (!file.read(reinterpret_cast<char*>(&sub_type_val), sizeof(sub_type_val))) {
                throw std::runtime_error("Failed to read array sub-type");
            }
            uint64_t len;
            if (!file.read(reinterpret_cast<char*>(&len), sizeof(len))) {
                throw std::runtime_error("Failed to read array length");
            }
            if (len > 1000000) { // Limit array size to 1M elements
                 throw std::runtime_error("GGUF array too large");
            }
            std::vector<GGUFValue> vec;
            vec.reserve(len);
            for (uint64_t i = 0; i < len; ++i) {
                vec.push_back(read_value(static_cast<GGUFType>(sub_type_val), depth + 1));
            }
            val.data = vec;
            break;
        }
    }
    return val;
}

} // namespace layer_forge
