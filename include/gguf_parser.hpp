#pragma once

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <cstdint>
#include <fstream>

namespace layer_forge {

enum class GGUFType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12
};

struct GGUFValue {
    GGUFType type;
    std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, 
                 float, bool, std::string, uint64_t, int64_t, double, 
                 std::vector<GGUFValue>> data;
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dims;
    uint32_t type; // GGML type
    uint64_t offset;
    size_t size;
};

class GGUFParser {
public:
    GGUFParser(const std::string& path);
    ~GGUFParser();

    bool parse();
    
    // Metadata accessors
    const std::map<std::string, GGUFValue>& get_metadata() const { return metadata; }
    const std::vector<GGUFTensorInfo>& get_tensors() const { return tensors; }

private:
    std::string file_path;
    std::ifstream file;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    
    std::map<std::string, GGUFValue> metadata;
    std::vector<GGUFTensorInfo> tensors;

    // Helper functions for reading GGUF types
    std::string read_string();
    GGUFValue read_value(GGUFType type);
};

} // namespace layer_forge
