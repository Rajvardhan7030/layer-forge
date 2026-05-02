#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "gguf_parser.hpp"
#include "mmap.hpp"
#include "layers.hpp"
#include "context.hpp"
#include <functional>

namespace layer_forge {

using TokenCallback = std::function<bool(const std::string&)>;

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path);
    ~InferenceEngine() = default;

    bool init();
    void generate(const std::vector<int>& input_ids, int max_tokens, TokenCallback callback = nullptr);

    // Metadata access for /v1/models
    const std::map<std::string, GGUFValue>& get_metadata() const { return parser->get_metadata(); }

private:
    std::string model_path;
    std::unique_ptr<GGUFParser> parser;
    std::unique_ptr<MemoryMappedFile> mmap_file;
    std::unique_ptr<InferenceContext> ctx;

    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    std::vector<std::unique_ptr<TransformerBlock>> blocks;
    std::vector<std::string> vocab;

    // Model parameters
    int n_layers = 0;
    int n_embd = 0;
    int n_heads = 0;
    int n_vocab = 0;
    
    // Global Layers
    std::shared_ptr<Tensor> token_embd;
    std::shared_ptr<RMSNorm> output_norm;
    std::shared_ptr<Linear> output_weight;

    void load_global_tensors();
};

} // namespace layer_forge
