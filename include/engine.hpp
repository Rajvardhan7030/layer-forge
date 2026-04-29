#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "gguf_parser.hpp"
#include "mmap.hpp"
#include "layers.hpp"
#include "context.hpp"

namespace layer_forge {

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path);
    ~InferenceEngine() = default;

    bool init();
    void generate(const std::vector<int>& input_ids, int max_tokens);

private:
    std::string model_path;
    std::unique_ptr<GGUFParser> parser;
    std::unique_ptr<MemoryMappedFile> mmap_file;
    std::unique_ptr<InferenceContext> ctx;
    
    std::map<std::string, std::shared_ptr<Tensor>> tensors;
    std::vector<std::unique_ptr<TransformerBlock>> blocks;
    
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
