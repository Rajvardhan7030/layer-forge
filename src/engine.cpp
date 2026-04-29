#include "engine.hpp"
#include <iostream>
#include <algorithm>

namespace layer_forge {

InferenceEngine::InferenceEngine(const std::string& path) : model_path(path) {}

bool InferenceEngine::init() {
    parser = std::make_unique<GGUFParser>(model_path);
    if (!parser->parse()) return false;

    mmap_file = std::make_unique<MemoryMappedFile>(model_path);
    
    // Extract hyperparams
    const auto& metadata = parser->get_metadata();
    n_layers = std::get<uint32_t>(metadata.at("llama.block_count").data);
    n_embd = std::get<uint32_t>(metadata.at("llama.embedding_length").data);
    n_heads = std::get<uint32_t>(metadata.at("llama.attention.head_count").data);
    
    if (metadata.count("tokenizer.ggml.tokens")) {
        n_vocab = std::get<std::vector<GGUFValue>>(metadata.at("tokenizer.ggml.tokens").data).size();
    } else {
        n_vocab = 32000; // Fallback
    }

    // Map all tensors
    for (const auto& info : parser->get_tensors()) {
        DataType type = DataType::F32; // Default
        if (info.type == 2) type = DataType::Q4_0; // GGML_TYPE_Q4_0
        // ... map other GGML types to our DataType ...

        tensors[info.name] = std::make_shared<Tensor>(
            info.dims, 
            type, 
            mmap_file->get_ptr(info.offset)
        );
    }

    // Initialize blocks
    for (int i = 0; i < n_layers; ++i) {
        auto block = std::make_unique<TransformerBlock>(i);
        // In a true streaming mode, we wouldn't load all weights here.
        // But for now, let's just setup the structure.
        block->load_weights(tensors);
        blocks.push_back(std::move(block));
    }

    token_embd = tensors.at("token_embd.weight");
    output_norm = std::make_shared<RMSNorm>(tensors.at("output_norm.weight"));
    output_weight = std::make_shared<Linear>(tensors.at("output.weight"));

    // Initialize Inference Context
    ctx = std::make_unique<InferenceContext>(n_embd, n_layers, n_vocab);

    return true;
}

void InferenceEngine::generate(const std::vector<int>& input_ids, int max_tokens) {
    std::cout << "Starting generation for " << input_ids.size() << " tokens..." << std::endl;
    
    for (int token_id : input_ids) {
        // 1. Embedding lookup (zero-copy into workspace1)
        float* embed_data = static_cast<float*>(token_embd->get_data());
        std::copy_n(embed_data + token_id * n_embd, n_embd, ctx->workspace1.begin());
        
        // 2. Transformer layers (streaming mode)
        for (auto& block : blocks) {
            // block->load_weights(tensors); // In full streaming, load here
            block->forward(*ctx);
            // block->unload_weights();     // and unload here
        }
        
        // 3. Final norm and logits
        Tensor h({(uint64_t)n_embd}, DataType::F32, ctx->workspace1.data());
        Tensor logits({(uint64_t)n_vocab}, DataType::F32, ctx->logits.data());
        
        output_norm->forward(h, h);
        output_weight->forward(h, logits);
        
        // ... greedy sampling ...
        std::cout << "Computed logits for token " << token_id << std::endl;
    }
}

} // namespace layer_forge
