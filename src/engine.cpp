#include "engine.hpp"
#include "compute.hpp"
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
    n_layers = as_uint64(metadata.at("llama.block_count"));
    n_embd = as_uint64(metadata.at("llama.embedding_length"));
    n_heads = as_uint64(metadata.at("llama.attention.head_count"));
    
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

        void* ptr = mmap_file->get_ptr(info.offset);
        if (!ptr && info.offset > 0) { // offset 0 might be valid for empty file but here we expect data
            throw std::runtime_error("Invalid offset for tensor: " + info.name);
        }

        tensors[info.name] = std::make_shared<Tensor>(
            info.dims, 
            type, 
            ptr
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

    static auto get_tensor = [](const std::map<std::string, std::shared_ptr<Tensor>>& tensors, const std::string& name) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("Missing tensor: " + name);
        }
        return it->second;
    };

    token_embd = get_tensor(tensors, "token_embd.weight");
    output_norm = std::make_shared<RMSNorm>(get_tensor(tensors, "output_norm.weight"));
    output_weight = std::make_shared<Linear>(get_tensor(tensors, "output.weight"));

    // Initialize Inference Context
    ctx = std::make_unique<InferenceContext>(n_embd, n_layers, n_vocab);

    return true;
}

void InferenceEngine::generate(const std::vector<int>& input_ids, int max_tokens) {
    std::cout << "Starting generation for " << input_ids.size() << " tokens..." << std::endl;
    
    for (int token_id : input_ids) {
        if (token_id < 0 || token_id >= n_vocab) {
            throw std::out_of_range("Invalid token ID: " + std::to_string(token_id));
        }

        // 1. Embedding lookup (zero-copy into workspace1)
        if (token_embd->get_type() == DataType::Q4_0) {
            const block_q4_0* embed_data = static_cast<const block_q4_0*>(token_embd->get_data());
            // n_embd is the number of elements per row, but blocks are size QK4_0
            if (n_embd % QK4_0 != 0) {
                throw std::runtime_error("Embedding dimension must be a multiple of " + std::to_string(QK4_0));
            }
            int blocks_per_row = n_embd / QK4_0;
            dequantize_q4_0(embed_data + token_id * blocks_per_row, ctx->workspace1.data(), n_embd);
        } else if (token_embd->get_type() == DataType::F32) {
            float* embed_data = static_cast<float*>(token_embd->get_data());
            std::copy_n(embed_data + token_id * n_embd, n_embd, ctx->workspace1.begin());
        } else {
            throw std::runtime_error("Unsupported embedding tensor type");
        }
        
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
