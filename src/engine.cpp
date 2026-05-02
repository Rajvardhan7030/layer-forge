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
        const auto& tokens_val = metadata.at("tokenizer.ggml.tokens");
        const auto& tokens_vec = std::get<std::vector<GGUFValue>>(tokens_val.data);
        n_vocab = tokens_vec.size();
        vocab.reserve(n_vocab);
        for (const auto& v : tokens_vec) {
            vocab.push_back(std::get<std::string>(v.data));
        }
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

void InferenceEngine::generate(const std::vector<int>& input_ids, int max_tokens, TokenCallback callback) {
    std::cout << "Starting generation for " << input_ids.size() << " prompt tokens..." << std::endl;
    
    std::vector<int> tokens = input_ids;
    
    // Process prompt (simplified for now, assuming prompt is small and fits in context)
    for (size_t i = 0; i < tokens.size(); ++i) {
        int token_id = tokens[i];
        
        // 1. Embedding lookup
        if (token_embd->get_type() == DataType::Q4_0) {
            const block_q4_0* embed_data = static_cast<const block_q4_0*>(token_embd->get_data());
            int blocks_per_row = n_embd / QK4_0;
            dequantize_q4_0(embed_data + token_id * blocks_per_row, ctx->workspace1.data(), n_embd);
        } else if (token_embd->get_type() == DataType::F32) {
            float* embed_data = static_cast<float*>(token_embd->get_data());
            std::copy_n(embed_data + token_id * n_embd, n_embd, ctx->workspace1.begin());
        }

        // 2. Transformer layers
        for (auto& block : blocks) {
            block->forward(*ctx);
        }

        // Only generate new tokens after the prompt
        if (i < tokens.size() - 1) continue;

        // 3. Final norm and logits
        Tensor h({(uint64_t)n_embd}, DataType::F32, ctx->workspace1.data());
        Tensor logits_tensor({(uint64_t)n_vocab}, DataType::F32, ctx->logits.data());
        output_norm->forward(h, h);
        output_weight->forward(h, logits_tensor);

        // 4. Greedy sampling
        int next_token = 0;
        float max_logit = -1e20f;
        for (int v = 0; v < n_vocab; ++v) {
            if (ctx->logits[v] > max_logit) {
                max_logit = ctx->logits[v];
                next_token = v;
            }
        }

        // Detokenize and callback
        if (callback) {
            std::string token_str = " ";
            if (next_token < (int)vocab.size()) {
                token_str = vocab[next_token];
                // Clean up GGUF token prefix (e.g.,   for Llama)
                if (token_str.find("\u2581") == 0) {
                    token_str = " " + token_str.substr(3);
                }
            }
            if (!callback(token_str)) break;
        }

        tokens.push_back(next_token);
        
        // EOS check
        if (next_token == 2) break; // Common EOS
        
        if (tokens.size() - input_ids.size() >= (size_t)max_tokens) break;
        
        // Decrement loop counter to keep generating until max_tokens or EOS
        i = tokens.size() - 2; 
    }
}

} // namespace layer_forge
