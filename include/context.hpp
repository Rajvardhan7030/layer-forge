#pragma once

#include <vector>
#include <memory>
#include "tensor.hpp"

namespace layer_forge {

/**
 * @brief InferenceContext holds pre-allocated buffers used across all layers.
 * This prevents per-layer memory allocations and reduces fragmentation.
 */
struct InferenceContext {
    int n_embd;
    int n_layers;
    
    // Shared buffers for hidden states and intermediate calculations
    std::vector<float> workspace1; // primary hidden states
    std::vector<float> workspace2; // residual/intermediate buffer
    std::vector<float> logits;     // final output logits

    InferenceContext(int embd, int layers, int vocab) 
        : n_embd(embd), n_layers(layers) {
        workspace1.resize(embd);
        workspace2.resize(embd);
        logits.resize(vocab);
    }
};

} // namespace layer_forge
