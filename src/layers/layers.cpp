#include "layers.hpp"
#include "compute.hpp"
#include <cmath>
#include <algorithm>

namespace layer_forge {

void RMSNorm::forward(const Tensor& input, Tensor& output) {
    const float* x = static_cast<const float*>(input.get_data());
    float* y = static_cast<float*>(output.get_data());
    const float* w = static_cast<const float*>(weight->get_data());
    size_t n = input.nelements();

    float ss = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        ss += x[i] * x[i];
    }
    ss /= n;
    ss += eps;
    ss = 1.0f / std::sqrt(ss);

    for (size_t i = 0; i < n; ++i) {
        y[i] = x[i] * ss * w[i];
    }
}

void Linear::forward(const Tensor& input, Tensor& output) {
    // Assuming input is a vector (single token inference)
    // weight is [out_features, in_features]
    // x is [in_features]
    // y is [out_features]
    
    int in_features = weight->get_shape()[0];
    int out_features = weight->get_shape()[1];
    
    const float* x = static_cast<const float*>(input.get_data());
    float* y = static_cast<float*>(output.get_data());
    
    if (weight->get_type() == DataType::Q4_0) {
        gemv_q4_0(out_features, in_features, static_cast<const block_q4_0*>(weight->get_data()), x, y);
    } else if (weight->get_type() == DataType::F32) {
        const float* w = static_cast<const float*>(weight->get_data());
        for (int i = 0; i < out_features; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < in_features; ++j) {
                sum += w[i * in_features + j] * x[j];
            }
            y[i] = sum;
        }
    } else {
        // TODO: Support other types (F16, etc.)
    }
    
    if (bias) {
        const float* b = static_cast<const float*>(bias->get_data());
        for (int i = 0; i < out_features; ++i) {
            y[i] += b[i];
        }
    }
}

TransformerBlock::TransformerBlock(int layer_id) : id(layer_id) {}

void TransformerBlock::load_weights(const std::map<std::string, std::shared_ptr<Tensor>>& model_tensors) {
    std::string prefix = "blk." + std::to_string(id) + ".";
    
    attn_norm = std::make_shared<RMSNorm>(model_tensors.at(prefix + "attn_norm.weight"));
    wq = std::make_shared<Linear>(model_tensors.at(prefix + "attn_q.weight"));
    wk = std::make_shared<Linear>(model_tensors.at(prefix + "attn_k.weight"));
    wv = std::make_shared<Linear>(model_tensors.at(prefix + "attn_v.weight"));
    wo = std::make_shared<Linear>(model_tensors.at(prefix + "attn_output.weight"));
    
    ffn_norm = std::make_shared<RMSNorm>(model_tensors.at(prefix + "ffn_norm.weight"));
    w1 = std::make_shared<Linear>(model_tensors.at(prefix + "ffn_gate.weight")); // MLP Gate
    w2 = std::make_shared<Linear>(model_tensors.at(prefix + "ffn_down.weight")); // MLP Down
    w3 = std::make_shared<Linear>(model_tensors.at(prefix + "ffn_up.weight"));   // MLP Up
}

void TransformerBlock::forward(InferenceContext& ctx) {
    // Hidden states are in ctx.workspace1
    Tensor hidden_states({(uint64_t)ctx.n_embd}, DataType::F32, ctx.workspace1.data());
    
    // 1. Self Attention
    // Copy hidden states to workspace2 for residual
    std::copy(ctx.workspace1.begin(), ctx.workspace1.end(), ctx.workspace2.begin());

    // norm
    attn_norm->forward(hidden_states, hidden_states);
    
    // ... projections would happen here ...
    
    // Residual connection: workspace1 += workspace2
    for (int i = 0; i < ctx.n_embd; ++i) {
        ctx.workspace1[i] += ctx.workspace2[i];
    }

    // 2. MLP
    // Copy current state for residual
    std::copy(ctx.workspace1.begin(), ctx.workspace1.end(), ctx.workspace2.begin());

    ffn_norm->forward(hidden_states, hidden_states);
    
    // ... MLP math ...
    
    // Residual connection
    for (int i = 0; i < ctx.n_embd; ++i) {
        ctx.workspace1[i] += ctx.workspace2[i];
    }
}

void TransformerBlock::unload_weights() {
    attn_norm.reset();
    wq.reset();
    wk.reset();
    wv.reset();
    wo.reset();

    ffn_norm.reset();
    w1.reset();
    w2.reset();
    w3.reset();
}

} // namespace layer_forge
