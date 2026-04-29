#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "tensor.hpp"
#include "context.hpp"

namespace layer_forge {

class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward(const Tensor& input, Tensor& output) = 0;
};

// RMSNorm Layer
class RMSNorm : public Layer {
public:
    RMSNorm(std::shared_ptr<Tensor> weight, float eps = 1e-6f)
        : weight(weight), eps(eps) {}
    void forward(const Tensor& input, Tensor& output) override;
private:
    std::shared_ptr<Tensor> weight;
    float eps;
};

// Linear/Dense Layer (Quantized)
class Linear : public Layer {
public:
    Linear(std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr)
        : weight(weight), bias(bias) {}
    void forward(const Tensor& input, Tensor& output) override;
private:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
};

// Transformer Block
class TransformerBlock {
public:
    TransformerBlock(int layer_id);
    void forward(InferenceContext& ctx);
    
    // Memory management
    void load_weights(const std::map<std::string, std::shared_ptr<Tensor>>& model_tensors);
    void unload_weights();

private:
    int id;
    std::shared_ptr<RMSNorm> attn_norm;
    std::shared_ptr<Linear> wq, wk, wv, wo;
    std::shared_ptr<RMSNorm> ffn_norm;
    std::shared_ptr<Linear> w1, w2, w3; // MLP layers
};

} // namespace layer_forge
