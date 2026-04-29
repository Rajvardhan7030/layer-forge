#pragma once

#include <string>
#include <vector>
#include <memory>

namespace layer_forge {

class Tensor {
public:
    // Basic tensor abstraction to be expanded
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward(const Tensor& input, Tensor& output) = 0;
};

class Model {
public:
    virtual ~Model() = default;
    virtual void load_metadata(const std::string& path) = 0;
    virtual void inference_loop() = 0;
};

} // namespace layer_forge
