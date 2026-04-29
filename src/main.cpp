#include <iostream>
#include "engine.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./forge_cli <model_path.gguf>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    layer_forge::InferenceEngine engine(model_path);

    std::cout << "Initializing Layer Forge with model: " << model_path << std::endl;
    if (!engine.init()) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return 1;
    }

    std::cout << "Engine initialized successfully." << std::endl;
    
    // Test generation with a dummy prompt
    std::vector<int> prompt = {1, 2, 3}; 
    engine.generate(prompt, 10);

    return 0;
}
