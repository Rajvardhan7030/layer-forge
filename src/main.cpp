#include <iostream>
#include <string>
#include <vector>
#include "engine.hpp"

void start_server(const std::string& host, int port, const std::string& models_dir);

int main(int argc, char** argv) {
    bool server_mode = false;
    int port = 8080;
    std::string models_dir = ".";
    std::string model_path = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--server") {
            server_mode = true;
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--models-dir" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (model_path.empty() && arg.find("--") != 0) {
            model_path = arg;
        }
    }

    if (server_mode) {
        start_server("0.0.0.0", port, models_dir);
        return 0;
    }

    if (model_path.empty()) {
        std::cout << "Usage: ./forge_cli <model_path.gguf> OR ./forge_cli --server [--port 8080] [--models-dir ./models]" << std::endl;
        return 1;
    }

    layer_forge::InferenceEngine engine(model_path);

    std::cout << "Initializing Layer Forge with model: " << model_path << std::endl;
    if (!engine.init()) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return 1;
    }

    std::cout << "Engine initialized successfully." << std::endl;
    
    // Test generation with a dummy prompt
    std::vector<int> prompt = {1, 2, 3}; 
    engine.generate(prompt, 10, [](const std::string& token) {
        std::cout << token << std::flush;
        return true;
    });
    std::cout << std::endl;

    return 0;
}
