#include "httplib.h"
#include "json.hpp"
#include "engine.hpp"
#include <iostream>
#include <filesystem>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace layer_forge;

std::mutex engine_mutex;
std::unique_ptr<InferenceEngine> current_engine;
std::string current_model_path;

void log(const std::string& level, const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "[%Y-%m-%d %H:%M:%S] ");
    std::cerr << ss.str() << level << " " << msg << std::endl;
}

std::string format_prompt(const json& messages) {
    std::stringstream ss;
    for (const auto& msg : messages) {
        std::string role = msg["role"];
        std::string content = msg["content"];
        ss << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
    }
    ss << "<|im_start|>assistant\n";
    return ss.str();
}

// Mock tokenizer: in a real scenario, we'd use the GGUF vocabulary to encode strings.
// For this prototype, we'll use a very basic mock that converts the prompt to dummy IDs.
std::vector<int> mock_tokenize(const std::string& prompt) {
    // Just a placeholder: in a real implementation, this would use a BPE or SentencePiece tokenizer.
    // For now, we'll just pass a few dummy tokens to trigger the engine.
    (void)prompt;
    return {1, 2, 3}; 
}

#include <csignal>

httplib::Server* global_svr = nullptr;

void signal_handler(int signal) {
    if (signal == SIGINT && global_svr) {
        log("INFO", "SIGINT received, shutting down server...");
        global_svr->stop();
    }
}

void start_server(const std::string& host, int port, const std::string& models_dir) {
    httplib::Server svr;
    global_svr = &svr;
    std::signal(SIGINT, signal_handler);

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        log("INFO", "GET /v1/models");
        json models = json::array();
        
        for (const auto& entry : fs::directory_iterator(models_dir)) {
            if (entry.path().extension() == ".gguf") {
                models.push_back({
                    {"id", entry.path().stem().string()},
                    {"object", "model"},
                    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                    {"owned_by", "layer-forge"}
                });
            }
        }

        res.set_content(json({{"object", "list"}, {"data", models}}).dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(json({{"error", {{"message", "Invalid JSON"}, {"type", "invalid_request_error"}}}}).dump(), "application/json");
            return;
        }

        std::string model_id = body.value("model", "");
        bool stream = body.value("stream", false);
        log("INFO", "POST /v1/chat/completions model=" + model_id + " stream=" + (stream ? "true" : "false"));

        std::string model_path = models_dir + "/" + model_id + ".gguf";
        if (!fs::exists(model_path)) {
            log("ERROR", "Model not found: " + model_id);
            res.status = 404;
            res.set_content(json({{"error", {{"message", "Model not found"}, {"type", "invalid_request_error"}}}}).dump(), "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(engine_mutex);
        if (current_model_path != model_path) {
            log("INFO", "Loading model: " + model_path);
            current_engine = std::make_unique<InferenceEngine>(model_path);
            if (!current_engine->init()) {
                res.status = 500;
                res.set_content(json({{"error", {{"message", "Failed to initialize model"}, {"type", "server_error"}}}}).dump(), "application/json");
                return;
            }
            current_model_path = model_path;
        }

        std::string prompt = format_prompt(body["messages"]);
        std::vector<int> input_ids = mock_tokenize(prompt);
        int max_tokens = body.value("max_tokens", 128);

        if (stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_chunked_content_provider("text/event-stream", [&](size_t, httplib::DataSink& sink) {
                current_engine->generate(input_ids, max_tokens, [&](const std::string& token) {
                    json chunk = {
                        {"id", "chatcmpl-123"},
                        {"object", "chat.completion.chunk"},
                        {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                        {"model", model_id},
                        {"choices", {{
                            {"index", 0},
                            {"delta", {{"content", token}}},
                            {"finish_reason", nullptr}
                        }}}
                    };
                    std::string data = "data: " + chunk.dump() + "\n\n";
                    sink.write(data.c_str(), data.size());
                    return true;
                });
                sink.write("data: [DONE]\n\n", 13);
                sink.done();
                return true;
            });
        } else {
            std::string full_content;
            current_engine->generate(input_ids, max_tokens, [&](const std::string& token) {
                full_content += token;
                return true;
            });

            json response = {
                {"id", "chatcmpl-123"},
                {"object", "chat.completion"},
                {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
                {"model", model_id},
                {"choices", {{
                    {"index", 0},
                    {"message", {{"role", "assistant"}, {"content", full_content}}},
                    {"finish_reason", "stop"}
                }}},
                {"usage", {{"prompt_tokens", input_ids.size()}, {"completion_tokens", 0}, {"total_tokens", 0}}}
            };
            res.set_content(response.dump(), "application/json");
        }
    });

    log("INFO", "Server listening on " + host + ":" + std::to_string(port));
    svr.listen(host.c_str(), port);
}
