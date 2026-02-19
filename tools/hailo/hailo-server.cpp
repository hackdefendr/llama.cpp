#include "hailo-common.h"
#include "hailo-engine.h"

#include "json.hpp"
#include "cpp-httplib/httplib.h"

#include <cstdlib>
#include <filesystem>
#include <string>

using json = nlohmann::json;

static HailoEngine g_engine;
static httplib::Server g_server;

static void handle_health(const httplib::Request &, httplib::Response & res) {
    res.set_content(R"({"status":"ok"})", "application/json");
}

static void handle_models(const httplib::Request &, httplib::Response & res) {
    json model_obj = {
        {"id",       g_engine.model_name()},
        {"object",   "model"},
        {"owned_by", "hailo"}
    };

    json response = {
        {"object", "list"},
        {"data",   json::array({model_obj})}
    };

    res.set_content(response.dump(), "application/json");
}

static void handle_chat_completions(const httplib::Request & req, httplib::Response & res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::exception & e) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"Invalid JSON","type":"invalid_request_error"}})", "application/json");
        return;
    }

    // Extract messages
    if (!body.contains("messages") || !body["messages"].is_array()) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"messages field is required and must be an array","type":"invalid_request_error"}})", "application/json");
        return;
    }

    std::vector<std::string> messages_json;
    for (const auto & msg : body["messages"]) {
        messages_json.push_back(msg.dump());
    }

    // Extract parameters
    bool stream = body.value("stream", false);
    float temperature = body.value("temperature", 0.7f);
    float top_p = body.value("top_p", 0.9f);
    int max_tokens = body.value("max_tokens", -1);

    if (stream) {
        // Streaming response using SSE
        std::string completion_id = generate_completion_id();
        std::string model = g_engine.model_name();

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        auto chunked_provider = [completion_id, model, messages_json, temperature, top_p, max_tokens](
            size_t /* offset */, httplib::DataSink & sink) -> bool
        {
            std::string finish_reason;

            finish_reason = g_engine.generate_streaming(
                messages_json,
                [&sink, &completion_id, &model](const std::string & token) -> bool {
                    auto chunk = format_chat_completion_chunk(completion_id, model, token, "");
                    std::string sse = "data: " + chunk.dump() + "\n\n";
                    return sink.write(sse.c_str(), sse.size());
                },
                temperature, top_p, max_tokens);

            // Send final chunk with finish_reason
            auto final_chunk = format_chat_completion_chunk(completion_id, model, "", finish_reason);
            std::string sse = "data: " + final_chunk.dump() + "\n\n";
            sink.write(sse.c_str(), sse.size());

            // Send [DONE]
            std::string done = "data: [DONE]\n\n";
            sink.write(done.c_str(), done.size());

            sink.done();
            return true;
        };

        res.set_chunked_content_provider("text/event-stream", chunked_provider);

    } else {
        // Non-streaming response
        auto [content, finish_reason] = g_engine.generate(
            messages_json, temperature, top_p, max_tokens);

        std::string completion_id = generate_completion_id();
        auto response = format_chat_completion(
            completion_id, g_engine.model_name(),
            content, finish_reason, 0, 0);

        res.set_content(response.dump(), "application/json");
    }
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --model <hef_path> [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model <path>       Path to HEF model file (required)\n");
    fprintf(stderr, "  --model-name <name>  Display name for the model (default: filename)\n");
    fprintf(stderr, "  --host <addr>        Listen address (default: 127.0.0.1)\n");
    fprintf(stderr, "  --port <n>           Listen port (default: 8080)\n");
}

int main(int argc, char ** argv) {
    std::string hef_path;
    std::string model_name;
    std::string host = "127.0.0.1";
    int port = 8080;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model") && i + 1 < argc) {
            hef_path = argv[++i];
        } else if ((arg == "--model-name") && i + 1 < argc) {
            model_name = argv[++i];
        } else if ((arg == "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if ((arg == "--port") && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (hef_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Default model name to filename without extension
    if (model_name.empty()) {
        model_name = std::filesystem::path(hef_path).stem().string();
    }

    setup_signal_handlers();

    if (!g_engine.init(hef_path, model_name)) {
        LOG_ERR("Failed to initialize engine");
        return 1;
    }

    // Register routes
    g_server.Get("/health",              handle_health);
    g_server.Get("/v1/models",           handle_models);
    g_server.Post("/v1/chat/completions", handle_chat_completions);

    // Graceful shutdown on signal
    std::thread signal_thread([] {
        while (!is_interrupted()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        LOG_INF("Signal received, shutting down...");
        g_server.stop();
    });
    signal_thread.detach();

    LOG_INF("Starting server on %s:%d", host.c_str(), port);
    LOG_INF("Model: %s (%s)", model_name.c_str(), hef_path.c_str());

    if (!g_server.listen(host, port)) {
        LOG_ERR("Failed to start server on %s:%d", host.c_str(), port);
        return 1;
    }

    g_engine.shutdown();
    LOG_INF("Server stopped");
    return 0;
}
