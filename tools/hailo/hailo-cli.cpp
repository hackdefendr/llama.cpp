#include "hailo-common.h"
#include "hailo-engine.h"

#include "json.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --model <hef_path> [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model <path>       Path to HEF model file (required)\n");
    fprintf(stderr, "  --model-name <name>  Display name for the model (default: filename)\n");
    fprintf(stderr, "  --temperature <f>    Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  --top-p <f>          Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  --max-tokens <n>     Max tokens to generate (default: -1, unlimited)\n");
}

int main(int argc, char ** argv) {
    std::string hef_path;
    std::string model_name;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int max_tokens = -1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model") && i + 1 < argc) {
            hef_path = argv[++i];
        } else if ((arg == "--model-name") && i + 1 < argc) {
            model_name = argv[++i];
        } else if ((arg == "--temperature") && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if ((arg == "--top-p") && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if ((arg == "--max-tokens") && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
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

    if (model_name.empty()) {
        model_name = std::filesystem::path(hef_path).stem().string();
    }

    setup_signal_handlers();

    HailoEngine engine;
    if (!engine.init(hef_path, model_name)) {
        LOG_ERR("Failed to initialize engine");
        return 1;
    }

    printf("Model loaded: %s\n", model_name.c_str());
    printf("Type a message to chat. Commands: /clear, /quit\n\n");

    std::vector<std::string> conversation_history;
    std::string line;

    while (true) {
        printf("> ");
        fflush(stdout);

        if (!std::getline(std::cin, line)) {
            // EOF (Ctrl+D)
            break;
        }

        if (is_interrupted()) {
            break;
        }

        // Trim whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            continue;
        }
        line = line.substr(start);
        auto end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) {
            line = line.substr(0, end + 1);
        }

        if (line.empty()) {
            continue;
        }

        // Handle commands
        if (line == "/quit" || line == "/exit") {
            break;
        }

        if (line == "/clear") {
            conversation_history.clear();
            printf("Conversation cleared.\n\n");
            continue;
        }

        // Build user message JSON and add to history
        json user_msg = {{"role", "user"}, {"content", line}};
        conversation_history.push_back(user_msg.dump());

        // Reset interrupt flag for this generation
        set_interrupted(false);

        // Stream response
        std::string assistant_response;

        auto finish_reason = engine.generate_streaming(
            conversation_history,
            [&assistant_response](const std::string & token) -> bool {
                printf("%s", token.c_str());
                fflush(stdout);
                assistant_response += token;
                return !is_interrupted();
            },
            temperature, top_p, max_tokens);

        printf("\n\n");

        // Add assistant response to conversation history for multi-turn
        if (!assistant_response.empty()) {
            json assistant_msg = {{"role", "assistant"}, {"content", assistant_response}};
            conversation_history.push_back(assistant_msg.dump());
        }

        (void)finish_reason;

        // Reset interrupt flag after generation
        set_interrupted(false);
    }

    printf("\nShutting down...\n");
    engine.shutdown();
    return 0;
}
