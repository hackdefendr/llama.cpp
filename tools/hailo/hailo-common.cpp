#include "hailo-common.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <random>
#include <sstream>
#include <iomanip>

static std::atomic<bool> g_interrupted{false};

const char * hailo_status_to_string(hailo_status status) {
    switch (status) {
        case HAILO_SUCCESS:                return "HAILO_SUCCESS";
        case HAILO_INVALID_ARGUMENT:       return "HAILO_INVALID_ARGUMENT";
        case HAILO_OUT_OF_HOST_MEMORY:     return "HAILO_OUT_OF_HOST_MEMORY";
        case HAILO_TIMEOUT:                return "HAILO_TIMEOUT";
        case HAILO_INSUFFICIENT_BUFFER:    return "HAILO_INSUFFICIENT_BUFFER";
        case HAILO_INVALID_OPERATION:      return "HAILO_INVALID_OPERATION";
        case HAILO_NOT_FOUND:              return "HAILO_NOT_FOUND";
        case HAILO_INTERNAL_FAILURE:       return "HAILO_INTERNAL_FAILURE";
        default:                           return "HAILO_UNKNOWN";
    }
}

std::string generate_completion_id() {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream ss;
    ss << "chatcmpl-" << std::hex << dist(rng);
    return ss.str();
}

nlohmann::json format_chat_completion(
    const std::string & id,
    const std::string & model,
    const std::string & content,
    const std::string & finish_reason,
    int prompt_tokens,
    int completion_tokens)
{
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    return {
        {"id",      id},
        {"object",  "chat.completion"},
        {"created", epoch},
        {"model",   model},
        {"choices", nlohmann::json::array({
            {
                {"index",          0},
                {"message",        {{"role", "assistant"}, {"content", content}}},
                {"finish_reason",  finish_reason}
            }
        })},
        {"usage", {
            {"prompt_tokens",     prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens",      prompt_tokens + completion_tokens}
        }}
    };
}

nlohmann::json format_chat_completion_chunk(
    const std::string & id,
    const std::string & model,
    const std::string & delta_content,
    const std::string & finish_reason)
{
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    nlohmann::json delta;
    if (!delta_content.empty()) {
        delta = {{"role", "assistant"}, {"content", delta_content}};
    }

    nlohmann::json choice = {
        {"index", 0},
        {"delta", delta}
    };
    if (!finish_reason.empty()) {
        choice["finish_reason"] = finish_reason;
    } else {
        choice["finish_reason"] = nullptr;
    }

    return {
        {"id",      id},
        {"object",  "chat.completion.chunk"},
        {"created", epoch},
        {"model",   model},
        {"choices", nlohmann::json::array({choice})}
    };
}

static void signal_handler(int signum) {
    (void)signum;
    g_interrupted.store(true);
}

void setup_signal_handlers() {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);
}

bool is_interrupted() {
    return g_interrupted.load();
}

void set_interrupted(bool value) {
    g_interrupted.store(value);
}
