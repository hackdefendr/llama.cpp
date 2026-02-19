#pragma once

#include <atomic>
#include <cstdio>
#include <string>

#include "json.hpp"
#include "hailo/hailort.h"

// Logging macros
#define LOG_INF(fmt, ...) fprintf(stderr, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define LOG_WRN(fmt, ...) fprintf(stderr, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

// Map hailo_status codes to readable strings
const char * hailo_status_to_string(hailo_status status);

// Generate a unique completion ID like "chatcmpl-<hex>"
std::string generate_completion_id();

// Build an OpenAI-compatible chat completion JSON response
nlohmann::json format_chat_completion(
    const std::string & id,
    const std::string & model,
    const std::string & content,
    const std::string & finish_reason,
    int prompt_tokens,
    int completion_tokens);

// Build an SSE streaming chunk JSON (delta format)
nlohmann::json format_chat_completion_chunk(
    const std::string & id,
    const std::string & model,
    const std::string & delta_content,
    const std::string & finish_reason);

// Signal handling
void setup_signal_handlers();
bool is_interrupted();
void set_interrupted(bool value);
