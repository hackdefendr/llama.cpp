#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "hailo/vdevice.hpp"
#include "hailo/genai/llm/llm.hpp"

class HailoEngine {
public:
    HailoEngine() = default;
    ~HailoEngine();

    HailoEngine(const HailoEngine &) = delete;
    HailoEngine & operator=(const HailoEngine &) = delete;

    // Initialize the engine with a HEF model file and display name
    bool init(const std::string & hef_path, const std::string & model_name);

    // Shut down the engine and release resources
    void shutdown();

    // Streaming generation: calls on_token(token_string) for each token.
    // If on_token returns false, generation is aborted.
    // Returns finish_reason: "stop" or "length"
    std::string generate_streaming(
        const std::vector<std::string> & messages_json,
        const std::function<bool(const std::string &)> & on_token,
        float temperature,
        float top_p,
        int max_tokens);

    // Non-streaming generation: returns {full_response_text, finish_reason}
    std::pair<std::string, std::string> generate(
        const std::vector<std::string> & messages_json,
        float temperature,
        float top_p,
        int max_tokens);

    std::string model_name() const;
    size_t max_context() const;

private:
    std::shared_ptr<hailort::VDevice> m_vdevice;
    std::unique_ptr<hailort::genai::LLM> m_llm;
    std::string m_model_name;
    std::string m_hef_path;
    mutable std::mutex m_mutex;
    size_t m_max_context = 0;
};
