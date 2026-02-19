#include "hailo-engine.h"
#include "hailo-common.h"

#include <sstream>

using namespace hailort;
using namespace hailort::genai;

HailoEngine::~HailoEngine() {
    shutdown();
}

bool HailoEngine::init(const std::string & hef_path, const std::string & model_name) {
    m_hef_path = hef_path;
    m_model_name = model_name;

    LOG_INF("Creating VDevice...");
    auto vdevice_exp = VDevice::create_shared();
    if (!vdevice_exp) {
        LOG_ERR("Failed to create VDevice: %s", hailo_status_to_string(vdevice_exp.status()));
        return false;
    }
    m_vdevice = vdevice_exp.release();

    LOG_INF("Loading model: %s", hef_path.c_str());
    auto llm_params = LLMParams(hef_path);
    auto llm_exp = LLM::create(m_vdevice, llm_params);
    if (!llm_exp) {
        LOG_ERR("Failed to create LLM: %s", hailo_status_to_string(llm_exp.status()));
        return false;
    }
    m_llm = std::make_unique<LLM>(std::move(llm_exp.release()));

    // Query max context capacity
    auto ctx_exp = m_llm->max_context_capacity();
    if (ctx_exp) {
        m_max_context = ctx_exp.release();
        LOG_INF("Max context capacity: %zu tokens", m_max_context);
    } else {
        LOG_WRN("Could not query max context capacity");
    }

    LOG_INF("Model loaded successfully: %s", m_model_name.c_str());
    return true;
}

void HailoEngine::shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_llm.reset();
    m_vdevice.reset();
}

std::string HailoEngine::generate_streaming(
    const std::vector<std::string> & messages_json,
    const std::function<bool(const std::string &)> & on_token,
    float temperature,
    float top_p,
    int max_tokens)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_llm) {
        LOG_ERR("Engine not initialized");
        return "stop";
    }

    // Clear context for fresh generation
    auto clear_status = m_llm->clear_context();
    if (clear_status != HAILO_SUCCESS) {
        LOG_ERR("Failed to clear context: %s", hailo_status_to_string(clear_status));
        return "stop";
    }

    // Create generator params
    auto params_exp = m_llm->create_generator_params();
    if (!params_exp) {
        LOG_ERR("Failed to create generator params: %s", hailo_status_to_string(params_exp.status()));
        return "stop";
    }
    auto params = params_exp.release();

    // Configure sampling parameters
    if (temperature > 0.0f) {
        params.set_do_sample(true);
        params.set_temperature(temperature);
    } else {
        params.set_do_sample(false);
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        params.set_top_p(top_p);
    }

    if (max_tokens > 0) {
        params.set_max_generated_tokens(static_cast<uint32_t>(max_tokens));
    }

    // Create generator, write prompt, and start generation
    auto gen_exp = m_llm->create_generator(params);
    if (!gen_exp) {
        LOG_ERR("Failed to create generator: %s", hailo_status_to_string(gen_exp.status()));
        return "stop";
    }
    auto generator = gen_exp.release();

    auto write_status = generator.write(messages_json);
    if (write_status != HAILO_SUCCESS) {
        LOG_ERR("Failed to write prompt: %s", hailo_status_to_string(write_status));
        return "stop";
    }

    auto completion_exp = generator.generate();
    if (!completion_exp) {
        LOG_ERR("Failed to start generation: %s", hailo_status_to_string(completion_exp.status()));
        return "stop";
    }
    auto completion = completion_exp.release();

    // Token read loop
    std::string finish_reason = "stop";

    while (true) {
        auto status = completion.generation_status();
        if (status == LLMGeneratorCompletion::Status::LOGICAL_END_OF_GENERATION) {
            finish_reason = "stop";
            break;
        }
        if (status == LLMGeneratorCompletion::Status::MAX_TOKENS_REACHED) {
            finish_reason = "length";
            break;
        }
        if (status == LLMGeneratorCompletion::Status::ABORTED) {
            finish_reason = "stop";
            break;
        }

        auto token_exp = completion.read();
        if (!token_exp) {
            LOG_ERR("Token read failed: %s", hailo_status_to_string(token_exp.status()));
            break;
        }
        auto token = token_exp.release();

        // Check if this is the last token (EOS) — don't forward it
        auto post_status = completion.generation_status();
        if (post_status != LLMGeneratorCompletion::Status::GENERATING) {
            if (post_status == LLMGeneratorCompletion::Status::MAX_TOKENS_REACHED) {
                finish_reason = "length";
            } else {
                finish_reason = "stop";
            }
            break;
        }

        // Forward token to callback
        if (!on_token(token)) {
            // Callback requested abort
            completion.abort();
            finish_reason = "stop";
            break;
        }
    }

    return finish_reason;
}

std::pair<std::string, std::string> HailoEngine::generate(
    const std::vector<std::string> & messages_json,
    float temperature,
    float top_p,
    int max_tokens)
{
    std::ostringstream full_response;

    auto finish_reason = generate_streaming(
        messages_json,
        [&full_response](const std::string & token) -> bool {
            full_response << token;
            return true;
        },
        temperature, top_p, max_tokens);

    return {full_response.str(), finish_reason};
}

std::string HailoEngine::model_name() const {
    return m_model_name;
}

size_t HailoEngine::max_context() const {
    return m_max_context;
}
