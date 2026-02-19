# Hailo 10H NPU Integration for llama.cpp

This integration adds standalone tools that run LLM inference on the **Hailo 10H NPU** accelerator via Hailo's GenAI LLM API. It provides an **OpenAI-compatible HTTP server** and an **interactive CLI** — both powered entirely by the Hailo NPU.

## Architecture

Unlike traditional ggml backends (CUDA, Vulkan, etc.) that execute individual tensor operations, the Hailo NPU runs pre-compiled whole models (HEF files) as black boxes. The entire LLM pipeline — tokenization, prefill, decode, sampling — is offloaded to the device.

The tools live under `tools/hailo/` and are independent of the ggml backend system:

```
tools/hailo/
├── CMakeLists.txt       # Build configuration
├── hailo-common.h/cpp   # Shared utilities (logging, JSON formatting, signals)
├── hailo-engine.h/cpp   # Core engine wrapping Hailo GenAI LLM API
├── hailo-server.cpp     # OpenAI-compatible HTTP server
└── hailo-cli.cpp        # Interactive terminal chat
```

## Prerequisites

- **Hailo 10H** module installed on a Raspberry Pi 5 (or compatible host)
- **HailoRT SDK** installed (`libhailort.so` and headers under `/usr/local/`)
- A compiled **HEF model file** (e.g. `Qwen2.5-1.5B-Instruct.hef`)

## Building

```bash
cmake -B build -DLLAMA_HAILO=ON
cmake --build build --target llama-hailo-server llama-hailo-cli -j$(nproc)
```

The binaries are placed in `build/bin/`:
- `llama-hailo-server`
- `llama-hailo-cli`

## Usage

### Server

```bash
./build/bin/llama-hailo-server --model /path/to/model.hef [options]
```

| Option | Default | Description |
|---|---|---|
| `--model <path>` | *(required)* | Path to HEF model file |
| `--model-name <name>` | filename stem | Display name in API responses |
| `--host <addr>` | `127.0.0.1` | Listen address |
| `--port <n>` | `8080` | Listen port |

The server exposes three endpoints:

- `GET /health` — Health check
- `GET /v1/models` — List loaded models (OpenAI format)
- `POST /v1/chat/completions` — Chat completions (OpenAI format, streaming and non-streaming)

### CLI

```bash
./build/bin/llama-hailo-cli --model /path/to/model.hef [options]
```

| Option | Default | Description |
|---|---|---|
| `--model <path>` | *(required)* | Path to HEF model file |
| `--model-name <name>` | filename stem | Display name |
| `--temperature <f>` | `0.7` | Sampling temperature |
| `--top-p <f>` | `0.9` | Top-p (nucleus) sampling |
| `--max-tokens <n>` | `-1` (unlimited) | Max tokens to generate |

Commands during chat:
- `/clear` — Reset conversation history
- `/quit` — Exit

## Smoke Tests

### Health check

```bash
curl -s http://localhost:8080/health
```

```json
{"status":"ok"}
```

### List models

```bash
curl -s http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen2.5-1.5B-Instruct",
      "object": "model",
      "owned_by": "hailo"
    }
  ]
}
```

### Non-streaming chat completion

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "max_tokens": 32
  }' | python3 -m json.tool
```

```json
{
    "id": "chatcmpl-f3b45c6ebd0194ee",
    "object": "chat.completion",
    "created": 1771512553,
    "model": "Qwen2.5-1.5B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "4."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
}
```

### Streaming chat completion

```bash
curl -s -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in French."}],
    "stream": true,
    "max_tokens": 20
  }'
```

Each token arrives as a Server-Sent Event:

```
data: {"choices":[{"delta":{"content":"Bon","role":"assistant"},"finish_reason":null,"index":0}],"created":1771512562,"id":"chatcmpl-42849cb6ea273a91","model":"Qwen2.5-1.5B-Instruct","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"jour","role":"assistant"},"finish_reason":null,"index":0}],"created":1771512562,"id":"chatcmpl-42849cb6ea273a91","model":"Qwen2.5-1.5B-Instruct","object":"chat.completion.chunk"}

...

data: {"choices":[{"delta":null,"finish_reason":"length","index":0}],"created":1771512565,"id":"chatcmpl-42849cb6ea273a91","model":"Qwen2.5-1.5B-Instruct","object":"chat.completion.chunk"}

data: [DONE]
```

### Interactive CLI session

```
$ ./build/bin/llama-hailo-cli --model /path/to/Qwen2.5-1.5B-Instruct.hef
[INFO]  Creating VDevice...
[INFO]  Loading model: /path/to/Qwen2.5-1.5B-Instruct.hef
[INFO]  Model loaded successfully: Qwen2.5-1.5B-Instruct
Model loaded: Qwen2.5-1.5B-Instruct
Type a message to chat. Commands: /clear, /quit

> Hello!
Hello! How can I assist you today?

> What is the capital of France?
The capital of France is Paris.

> /clear
Conversation cleared.

> /quit
Shutting down...
```

## Supported Request Parameters

The `/v1/chat/completions` endpoint accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | array | *(required)* | Array of `{role, content}` message objects |
| `stream` | bool | `false` | Enable SSE streaming |
| `temperature` | float | `0.7` | Sampling temperature (0 = greedy) |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `max_tokens` | int | `-1` | Max tokens to generate (-1 = model default) |

## Notes

- The Hailo NPU supports **one active generation at a time**. Concurrent requests to the server are serialized via a mutex.
- Model loading takes 20-40 seconds on first start as the HEF is loaded onto the device.
- The `usage` field in responses currently reports zero token counts since the Hailo API does not expose tokenization metrics during generation.
- Context is cleared between server requests. The CLI maintains multi-turn conversation history.
