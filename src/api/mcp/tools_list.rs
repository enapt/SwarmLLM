use serde_json::{json, Value};

use super::types::JsonRpcResponse;

pub(super) fn handle_tools_list(id: Option<Value>) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "tools": [
                {
                    "name": "chat",
                    "description": "Send a chat completion request to the SwarmLLM inference engine",
                    "annotations": {
                        "title": "Chat Completion",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "Model ID to use for inference"
                            },
                            "messages": {
                                "type": "array",
                                "description": "Array of chat messages",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": { "type": "string", "enum": ["system", "user", "assistant"] },
                                        "content": { "type": "string" }
                                    },
                                    "required": ["role", "content"]
                                }
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature (0.0-2.0)"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate"
                            }
                        },
                        "required": ["model", "messages"]
                    }
                },
                {
                    "name": "models",
                    "description": "List available models on this SwarmLLM node and network",
                    "annotations": {
                        "title": "List Models",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": true,
                        "openWorldHint": false
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "compare",
                    "description": "Send the same prompt to multiple models concurrently and return all responses side-by-side for comparison. Supports local, network, and cloud models.",
                    "annotations": {
                        "title": "Compare Models",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to send to all models"
                            },
                            "system": {
                                "type": "string",
                                "description": "Optional system prompt"
                            },
                            "models": {
                                "type": "array",
                                "description": "Array of model IDs to compare (e.g. [\"qwen2.5-coder-7b\", \"gpt-5.4\", \"claude-sonnet-5\"])",
                                "items": { "type": "string" }
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature (0.0-2.0, default 0.7)"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens per response (default 1024)"
                            }
                        },
                        "required": ["prompt", "models"]
                    }
                },
                {
                    "name": "research",
                    "description": "Fan out a research question to multiple models in parallel and collect all responses. Designed for knowledge gathering — send a question to cheap/fast models to get diverse perspectives without using expensive model tokens. Each model's response is returned separately with latency and token usage.",
                    "annotations": {
                        "title": "Research (Multi-Model)",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The research question to send to all models"
                            },
                            "system": {
                                "type": "string",
                                "description": "Optional system prompt to guide research focus"
                            },
                            "models": {
                                "type": "array",
                                "description": "Array of model IDs to query. If omitted, uses all available models (local + cloud).",
                                "items": { "type": "string" }
                            },
                            "max_models": {
                                "type": "integer",
                                "description": "Maximum number of models to query when models is omitted (default 5)"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens per response (default 2048)"
                            }
                        },
                        "required": ["question"]
                    }
                },
                {
                    "name": "batch_prompts",
                    "description": "Execute multiple independent prompts in parallel, each targeting a specific model. Returns all results once complete. Ideal for offloading parallel subtasks — e.g., ask one model to summarize, another to translate, another to review code, all at once.",
                    "annotations": {
                        "title": "Batch Prompts (Parallel)",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "description": "Array of independent prompt tasks to execute in parallel",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "string",
                                            "description": "Caller-defined ID for this task (returned in results for matching)"
                                        },
                                        "model": {
                                            "type": "string",
                                            "description": "Model ID to use for this task"
                                        },
                                        "prompt": {
                                            "type": "string",
                                            "description": "The prompt to send"
                                        },
                                        "system": {
                                            "type": "string",
                                            "description": "Optional system prompt"
                                        },
                                        "max_tokens": {
                                            "type": "integer",
                                            "description": "Max tokens for this task (default 1024)"
                                        },
                                        "temperature": {
                                            "type": "number",
                                            "description": "Temperature for this task (default 0.7)"
                                        }
                                    },
                                    "required": ["id", "model", "prompt"]
                                }
                            }
                        },
                        "required": ["tasks"]
                    }
                },
                {
                    "name": "delegate",
                    "description": "Offload a task to the most appropriate model based on a tier preference. Tiers: 'fast' picks the lowest-latency local model, 'cheap' picks a small/free model, 'smart' picks the most capable available model (may use cloud). Saves your subscription tokens by routing routine work to local/cheap models automatically.",
                    "annotations": {
                        "title": "Delegate Task",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": false,
                        "openWorldHint": true
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The task/prompt to delegate"
                            },
                            "tier": {
                                "type": "string",
                                "enum": ["fast", "cheap", "smart"],
                                "description": "Model selection strategy: 'fast' = lowest latency, 'cheap' = smallest/free model, 'smart' = most capable"
                            },
                            "system": {
                                "type": "string",
                                "description": "Optional system prompt"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate (default 1024)"
                            }
                        },
                        "required": ["prompt", "tier"]
                    }
                },
                {
                    "name": "node_info",
                    "description": "Get detailed information about this SwarmLLM node: loaded models, connected peers, VRAM/disk usage, credit balance, available cloud providers, and network status.",
                    "annotations": {
                        "title": "Node Information",
                        "readOnlyHint": true,
                        "destructiveHint": false,
                        "idempotentHint": true,
                        "openWorldHint": false
                    },
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }),
    )
}
