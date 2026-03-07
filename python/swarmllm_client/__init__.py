"""SwarmLLM Python client SDK.

Provides sync and async clients for the SwarmLLM OpenAI-compatible API
and SwarmLLM-specific admin/pool/identity endpoints.

Quick start::

    from swarmllm_client import SwarmLLM

    client = SwarmLLM("http://localhost:8800", api_key="your-key")

    # Chat completion
    response = client.chat("Hello!", model="qwen2.5-coder-7b")
    print(response.content)

    # Streaming
    for chunk in client.chat("Tell me a story", stream=True):
        print(chunk, end="", flush=True)

    # Admin
    print(client.admin.stats())
    print(client.admin.peers())

Alternatively, use the official OpenAI SDK::

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8800/v1", api_key="your-key")
"""

from swarmllm_client.client import SwarmLLM, SwarmLLMError
from swarmllm_client.async_client import AsyncSwarmLLM
from swarmllm_client.types import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    Model,
    NodeStats,
    PeerInfo,
    CreditInfo,
    ShardStorage,
    Usage,
)

# Optional framework integrations (only available if deps are installed)
try:
    from swarmllm_client.integrations.langchain import ChatSwarmLLM
except ImportError:
    pass

try:
    from swarmllm_client.integrations.llamaindex import SwarmLLM as LlamaIndexSwarmLLM
except ImportError:
    pass

__version__ = "0.1.0"
__all__ = [
    "SwarmLLM",
    "AsyncSwarmLLM",
    "SwarmLLMError",
    "ChatMessage",
    "ChatResponse",
    "EmbeddingResponse",
    "Model",
    "NodeStats",
    "PeerInfo",
    "CreditInfo",
    "ShardStorage",
    "Usage",
]
