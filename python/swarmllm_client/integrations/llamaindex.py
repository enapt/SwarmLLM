"""LlamaIndex integration for SwarmLLM.

Usage::

    from swarmllm_client.integrations.llamaindex import SwarmLLM as LlamaIndexSwarmLLM

    llm = LlamaIndexSwarmLLM(
        base_url="http://localhost:8800",
        model_name="TinyLlama-1.1B",
    )
    response = llm.complete("What is SwarmLLM?")
    print(response.text)

    # Streaming
    for chunk in llm.stream_complete("Tell me a story"):
        print(chunk.delta, end="", flush=True)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    from llama_index.core.bridge.pydantic import Field
    from llama_index.core.llms import (
        ChatMessage as LIChatMessage,
        ChatResponse as LIChatResponse,
        ChatResponseGen,
        CompletionResponse,
        CompletionResponseGen,
        CustomLLM,
        LLMMetadata,
        MessageRole,
    )
    from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
except ImportError:
    raise ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install it with: pip install llama-index-core"
    )

from swarmllm_client.client import SwarmLLM as SwarmLLMClient


# What to tell LlamaIndex a reply might be worth, when we are deliberately NOT
# telling the server. `max_tokens` now defaults to None so the node sizes the
# reply against the model's real context window, but LlamaIndex's own prompt
# arithmetic needs a number for `num_output`, so it keeps the previous constant.
DEFAULT_NUM_OUTPUT = 2048


_ROLE_MAP = {
    MessageRole.SYSTEM: "system",
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
}


class SwarmLLM(CustomLLM):
    """LlamaIndex LLM backed by a SwarmLLM node.

    Args:
        base_url: SwarmLLM node URL.
        api_key: Bearer token for authenticated endpoints.
        model_name: Model to use (auto-detected if not specified).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        context_window: Context window size for LLMMetadata.
    """

    base_url: str = Field(default="http://localhost:8800")
    api_key: Optional[str] = Field(default=None)
    model_name: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    timeout: float = Field(default=120.0)
    context_window: int = Field(default=4096)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            # LlamaIndex sizes its own prompt maths with this and needs a
            # number; `max_tokens` is now None by default so the server can pick
            # against the real context window, so fall back to the previous
            # constant for the metadata only.
            num_output=self.max_tokens or DEFAULT_NUM_OUTPUT,
            model_name=self.model_name or "swarmllm",
        )

    def _get_client(self) -> SwarmLLMClient:
        return SwarmLLMClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        client = self._get_client()
        response = client.chat(
            prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        return CompletionResponse(
            text=response.content,
            raw=response.raw,
        )

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        client = self._get_client()
        chunks = client.chat(
            prompt,
            model=self.model_name,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        accumulated = ""

        def gen() -> CompletionResponseGen:
            nonlocal accumulated
            for text in chunks:
                accumulated += text
                yield CompletionResponse(
                    text=accumulated,
                    delta=text,
                )

        return gen()

    @llm_chat_callback()
    def chat(
        self,
        messages: Sequence[LIChatMessage],
        **kwargs: Any,
    ) -> LIChatResponse:
        client = self._get_client()
        msg_dicts = [
            {
                "role": _ROLE_MAP.get(m.role, "user"),
                "content": m.content,
            }
            for m in messages
        ]
        response = client.chat_completion(
            messages=msg_dicts,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        return LIChatResponse(
            message=LIChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.content,
            ),
            raw=response.raw,
        )

    @llm_chat_callback()
    def stream_chat(
        self,
        messages: Sequence[LIChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        client = self._get_client()
        msg_dicts = [
            {
                "role": _ROLE_MAP.get(m.role, "user"),
                "content": m.content,
            }
            for m in messages
        ]
        chunks = client.chat_completion(
            messages=msg_dicts,
            model=self.model_name,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        accumulated = ""

        def gen() -> ChatResponseGen:
            nonlocal accumulated
            for text in chunks:
                accumulated += text
                yield LIChatResponse(
                    message=LIChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=accumulated,
                    ),
                    delta=text,
                    raw={},
                )

        return gen()
