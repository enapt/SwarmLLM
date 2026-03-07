"""LangChain integration for SwarmLLM.

Usage::

    from swarmllm_client.integrations.langchain import ChatSwarmLLM

    llm = ChatSwarmLLM(base_url="http://localhost:8800", model_name="TinyLlama-1.1B")
    response = llm.invoke("What is SwarmLLM?")
    print(response.content)

    # Streaming
    for chunk in llm.stream("Tell me a story"):
        print(chunk.content, end="", flush=True)

    # In a chain
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
    chain = prompt | llm
    chain.invoke({"input": "Hello!"})
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
except ImportError:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install it with: pip install langchain-core"
    )

from swarmllm_client.client import SwarmLLM


def _message_to_dict(msg: BaseMessage) -> dict[str, str]:
    """Convert a LangChain message to an OpenAI-format dict."""
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    elif isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}
    else:
        return {"role": "user", "content": msg.content}


class ChatSwarmLLM(BaseChatModel):
    """LangChain chat model backed by a SwarmLLM node.

    Args:
        base_url: SwarmLLM node URL.
        api_key: Bearer token for authenticated endpoints.
        model_name: Model to use (auto-detected if not specified).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling threshold.
        timeout: Request timeout in seconds.
    """

    base_url: str = "http://localhost:8800"
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    timeout: float = 120.0

    @property
    def _llm_type(self) -> str:
        return "swarmllm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _get_client(self) -> SwarmLLM:
        return SwarmLLM(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = self._get_client()
        msg_dicts = [_message_to_dict(m) for m in messages]

        response = client.chat_completion(
            messages=msg_dicts,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=stop,
            **kwargs,
        )

        generation = ChatGeneration(
            message=AIMessage(content=response.content),
            generation_info={
                "finish_reason": response.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            },
        )
        return ChatResult(
            generations=[generation],
            llm_output={"model": response.model},
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        client = self._get_client()
        msg_dicts = [_message_to_dict(m) for m in messages]

        chunks = client.chat_completion(
            messages=msg_dicts,
            model=self.model_name,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=stop,
            **kwargs,
        )

        for text in chunks:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=text),
            )
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk
