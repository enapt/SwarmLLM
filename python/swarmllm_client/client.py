"""Synchronous SwarmLLM client using requests."""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional, Union

import requests

from swarmllm_client.admin import AdminClient, IdentityClient, PoolClient
from swarmllm_client.types import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    Model,
    Usage,
)


class SwarmLLMError(Exception):
    """Error from the SwarmLLM API."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"SwarmLLM API error ({status_code}): {message}")


class SwarmLLM:
    """Synchronous SwarmLLM client.

    Provides access to the OpenAI-compatible API (chat completions, embeddings,
    models) and SwarmLLM-specific admin/identity/pool endpoints.

    Args:
        base_url: SwarmLLM node URL (e.g. "http://localhost:8800").
        api_key: Bearer token for authenticated endpoints.
        timeout: Request timeout in seconds.

    Example::

        client = SwarmLLM("http://localhost:8800", api_key="sk-...")
        response = client.chat("Hello!", model="qwen2.5-coder-7b")
        print(response.content)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8800",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

        self.admin = AdminClient(self)
        self.identity = IdentityClient(self)
        self.pool = PoolClient(self)

    # ---- Internal HTTP helpers ----

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _handle_response(self, resp: requests.Response) -> Any:
        if not resp.ok:
            try:
                body = resp.json()
                msg = body.get("error", resp.text)
            except Exception:
                msg = resp.text
            raise SwarmLLMError(resp.status_code, msg)
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.text

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        resp = self._session.get(self._url(path), params=params, timeout=self.timeout)
        return self._handle_response(resp)

    def _post(self, path: str, json: Optional[dict[str, Any]] = None) -> Any:
        resp = self._session.post(self._url(path), json=json, timeout=self.timeout)
        return self._handle_response(resp)

    def _put(self, path: str, json: Optional[dict[str, Any]] = None) -> Any:
        resp = self._session.put(self._url(path), json=json, timeout=self.timeout)
        return self._handle_response(resp)

    def _delete(self, path: str) -> Any:
        resp = self._session.delete(self._url(path), timeout=self.timeout)
        return self._handle_response(resp)

    # ---- OpenAI-compatible API ----

    def models(self) -> list[Model]:
        """GET /v1/models — List available models."""
        data = self._get("/v1/models")
        return [
            Model(
                id=m["id"],
                object=m.get("object", "model"),
                owned_by=m.get("owned_by", "swarmllm"),
            )
            for m in data.get("data", [])
        ]

    def chat_completion(
        self,
        messages: list[Union[dict[str, Any], ChatMessage]],
        model: Optional[str] = None,
        *,
        stream: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        session_id: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        stop: Optional[Union[str, list[str]]] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs: Any,
    ) -> Union[ChatResponse, Iterator[str]]:
        """POST /v1/chat/completions — Chat completion.

        Args:
            messages: List of chat messages (dicts or ChatMessage objects).
            model: Model name. If None, uses the first available model.
            stream: If True, returns an iterator of content chunks.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            session_id: KV-cache session ID for multi-turn conversations.
            tools: Tool definitions for function calling.
            stop: Stop sequence(s).
            frequency_penalty: Frequency penalty (0.0 to 2.0).
            presence_penalty: Presence penalty (0.0 to 2.0).
            **kwargs: Additional parameters passed to the API.

        Returns:
            ChatResponse for non-streaming, Iterator[str] for streaming.
        """
        if model is None:
            available = self.models()
            if not available:
                raise SwarmLLMError(404, "No models available on the node")
            model = available[0].id

        msg_dicts = [
            m.to_dict() if isinstance(m, ChatMessage) else m for m in messages
        ]
        body: dict[str, Any] = {
            "model": model,
            "messages": msg_dicts,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            **kwargs,
        }
        if session_id:
            body["session_id"] = session_id
        if tools:
            body["tools"] = tools
        if stop:
            body["stop"] = stop

        if stream:
            return self._stream_chat(body)

        resp = self._session.post(
            self._url("/v1/chat/completions"), json=body, timeout=self.timeout
        )
        data = self._handle_response(resp)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage_data = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""),
            content=message.get("content", ""),
            finish_reason=choice.get("finish_reason"),
            model=data.get("model", model),
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            session_id=data.get("session_id"),
            raw=data,
        )

    def _stream_chat(self, body: dict[str, Any]) -> Iterator[str]:
        """Internal: stream SSE chunks and yield content strings."""
        resp = self._session.post(
            self._url("/v1/chat/completions"),
            json=body,
            stream=True,
            timeout=self.timeout,
        )
        if not resp.ok:
            try:
                msg = resp.json().get("error", resp.text)
            except Exception:
                msg = resp.text
            raise SwarmLLMError(resp.status_code, msg)

        for line in resp.iter_lines(decode_unicode=True):
            if not line or line.startswith(":"):
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    return
                try:
                    chunk = json.loads(data_str)
                    content = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    def chat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        stream: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[ChatResponse, Iterator[str]]:
        """Convenience method: single-turn chat.

        Args:
            prompt: User message text.
            model: Model name (auto-detected if None).
            system: Optional system prompt.
            stream: Enable streaming.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            session_id: KV-cache session ID.

        Returns:
            ChatResponse for non-streaming, Iterator[str] for streaming.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_completion(
            messages,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            session_id=session_id,
            **kwargs,
        )

    def embeddings(
        self,
        input: Union[str, list[str]],
        model: Optional[str] = None,
        encoding_format: str = "float",
    ) -> EmbeddingResponse:
        """POST /v1/embeddings — Create embeddings.

        Args:
            input: Text string or list of strings to embed.
            model: Model name. If None, uses the first available model.
            encoding_format: Output format ("float" or "base64").

        Returns:
            EmbeddingResponse with embedding vectors.
        """
        if model is None:
            available = self.models()
            if not available:
                raise SwarmLLMError(404, "No models available on the node")
            model = available[0].id

        body: dict[str, Any] = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format,
        }
        data = self._post("/v1/embeddings", json=body)
        vectors = [item["embedding"] for item in data.get("data", [])]
        usage = data.get("usage", {})
        return EmbeddingResponse(
            data=vectors,
            model=data.get("model", model),
            usage_prompt_tokens=usage.get("prompt_tokens", 0),
            usage_total_tokens=usage.get("total_tokens", 0),
            raw=data,
        )

    def status(self) -> dict[str, Any]:
        """GET /v1/status — Node status (SwarmLLM extension)."""
        return self._get("/v1/status")

    def health(self) -> str:
        """GET /health — Health check."""
        return self._get("/health")

    def health_ready(self) -> dict[str, Any]:
        """GET /health/ready — Readiness probe with subsystem status."""
        return self._get("/health/ready")

    def metrics(self) -> str:
        """GET /metrics — Prometheus metrics."""
        return self._get("/metrics")

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> SwarmLLM:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
