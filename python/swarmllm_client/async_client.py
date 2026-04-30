"""Asynchronous SwarmLLM client using aiohttp."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional, Union

import aiohttp

from swarmllm_client.client import SwarmLLMError
from swarmllm_client.types import (
    ChatMessage,
    ChatResponse,
    CreditInfo,
    EmbeddingResponse,
    Model,
    NodeStats,
    PeerInfo,
    ShardStorage,
    Usage,
)


class AsyncAdminClient:
    """Async client for SwarmLLM admin endpoints."""

    def __init__(self, parent: AsyncSwarmLLM) -> None:
        self._p = parent

    async def stats(self) -> NodeStats:
        """GET /api/admin/stats — Node statistics."""
        data = await self._p._get("/api/admin/stats")
        return NodeStats(
            node_id=data.get("node_id", ""),
            version=data.get("version", ""),
            uptime_seconds=data.get("uptime_seconds", 0),
            peers_connected=data.get("peers", 0),
            credits_balance=data.get("credits", {}).get("balance", 0),
            credit_tier=data.get("tier", ""),
            hosted_shards=data.get("hosted_shards", 0),
            requests_served=data.get("requests_served", 0),
            forwards_served=data.get("forwards_served", 0),
            requests_made=data.get("requests_made", 0),
            active_requests=data.get("active_requests", 0),
            raw=data,
        )

    async def peers(self) -> list[PeerInfo]:
        """GET /api/admin/peers — Connected peers."""
        data = await self._p._get("/api/admin/peers")
        return [
            PeerInfo(
                node_id=p.get("node_id", ""),
                healthy=p.get("healthy", False),
                latency_ms=p.get("latency_ms"),
                trust_score=p.get("trust_score"),
                gpu=p.get("gpu"),
                hosted_models=p.get("hosted_models", []),
                raw=p,
            )
            for p in data
        ]

    async def credits(self) -> CreditInfo:
        """GET /api/admin/credits — Credit balance."""
        data = await self._p._get("/api/admin/credits")
        return CreditInfo(
            balance=data.get("balance", 0),
            lifetime_earned=data.get("lifetime_earned", 0),
            lifetime_spent=data.get("lifetime_spent", 0),
            tier=data.get("tier", ""),
            last_updated=data.get("last_updated", ""),
            raw=data,
        )

    async def api_key(self) -> str:
        """GET /api/admin/api-key — Retrieve the API key."""
        data = await self._p._get("/api/admin/api-key")
        return data.get("api_key", "")

    async def models(self) -> list[dict[str, Any]]:
        """GET /api/admin/models — Models with shard status."""
        return await self._p._get("/api/admin/models")

    async def model_status(self, model_id: str) -> dict[str, Any]:
        """GET /api/admin/models/:id/status — Model acquisition progress."""
        return await self._p._get(f"/api/admin/models/{model_id}/status")

    async def add_model(self, model_id: str) -> dict[str, Any]:
        """POST /api/admin/models/:id/add — Trigger model acquisition."""
        return await self._p._post(f"/api/admin/models/{model_id}/add")

    async def delete_model(self, model_id: str) -> dict[str, Any]:
        """DELETE /api/admin/models/:id — Remove model."""
        return await self._p._delete(f"/api/admin/models/{model_id}")

    async def delete_shard(self, model_id: str, shard_index: int) -> dict[str, Any]:
        """DELETE /api/admin/models/:id/shards/:index — Delete a shard."""
        return await self._p._delete(
            f"/api/admin/models/{model_id}/shards/{shard_index}"
        )

    async def lock_shard(
        self, model_id: str, shard_index: int, locked: bool = True
    ) -> dict[str, Any]:
        """PUT /api/admin/models/:id/shards/:index/lock — Lock/unlock shard."""
        return await self._p._put(
            f"/api/admin/models/{model_id}/shards/{shard_index}/lock",
            json={"locked": locked},
        )

    async def model_metadata(self, model_id: str) -> dict[str, Any]:
        """GET /api/admin/models/:id/metadata — GGUF metadata."""
        return await self._p._get(f"/api/admin/models/{model_id}/metadata")

    async def get_auto_manage(self, model_id: str) -> dict[str, Any]:
        """GET /api/admin/models/:id/auto-manage — Per-model auto-manage policy."""
        return await self._p._get(f"/api/admin/models/{model_id}/auto-manage")

    async def set_auto_manage(
        self, model_id: str, policy: dict[str, Any]
    ) -> dict[str, Any]:
        """PUT /api/admin/models/:id/auto-manage — Update auto-manage policy."""
        return await self._p._put(
            f"/api/admin/models/{model_id}/auto-manage", json=policy
        )

    async def shard_storage(self) -> ShardStorage:
        """GET /api/admin/shard-storage — Per-model storage breakdown."""
        data = await self._p._get("/api/admin/shard-storage")
        return ShardStorage(
            models=data.get("models", []),
            total_local_bytes=data.get("total_local_bytes", 0),
            disk_usage_bytes=data.get("disk_usage_bytes", 0),
            raw=data,
        )

    async def config(self) -> dict[str, Any]:
        """GET /api/admin/config — Read daemon configuration."""
        return await self._p._get("/api/admin/config")

    async def update_config(self, **kwargs: Any) -> dict[str, Any]:
        """PUT /api/admin/config — Update daemon configuration."""
        return await self._p._put("/api/admin/config", json=kwargs)

    async def reload_config(self) -> dict[str, Any]:
        """POST /api/admin/config/reload — Hot-reload operational parameters."""
        return await self._p._post("/api/admin/config/reload")

    async def hf_search(self, query: str) -> list[dict[str, Any]]:
        """GET /api/admin/hf/search — Search HuggingFace for GGUF models."""
        return await self._p._get("/api/admin/hf/search", params={"query": query})

    async def hf_probe(self, repo_id: str, filename: str) -> dict[str, Any]:
        """GET /api/admin/hf/probe — Probe a remote GGUF file."""
        return await self._p._get(
            "/api/admin/hf/probe",
            params={"repo_id": repo_id, "filename": filename},
        )

    async def hf_download_shards(
        self,
        repo_id: str,
        filename: str,
        shards: list[int],
        model_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /api/admin/hf/download-shards — Download specific shard ranges."""
        body: dict[str, Any] = {
            "repo_id": repo_id,
            "filename": filename,
            "shards": shards,
        }
        if model_id is not None:
            body["model_id"] = model_id
        return await self._p._post("/api/admin/hf/download-shards", json=body)

    async def hf_source(self, model_id: str) -> dict[str, Any]:
        """GET /api/admin/hf/source/:model_id — HF source info."""
        return await self._p._get(f"/api/admin/hf/source/{model_id}")

    async def downloads(self) -> list[dict[str, Any]]:
        """GET /api/admin/downloads — Active download queue."""
        return await self._p._get("/api/admin/downloads")

    async def cancel_download(self, model_id: str) -> dict[str, Any]:
        """POST /api/admin/downloads/:model_id/cancel — Cancel a download."""
        return await self._p._post(f"/api/admin/downloads/{model_id}/cancel")

    async def network_map(self) -> dict[str, Any]:
        """GET /api/admin/network-map — Network topology heatmap data."""
        return await self._p._get("/api/admin/network-map")

    async def network_code(self) -> dict[str, Any]:
        """GET /api/admin/network-code — Get shareable invite code."""
        return await self._p._get("/api/admin/network-code")

    async def join_network(self, code: str) -> dict[str, Any]:
        """POST /api/admin/join-network — Join via invite code."""
        return await self._p._post("/api/admin/join-network", json={"code": code})

    async def schedule(self) -> dict[str, Any]:
        """GET /api/admin/schedule — Resource schedule."""
        return await self._p._get("/api/admin/schedule")

    async def update_schedule(self, schedule: dict[str, Any]) -> dict[str, Any]:
        """PUT /api/admin/schedule — Update resource schedule."""
        return await self._p._put("/api/admin/schedule", json=schedule)

    async def prune_history(self) -> dict[str, Any]:
        """GET /api/admin/prune-history — Recent auto-prune events."""
        return await self._p._get("/api/admin/prune-history")

    async def shutdown(self) -> dict[str, Any]:
        """POST /api/admin/shutdown — Gracefully shut down the node."""
        return await self._p._post("/api/admin/shutdown")


class AsyncIdentityClient:
    """Async client for identity endpoints."""

    def __init__(self, parent: AsyncSwarmLLM) -> None:
        self._p = parent

    async def get_nickname(self) -> dict[str, Any]:
        """GET /api/identity/nickname"""
        return await self._p._get("/api/identity/nickname")

    async def set_nickname(self, nickname: str) -> dict[str, Any]:
        """PUT /api/identity/nickname"""
        return await self._p._put(
            "/api/identity/nickname", json={"nickname": nickname}
        )

    async def delete_nickname(self) -> dict[str, Any]:
        """DELETE /api/identity/nickname"""
        return await self._p._delete("/api/identity/nickname")

    async def leaderboard(self) -> list[dict[str, Any]]:
        """GET /api/identity/leaderboard"""
        return await self._p._get("/api/identity/leaderboard")

    async def peers(self) -> list[dict[str, Any]]:
        """GET /api/identity/peers"""
        return await self._p._get("/api/identity/peers")


class AsyncPoolClient:
    """Async client for device pool endpoints."""

    def __init__(self, parent: AsyncSwarmLLM) -> None:
        self._p = parent

    async def state(self) -> dict[str, Any]:
        """GET /api/pool/state"""
        return await self._p._get("/api/pool/state")

    async def create(self, name: str) -> dict[str, Any]:
        """POST /api/pool/create"""
        return await self._p._post("/api/pool/create", json={"name": name})

    async def invite(self, node_id: str) -> dict[str, Any]:
        """POST /api/pool/invite"""
        return await self._p._post("/api/pool/invite", json={"node_id": node_id})

    async def accept(self, invitation_id: str) -> dict[str, Any]:
        """POST /api/pool/accept"""
        return await self._p._post(
            "/api/pool/accept", json={"invitation_id": invitation_id}
        )

    async def remove(self, node_id: str) -> dict[str, Any]:
        """POST /api/pool/remove"""
        return await self._p._post("/api/pool/remove", json={"node_id": node_id})

    async def leave(self) -> dict[str, Any]:
        """POST /api/pool/leave"""
        return await self._p._post("/api/pool/leave")

    async def invitations(self) -> list[dict[str, Any]]:
        """GET /api/pool/invitations"""
        return await self._p._get("/api/pool/invitations")

    async def leaderboard(self) -> list[dict[str, Any]]:
        """GET /api/pool/leaderboard"""
        return await self._p._get("/api/pool/leaderboard")


class AsyncSwarmLLM:
    """Asynchronous SwarmLLM client.

    Provides access to the OpenAI-compatible API and SwarmLLM-specific
    admin/identity/pool endpoints using aiohttp.

    Args:
        base_url: SwarmLLM node URL (e.g. "http://localhost:8800").
        api_key: Bearer token for authenticated endpoints.
        timeout: Request timeout in seconds.

    Example::

        async with AsyncSwarmLLM("http://localhost:8800", api_key="sk-...") as client:
            response = await client.chat("Hello!", model="qwen2.5-coder-7b")
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
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

        self.admin = AsyncAdminClient(self)
        self.identity = AsyncIdentityClient(self)
        self.pool = AsyncPoolClient(self)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout, headers=self._headers()
            )
        return self._session

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def _handle_response(self, resp: aiohttp.ClientResponse) -> Any:
        if resp.status >= 400:
            text = await resp.text()
            try:
                body = json.loads(text)
                msg = body.get("error", text)
            except (json.JSONDecodeError, AttributeError):
                msg = text
            raise SwarmLLMError(resp.status, msg)
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return await resp.json()
        return await resp.text()

    async def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        session = await self._ensure_session()
        async with session.get(self._url(path), params=params) as resp:
            return await self._handle_response(resp)

    async def _post(self, path: str, json: Optional[dict[str, Any]] = None) -> Any:
        session = await self._ensure_session()
        async with session.post(self._url(path), json=json) as resp:
            return await self._handle_response(resp)

    async def _put(self, path: str, json: Optional[dict[str, Any]] = None) -> Any:
        session = await self._ensure_session()
        async with session.put(self._url(path), json=json) as resp:
            return await self._handle_response(resp)

    async def _delete(self, path: str) -> Any:
        session = await self._ensure_session()
        async with session.delete(self._url(path)) as resp:
            return await self._handle_response(resp)

    # ---- OpenAI-compatible API ----

    async def models(self) -> list[Model]:
        """GET /v1/models — List available models."""
        data = await self._get("/v1/models")
        return [
            Model(
                id=m["id"],
                object=m.get("object", "model"),
                owned_by=m.get("owned_by", "swarmllm"),
            )
            for m in data.get("data", [])
        ]

    async def chat_completion(
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
    ) -> Union[ChatResponse, AsyncIterator[str]]:
        """POST /v1/chat/completions — Chat completion (async).

        Args:
            messages: List of chat messages.
            model: Model name. If None, uses the first available model.
            stream: If True, returns an async iterator of content chunks.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            session_id: KV-cache session ID for multi-turn conversations.
            tools: Tool definitions for function calling.
            stop: Stop sequence(s).
            frequency_penalty: Frequency penalty.
            presence_penalty: Presence penalty.

        Returns:
            ChatResponse for non-streaming, AsyncIterator[str] for streaming.
        """
        if model is None:
            available = await self.models()
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

        data = await self._post("/v1/chat/completions", json=body)
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

    async def _stream_chat(self, body: dict[str, Any]) -> AsyncIterator[str]:
        """Internal: stream SSE chunks and yield content strings."""
        session = await self._ensure_session()
        async with session.post(
            self._url("/v1/chat/completions"), json=body
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise SwarmLLMError(resp.status, text)
            async for line_bytes in resp.content:
                line = line_bytes.decode("utf-8").strip()
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

    async def chat(
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
    ) -> Union[ChatResponse, AsyncIterator[str]]:
        """Convenience method: single-turn async chat.

        Args:
            prompt: User message text.
            model: Model name (auto-detected if None).
            system: Optional system prompt.
            stream: Enable streaming.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            session_id: KV-cache session ID.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat_completion(
            messages,
            model=model,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            session_id=session_id,
            **kwargs,
        )

    async def embeddings(
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
        """
        if model is None:
            available = await self.models()
            if not available:
                raise SwarmLLMError(404, "No models available on the node")
            model = available[0].id

        body: dict[str, Any] = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format,
        }
        data = await self._post("/v1/embeddings", json=body)
        vectors = [item["embedding"] for item in data.get("data", [])]
        usage = data.get("usage", {})
        return EmbeddingResponse(
            data=vectors,
            model=data.get("model", model),
            usage_prompt_tokens=usage.get("prompt_tokens", 0),
            usage_total_tokens=usage.get("total_tokens", 0),
            raw=data,
        )

    async def status(self) -> dict[str, Any]:
        """GET /v1/status — Node status."""
        return await self._get("/v1/status")

    async def health(self) -> str:
        """GET /health — Health check."""
        return await self._get("/health")

    async def health_ready(self) -> dict[str, Any]:
        """GET /health/ready — Readiness probe."""
        return await self._get("/health/ready")

    async def metrics(self) -> str:
        """GET /metrics — Prometheus metrics."""
        return await self._get("/metrics")

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> AsyncSwarmLLM:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
