"""Tests for the asynchronous SwarmLLM client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from swarmllm_client import AsyncSwarmLLM, ChatMessage, ChatResponse
from swarmllm_client.client import SwarmLLMError
from swarmllm_client.types import (
    CreditInfo,
    Model,
    NodeStats,
    PeerInfo,
    ShardStorage,
)


# ---- Helpers ----


def _mock_json_response(data, status: int = 200):
    """Create a mock aiohttp response that works as an async context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.headers = {"content-type": "application/json"}
    resp.json = AsyncMock(return_value=data)
    resp.text = AsyncMock(return_value=json.dumps(data))
    return resp


def _mock_text_response(text: str, status: int = 200):
    resp = AsyncMock()
    resp.status = status
    resp.headers = {"content-type": "text/plain"}
    resp.text = AsyncMock(return_value=text)
    return resp


class _FakeContextManager:
    """Wraps a mock response to act as an async context manager for aiohttp."""

    def __init__(self, resp):
        self.resp = resp

    async def __aenter__(self):
        return self.resp

    async def __aexit__(self, *args):
        pass


def _make_session(resp):
    """Create a mock session with .closed = False and the right HTTP methods."""
    session = MagicMock()
    session.closed = False
    cm = _FakeContextManager(resp)
    session.get = MagicMock(return_value=cm)
    session.post = MagicMock(return_value=cm)
    session.put = MagicMock(return_value=cm)
    session.delete = MagicMock(return_value=cm)
    session.close = AsyncMock()
    return session


# ---- Client lifecycle ----


class TestAsyncClientInit:
    def test_default_url(self):
        c = AsyncSwarmLLM()
        assert c.base_url == "http://localhost:8800"

    def test_trailing_slash_stripped(self):
        c = AsyncSwarmLLM("http://host:9000/")
        assert c.base_url == "http://host:9000"

    def test_headers_with_api_key(self):
        c = AsyncSwarmLLM(api_key="sk-test")
        headers = c._headers()
        assert headers["Authorization"] == "Bearer sk-test"

    def test_headers_without_api_key(self):
        c = AsyncSwarmLLM()
        assert c._headers() == {}

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncSwarmLLM() as c:
            assert c.base_url == "http://localhost:8800"


# ---- Models ----


class TestAsyncModels:
    @pytest.mark.asyncio
    async def test_models(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({
            "data": [
                {"id": "model-a", "object": "model", "owned_by": "swarmllm"},
                {"id": "model-b"},
            ]
        })
        c._session = _make_session(resp)

        models = await c.models()
        assert len(models) == 2
        assert isinstance(models[0], Model)
        assert models[0].id == "model-a"


# ---- Chat completion ----


class TestAsyncChatCompletion:
    CHAT_DATA = {
        "id": "chatcmpl-456",
        "model": "model-a",
        "choices": [
            {
                "message": {"role": "assistant", "content": "World!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "session_id": "sess-x",
    }

    @pytest.mark.asyncio
    async def test_non_streaming(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response(self.CHAT_DATA)
        c._session = _make_session(resp)

        result = await c.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="model-a",
        )
        assert isinstance(result, ChatResponse)
        assert result.content == "World!"
        assert result.usage.total_tokens == 5
        assert result.session_id == "sess-x"

    @pytest.mark.asyncio
    async def test_chat_convenience(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response(self.CHAT_DATA)
        c._session = _make_session(resp)

        result = await c.chat("Hello", model="model-a", system="Be brief")
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_auto_model_selection(self):
        c = AsyncSwarmLLM()
        models_data = {"data": [{"id": "auto-m"}]}
        chat_data = self.CHAT_DATA

        models_resp = _mock_json_response(models_data)
        chat_resp = _mock_json_response(chat_data)

        session = MagicMock()
        session.closed = False
        session.get = MagicMock(return_value=_FakeContextManager(models_resp))
        session.post = MagicMock(return_value=_FakeContextManager(chat_resp))
        session.close = AsyncMock()
        c._session = session

        await c.chat_completion([{"role": "user", "content": "hi"}])
        assert session.get.called
        assert session.post.called


# ---- Error handling ----


class TestAsyncErrors:
    @pytest.mark.asyncio
    async def test_api_error(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({"error": "not found"}, status=404)
        c._session = _make_session(resp)

        with pytest.raises(SwarmLLMError) as exc_info:
            await c.models()
        assert exc_info.value.status_code == 404


# ---- Admin client ----


class TestAsyncAdminClient:
    @pytest.mark.asyncio
    async def test_stats(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({
            "node_id": "node-1",
            "version": "0.1.0",
            "uptime_seconds": 7200,
            "peers": 3,
            "credits": {"balance": 200},
            "tier": "Silver",
            "hosted_shards": 5,
        })
        c._session = _make_session(resp)

        stats = await c.admin.stats()
        assert isinstance(stats, NodeStats)
        assert stats.node_id == "node-1"
        assert stats.hosted_shards == 5

    @pytest.mark.asyncio
    async def test_peers(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response([
            {"node_id": "p1", "healthy": True, "latency_ms": 10.0}
        ])
        c._session = _make_session(resp)

        peers = await c.admin.peers()
        assert len(peers) == 1
        assert isinstance(peers[0], PeerInfo)
        assert peers[0].healthy is True

    @pytest.mark.asyncio
    async def test_credits(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({
            "balance": 300.0,
            "lifetime_earned": 600.0,
            "lifetime_spent": 300.0,
            "tier": "Silver",
            "last_updated": "2026-03-01T00:00:00Z",
        })
        c._session = _make_session(resp)

        credits = await c.admin.credits()
        assert isinstance(credits, CreditInfo)
        assert credits.balance == 300.0
        assert credits.lifetime_earned == 600.0

    @pytest.mark.asyncio
    async def test_shard_storage(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({
            "models": [{"model_id": "m1"}],
            "total_local_bytes": 1000000,
            "disk_usage_bytes": 2000000,
        })
        c._session = _make_session(resp)

        storage = await c.admin.shard_storage()
        assert isinstance(storage, ShardStorage)
        assert storage.total_local_bytes == 1000000

    @pytest.mark.asyncio
    async def test_download_shards(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({"status": "started"})
        c._session = _make_session(resp)

        result = await c.admin.hf_download_shards("repo/model", "file.gguf", [0, 1])
        assert result["status"] == "started"


# ---- Identity client ----


class TestAsyncIdentityClient:
    @pytest.mark.asyncio
    async def test_set_nickname(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({"nickname": "my-node"})
        c._session = _make_session(resp)

        result = await c.identity.set_nickname("my-node")
        assert result["nickname"] == "my-node"

    @pytest.mark.asyncio
    async def test_leaderboard(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response([{"node_id": "a", "credits": 50}])
        c._session = _make_session(resp)

        lb = await c.identity.leaderboard()
        assert len(lb) == 1


# ---- Pool client ----


class TestAsyncPoolClient:
    @pytest.mark.asyncio
    async def test_create(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({"pool_id": "p1"})
        c._session = _make_session(resp)

        result = await c.pool.create("test-pool")
        assert result["pool_id"] == "p1"

    @pytest.mark.asyncio
    async def test_leave(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({"status": "left"})
        c._session = _make_session(resp)

        result = await c.pool.leave()
        assert result["status"] == "left"


# ---- Health / Status ----


class TestAsyncHealthStatus:
    @pytest.mark.asyncio
    async def test_health(self):
        c = AsyncSwarmLLM()
        resp = _mock_text_response("OK")
        c._session = _make_session(resp)

        result = await c.health()
        assert result == "OK"

    @pytest.mark.asyncio
    async def test_status(self):
        c = AsyncSwarmLLM()
        resp = _mock_json_response({"status": "ready"})
        c._session = _make_session(resp)

        result = await c.status()
        assert result["status"] == "ready"
