"""Tests for the synchronous SwarmLLM client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from swarmllm_client import SwarmLLM, ChatMessage, ChatResponse, Usage
from swarmllm_client.client import SwarmLLMError
from swarmllm_client.types import (
    CreditInfo,
    EmbeddingResponse,
    Model,
    NodeStats,
    PeerInfo,
    ShardStorage,
)


# ---- Helpers ----


def _mock_response(
    json_data: dict | list | None = None,
    text: str = "",
    status_code: int = 200,
    ok: bool = True,
    content_type: str = "application/json",
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = ok
    resp.text = text or json.dumps(json_data) if json_data else text
    resp.headers = {"content-type": content_type}
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


# ---- Client construction ----


class TestClientInit:
    def test_default_url(self):
        c = SwarmLLM()
        assert c.base_url == "http://localhost:8800"

    def test_trailing_slash_stripped(self):
        c = SwarmLLM("http://host:9000/")
        assert c.base_url == "http://host:9000"

    def test_api_key_header(self):
        c = SwarmLLM(api_key="sk-test")
        assert c._session.headers["Authorization"] == "Bearer sk-test"

    def test_context_manager(self):
        with SwarmLLM() as c:
            assert c.base_url == "http://localhost:8800"


# ---- Models ----


class TestModels:
    def test_models(self):
        c = SwarmLLM()
        mock_resp = _mock_response(
            json_data={
                "data": [
                    {"id": "qwen2.5-coder-7b", "object": "model", "owned_by": "swarmllm"},
                    {"id": "tinyllama-1.1b", "object": "model"},
                ]
            }
        )
        with patch.object(c._session, "get", return_value=mock_resp):
            models = c.models()
        assert len(models) == 2
        assert isinstance(models[0], Model)
        assert models[0].id == "qwen2.5-coder-7b"
        assert models[1].owned_by == "swarmllm"


# ---- Chat completion ----


class TestChatCompletion:
    CHAT_RESPONSE = {
        "id": "chatcmpl-123",
        "model": "qwen2.5-coder-7b",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "session_id": "sess-abc",
    }

    def test_non_streaming(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data=self.CHAT_RESPONSE)
        with patch.object(c._session, "post", return_value=mock_resp):
            result = c.chat_completion(
                [{"role": "user", "content": "Hi"}],
                model="qwen2.5-coder-7b",
            )
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello!"
        assert result.finish_reason == "stop"
        assert result.model == "qwen2.5-coder-7b"
        assert result.usage.total_tokens == 8
        assert result.session_id == "sess-abc"

    def test_auto_model_selection(self):
        c = SwarmLLM()
        models_resp = _mock_response(
            json_data={"data": [{"id": "auto-model", "object": "model"}]}
        )
        chat_resp = _mock_response(json_data=self.CHAT_RESPONSE)

        with patch.object(c._session, "get", return_value=models_resp), patch.object(
            c._session, "post", return_value=chat_resp
        ) as mock_post:
            c.chat_completion([{"role": "user", "content": "Hi"}])
        body = mock_post.call_args[1]["json"]
        assert body["model"] == "auto-model"

    def test_chat_message_objects(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data=self.CHAT_RESPONSE)
        msgs = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
        ]
        with patch.object(c._session, "post", return_value=mock_resp) as mock_post:
            c.chat_completion(msgs, model="m")
        body = mock_post.call_args[1]["json"]
        assert body["messages"][0] == {"role": "system", "content": "You are helpful."}

    def test_chat_convenience(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data=self.CHAT_RESPONSE)
        with patch.object(c._session, "post", return_value=mock_resp) as mock_post:
            result = c.chat("Hi", model="m", system="Be brief")
        body = mock_post.call_args[1]["json"]
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        assert isinstance(result, ChatResponse)

    def test_streaming(self):
        c = SwarmLLM()
        lines = [
            'data: {"choices":[{"delta":{"content":"He"}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(lines)

        with patch.object(c._session, "post", return_value=mock_resp):
            chunks = list(c.chat_completion(
                [{"role": "user", "content": "Hi"}],
                model="m",
                stream=True,
            ))
        assert chunks == ["He", "llo"]

    def test_tools_and_stop(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data=self.CHAT_RESPONSE)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        with patch.object(c._session, "post", return_value=mock_resp) as mock_post:
            c.chat_completion(
                [{"role": "user", "content": "weather?"}],
                model="m",
                tools=tools,
                stop=["END"],
            )
        body = mock_post.call_args[1]["json"]
        assert body["tools"] == tools
        assert body["stop"] == ["END"]


# ---- Embeddings ----


class TestEmbeddings:
    def test_embeddings(self):
        c = SwarmLLM()
        mock_resp = _mock_response(
            json_data={
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "model": "emb-model",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
        )
        with patch.object(c._session, "post", return_value=mock_resp):
            result = c.embeddings("Hello", model="emb-model")
        assert isinstance(result, EmbeddingResponse)
        assert result.data == [[0.1, 0.2, 0.3]]
        assert result.usage_prompt_tokens == 4


# ---- Error handling ----


class TestErrors:
    def test_api_error(self):
        c = SwarmLLM()
        mock_resp = _mock_response(
            json_data={"error": "model not found"},
            status_code=404,
            ok=False,
        )
        with patch.object(c._session, "get", return_value=mock_resp):
            with pytest.raises(SwarmLLMError) as exc_info:
                c.models()
        assert exc_info.value.status_code == 404
        assert "model not found" in exc_info.value.message

    def test_no_models_error(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data={"data": []})
        with patch.object(c._session, "get", return_value=mock_resp):
            with pytest.raises(SwarmLLMError) as exc_info:
                c.chat_completion([{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 404


# ---- Admin client ----


class TestAdminClient:
    def test_stats(self):
        c = SwarmLLM()
        mock_resp = _mock_response(
            json_data={
                "node_id": "abc123",
                "version": "0.1.0",
                "uptime_seconds": 3600,
                "peers": 5,
                "credits": {"balance": 100.5},
                "tier": "Gold",
                "hosted_shards": 3,
            }
        )
        with patch.object(c._session, "get", return_value=mock_resp):
            stats = c.admin.stats()
        assert isinstance(stats, NodeStats)
        assert stats.node_id == "abc123"
        assert stats.credits_balance == 100.5
        assert stats.hosted_shards == 3

    def test_peers(self):
        c = SwarmLLM()
        mock_resp = _mock_response(
            json_data=[
                {
                    "node_id": "peer1",
                    "healthy": True,
                    "latency_ms": 42.0,
                    "trust_score": 0.95,
                    "gpu": "RTX 3090",
                    "hosted_models": ["model-a"],
                }
            ]
        )
        with patch.object(c._session, "get", return_value=mock_resp):
            peers = c.admin.peers()
        assert len(peers) == 1
        assert isinstance(peers[0], PeerInfo)
        assert peers[0].healthy is True
        assert peers[0].gpu == "RTX 3090"

    def test_credits(self):
        c = SwarmLLM()
        mock_resp = _mock_response(
            json_data={
                "balance": 500.0,
                "lifetime_earned": 1000.0,
                "lifetime_spent": 500.0,
                "tier": "Gold",
                "last_updated": "2026-03-01T12:00:00Z",
            }
        )
        with patch.object(c._session, "get", return_value=mock_resp):
            credits = c.admin.credits()
        assert isinstance(credits, CreditInfo)
        assert credits.balance == 500.0
        assert credits.lifetime_earned == 1000.0
        assert credits.tier == "Gold"

    def test_shard_storage(self):
        c = SwarmLLM()
        mock_resp = _mock_response(
            json_data={
                "models": [{"model_id": "m1", "shards": 4}],
                "total_local_bytes": 5000000000,
                "disk_usage_bytes": 6000000000,
            }
        )
        with patch.object(c._session, "get", return_value=mock_resp):
            storage = c.admin.shard_storage()
        assert isinstance(storage, ShardStorage)
        assert storage.total_local_bytes == 5000000000
        assert len(storage.models) == 1

    def test_hf_download_shards(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data={"status": "started"})
        with patch.object(c._session, "post", return_value=mock_resp) as mock_post:
            c.admin.hf_download_shards("TheBloke/TinyLlama", "tiny.gguf", [0, 1])
        body = mock_post.call_args[1]["json"]
        assert body["repo_id"] == "TheBloke/TinyLlama"
        assert body["shards"] == [0, 1]

    def test_lock_shard(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data={"locked": True})
        with patch.object(c._session, "put", return_value=mock_resp) as mock_put:
            c.admin.lock_shard("model-a", 2, locked=True)
        assert "/models/model-a/shards/2/lock" in mock_put.call_args[0][0]


# ---- Health / Status ----


class TestHealthStatus:
    def test_health(self):
        c = SwarmLLM()
        mock_resp = _mock_response(text="OK", content_type="text/plain", json_data=None)
        mock_resp.text = "OK"
        with patch.object(c._session, "get", return_value=mock_resp):
            assert c.health() == "OK"

    def test_status(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data={"status": "ready", "models": 2})
        with patch.object(c._session, "get", return_value=mock_resp):
            result = c.status()
        assert result["status"] == "ready"


# ---- Identity client ----


class TestIdentityClient:
    def test_set_nickname(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data={"nickname": "my-node"})
        with patch.object(c._session, "put", return_value=mock_resp) as mock_put:
            c.identity.set_nickname("my-node")
        body = mock_put.call_args[1]["json"]
        assert body["nickname"] == "my-node"

    def test_leaderboard(self):
        c = SwarmLLM()
        mock_resp = _mock_response(
            json_data=[{"node_id": "a", "credits": 100}]
        )
        with patch.object(c._session, "get", return_value=mock_resp):
            lb = c.identity.leaderboard()
        assert len(lb) == 1


# ---- Pool client ----


class TestPoolClient:
    def test_create_pool(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data={"pool_id": "p1"})
        with patch.object(c._session, "post", return_value=mock_resp) as mock_post:
            c.pool.create("my-pool")
        body = mock_post.call_args[1]["json"]
        assert body["name"] == "my-pool"

    def test_invite(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data={"invitation_id": "inv-1"})
        with patch.object(c._session, "post", return_value=mock_resp) as mock_post:
            c.pool.invite("node-abc")
        body = mock_post.call_args[1]["json"]
        assert body["node_id"] == "node-abc"

    def test_leave(self):
        c = SwarmLLM()
        mock_resp = _mock_response(json_data={"status": "left"})
        with patch.object(c._session, "post", return_value=mock_resp):
            result = c.pool.leave()
        assert result["status"] == "left"


# ---- Types ----


class TestTypes:
    def test_chat_message_to_dict(self):
        msg = ChatMessage(role="user", content="Hi")
        assert msg.to_dict() == {"role": "user", "content": "Hi"}

    def test_chat_message_with_tool_calls(self):
        msg = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "t1", "function": {"name": "f", "arguments": "{}"}}],
        )
        d = msg.to_dict()
        assert "tool_calls" in d
        assert d["tool_calls"][0]["id"] == "t1"

    def test_chat_message_tool_response(self):
        msg = ChatMessage(role="tool", content="result", tool_call_id="t1")
        d = msg.to_dict()
        assert d["tool_call_id"] == "t1"

    def test_usage_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.total_tokens == 0
