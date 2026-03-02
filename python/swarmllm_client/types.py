"""Data types for the SwarmLLM client SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """Response from a chat completion request."""

    id: str = ""
    content: str = ""
    finish_reason: Optional[str] = None
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    session_id: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Response from an embedding request."""

    data: list[list[float]] = field(default_factory=list)
    model: str = ""
    usage_prompt_tokens: int = 0
    usage_total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    """A model available on the node."""

    id: str
    object: str = "model"
    owned_by: str = "swarmllm"


@dataclass
class NodeStats:
    """Node statistics from /api/admin/stats."""

    node_id: str = ""
    version: str = ""
    uptime_seconds: int = 0
    peers_connected: int = 0
    credits_balance: float = 0
    credit_tier: str = ""
    hosted_shards: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PeerInfo:
    """Information about a connected peer."""

    node_id: str = ""
    healthy: bool = False
    latency_ms: Optional[float] = None
    trust_score: Optional[float] = None
    gpu: Optional[str] = None
    hosted_models: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CreditInfo:
    """Credit balance and tier information."""

    balance: float = 0
    lifetime_earned: float = 0
    lifetime_spent: float = 0
    tier: str = ""
    last_updated: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardStorage:
    """Shard storage breakdown."""

    models: list[dict[str, Any]] = field(default_factory=list)
    total_local_bytes: int = 0
    disk_usage_bytes: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
