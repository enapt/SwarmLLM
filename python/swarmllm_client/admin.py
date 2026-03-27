"""Admin, Identity, and Pool API clients for SwarmLLM."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from swarmllm_client.types import (
    CreditInfo,
    NodeStats,
    PeerInfo,
    ShardStorage,
)

if TYPE_CHECKING:
    from swarmllm_client.client import SwarmLLM


class AdminClient:
    """Client for SwarmLLM admin endpoints (/api/admin/*)."""

    def __init__(self, parent: SwarmLLM) -> None:
        self._p = parent

    # ---- Node info ----

    def stats(self) -> NodeStats:
        """GET /api/admin/stats — Node statistics and hardware info."""
        data = self._p._get("/api/admin/stats")
        return NodeStats(
            node_id=data.get("node_id", ""),
            version=data.get("version", ""),
            uptime_seconds=data.get("uptime_seconds", 0),
            peers_connected=data.get("peers", 0),
            credits_balance=data.get("credits", {}).get("balance", 0),
            credit_tier=data.get("tier", ""),
            hosted_shards=data.get("hosted_shards", 0),
            raw=data,
        )

    def peers(self) -> list[PeerInfo]:
        """GET /api/admin/peers — Connected peers."""
        data = self._p._get("/api/admin/peers")
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

    def credits(self) -> CreditInfo:
        """GET /api/admin/credits — Credit balance and tier."""
        data = self._p._get("/api/admin/credits")
        return CreditInfo(
            balance=data.get("balance", 0),
            lifetime_earned=data.get("lifetime_earned", 0),
            lifetime_spent=data.get("lifetime_spent", 0),
            tier=data.get("tier", ""),
            last_updated=data.get("last_updated", ""),
            raw=data,
        )

    def api_key(self) -> str:
        """GET /api/admin/api-key — Retrieve the current API key."""
        data = self._p._get("/api/admin/api-key")
        return data.get("api_key", "")

    # ---- Models & Shards ----

    def models(self) -> list[dict[str, Any]]:
        """GET /api/admin/models — Models with shard status."""
        return self._p._get("/api/admin/models")

    def model_status(self, model_id: str) -> dict[str, Any]:
        """GET /api/admin/models/:id/status — Model acquisition progress."""
        return self._p._get(f"/api/admin/models/{model_id}/status")

    def add_model(self, model_id: str) -> dict[str, Any]:
        """POST /api/admin/models/:id/add — Trigger model acquisition."""
        return self._p._post(f"/api/admin/models/{model_id}/add")

    def delete_model(self, model_id: str) -> dict[str, Any]:
        """DELETE /api/admin/models/:id — Remove model and all its shards."""
        return self._p._delete(f"/api/admin/models/{model_id}")

    def delete_shard(self, model_id: str, shard_index: int) -> dict[str, Any]:
        """DELETE /api/admin/models/:id/shards/:index — Delete a single shard."""
        return self._p._delete(f"/api/admin/models/{model_id}/shards/{shard_index}")

    def lock_shard(self, model_id: str, shard_index: int, locked: bool = True) -> dict[str, Any]:
        """PUT /api/admin/models/:id/shards/:index/lock — Lock/unlock shard from pruning."""
        return self._p._put(
            f"/api/admin/models/{model_id}/shards/{shard_index}/lock",
            json={"locked": locked},
        )

    def model_metadata(self, model_id: str) -> dict[str, Any]:
        """GET /api/admin/models/:id/metadata — GGUF metadata browser."""
        return self._p._get(f"/api/admin/models/{model_id}/metadata")

    def get_auto_manage(self, model_id: str) -> dict[str, Any]:
        """GET /api/admin/models/:id/auto-manage — Per-model auto-manage policy."""
        return self._p._get(f"/api/admin/models/{model_id}/auto-manage")

    def set_auto_manage(self, model_id: str, policy: dict[str, Any]) -> dict[str, Any]:
        """PUT /api/admin/models/:id/auto-manage — Update per-model auto-manage policy."""
        return self._p._put(f"/api/admin/models/{model_id}/auto-manage", json=policy)

    def shard_storage(self) -> ShardStorage:
        """GET /api/admin/shard-storage — Per-model storage breakdown."""
        data = self._p._get("/api/admin/shard-storage")
        return ShardStorage(
            models=data.get("models", []),
            total_local_bytes=data.get("total_local_bytes", 0),
            disk_usage_bytes=data.get("disk_usage_bytes", 0),
            raw=data,
        )

    # ---- Config ----

    def config(self) -> dict[str, Any]:
        """GET /api/admin/config — Read daemon configuration."""
        return self._p._get("/api/admin/config")

    def update_config(self, **kwargs: Any) -> dict[str, Any]:
        """PUT /api/admin/config — Update daemon configuration.

        Accepts keyword arguments matching config fields:
        contribution, max_concurrent_requests, max_bandwidth_mbps,
        max_disk_mb, auto_manage_shards, shard_size_mb, etc.
        """
        return self._p._put("/api/admin/config", json=kwargs)

    def reload_config(self) -> dict[str, Any]:
        """POST /api/admin/config/reload — Hot-reload operational parameters."""
        return self._p._post("/api/admin/config/reload")

    # ---- HuggingFace ----

    def hf_search(self, query: str) -> list[dict[str, Any]]:
        """GET /api/admin/hf/search — Search HuggingFace for GGUF models."""
        return self._p._get("/api/admin/hf/search", params={"query": query})

    def hf_probe(self, repo_id: str, filename: str) -> dict[str, Any]:
        """GET /api/admin/hf/probe — Probe a remote GGUF file for shard info."""
        return self._p._get(
            "/api/admin/hf/probe",
            params={"repo_id": repo_id, "filename": filename},
        )

    def hf_download_shards(
        self,
        repo_id: str,
        filename: str,
        shards: list[int],
        model_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /api/admin/hf/download-shards — Download specific shard ranges.

        Args:
            repo_id: HuggingFace repository (e.g. "TheBloke/TinyLlama-1.1B-GGUF").
            filename: GGUF filename (e.g. "tinyllama-1.1b.Q4_K_M.gguf").
            shards: Shard indices to download (e.g. [0, 1, 2]).
            model_id: Optional target model ID to merge shards into.
        """
        body: dict[str, Any] = {
            "repo_id": repo_id,
            "filename": filename,
            "shards": shards,
        }
        if model_id is not None:
            body["model_id"] = model_id
        return self._p._post("/api/admin/hf/download-shards", json=body)

    def hf_source(self, model_id: str) -> dict[str, Any]:
        """GET /api/admin/hf/source/:model_id — HF source info for a model."""
        return self._p._get(f"/api/admin/hf/source/{model_id}")

    # ---- Downloads ----

    def downloads(self) -> list[dict[str, Any]]:
        """GET /api/admin/downloads — Active download queue."""
        return self._p._get("/api/admin/downloads")

    def cancel_download(self, model_id: str) -> dict[str, Any]:
        """POST /api/admin/downloads/:model_id/cancel — Cancel a download."""
        return self._p._post(f"/api/admin/downloads/{model_id}/cancel")

    # ---- Network ----

    def network_map(self) -> dict[str, Any]:
        """GET /api/admin/network-map — Network topology heatmap data."""
        return self._p._get("/api/admin/network-map")

    def network_code(self) -> dict[str, Any]:
        """GET /api/admin/network-code — Get shareable invite code."""
        return self._p._get("/api/admin/network-code")

    def join_network(self, code: str) -> dict[str, Any]:
        """POST /api/admin/join-network — Join network via invite code or multiaddr."""
        return self._p._post("/api/admin/join-network", json={"code": code})

    # ---- Schedule & Pruning ----

    def schedule(self) -> dict[str, Any]:
        """GET /api/admin/schedule — Resource schedule."""
        return self._p._get("/api/admin/schedule")

    def update_schedule(self, schedule: dict[str, Any]) -> dict[str, Any]:
        """PUT /api/admin/schedule — Update resource schedule."""
        return self._p._put("/api/admin/schedule", json=schedule)

    def prune_history(self) -> dict[str, Any]:
        """GET /api/admin/prune-history — Recent auto-prune events."""
        return self._p._get("/api/admin/prune-history")

    # ---- Lifecycle ----

    def shutdown(self) -> dict[str, Any]:
        """POST /api/admin/shutdown — Gracefully shut down the node (localhost only)."""
        return self._p._post("/api/admin/shutdown")


class IdentityClient:
    """Client for SwarmLLM identity endpoints (/api/identity/*)."""

    def __init__(self, parent: SwarmLLM) -> None:
        self._p = parent

    def get_nickname(self) -> dict[str, Any]:
        """GET /api/identity/nickname — Get this node's nickname."""
        return self._p._get("/api/identity/nickname")

    def set_nickname(self, nickname: str) -> dict[str, Any]:
        """PUT /api/identity/nickname — Set this node's nickname."""
        return self._p._put("/api/identity/nickname", json={"nickname": nickname})

    def delete_nickname(self) -> dict[str, Any]:
        """DELETE /api/identity/nickname — Remove nickname."""
        return self._p._delete("/api/identity/nickname")

    def leaderboard(self) -> list[dict[str, Any]]:
        """GET /api/identity/leaderboard — Network credit leaderboard."""
        return self._p._get("/api/identity/leaderboard")

    def peers(self) -> list[dict[str, Any]]:
        """GET /api/identity/peers — Peer identity directory."""
        return self._p._get("/api/identity/peers")


class PoolClient:
    """Client for SwarmLLM device pool endpoints (/api/pool/*)."""

    def __init__(self, parent: SwarmLLM) -> None:
        self._p = parent

    def state(self) -> dict[str, Any]:
        """GET /api/pool/state — Current pool membership."""
        return self._p._get("/api/pool/state")

    def create(self, name: str) -> dict[str, Any]:
        """POST /api/pool/create — Create a new device pool."""
        return self._p._post("/api/pool/create", json={"name": name})

    def invite(self, node_id: str) -> dict[str, Any]:
        """POST /api/pool/invite — Invite a node to the pool."""
        return self._p._post("/api/pool/invite", json={"node_id": node_id})

    def accept(self, invitation_id: str) -> dict[str, Any]:
        """POST /api/pool/accept — Accept a pool invitation."""
        return self._p._post("/api/pool/accept", json={"invitation_id": invitation_id})

    def remove(self, node_id: str) -> dict[str, Any]:
        """POST /api/pool/remove — Remove a member from the pool."""
        return self._p._post("/api/pool/remove", json={"node_id": node_id})

    def leave(self) -> dict[str, Any]:
        """POST /api/pool/leave — Leave the current pool."""
        return self._p._post("/api/pool/leave")

    def invitations(self) -> list[dict[str, Any]]:
        """GET /api/pool/invitations — Pending pool invitations."""
        return self._p._get("/api/pool/invitations")

    def leaderboard(self) -> list[dict[str, Any]]:
        """GET /api/pool/leaderboard — Member contribution rankings."""
        return self._p._get("/api/pool/leaderboard")
