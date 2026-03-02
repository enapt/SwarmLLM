# Shard-Only Mode

SwarmLLM supports **shard-only operation** — a node only needs individual shard files (~512 MB each) plus a small GGUF header (~6 MB), not the full model file.

## How It Works

A model directory in shard-only mode:

```
~/.swarmllm/models/qwen2.5-coder-7b/
├── manifest.json        # Model metadata + shard layout
├── gguf_header.bin      # First ~6MB of GGUF (metadata + tensor index)
├── shard_000.bin        # 512MB shard
├── shard_001.bin
├── shard_002.bin
└── ...
```

SwarmLLM automatically extracts `gguf_header.bin` from `shard_000.bin` when first needed. The `ShardReader` constructs a virtual GGUF from header + shard files, so the model parser works exactly as if the full GGUF were present.

## Why This Matters

- A 7B model is ~4.5 GB as a full GGUF, but a single shard is only ~512 MB
- Nodes only load the layers they're assigned — no wasted disk or VRAM
- You can participate in inference for a 70B model on a machine with 8 GB VRAM by hosting just a few shards

## Manual Shard Assignment (--shards)

For multi-node split inference, assign each node a subset of shards:

```bash
./swarmllm run --shards "0-3"    # This node handles shards 0, 1, 2, 3
```

The range is persisted to the database and restored on subsequent runs. Start without `--shards` to clear.

**Behavior when `--shards` is set:**
- The node only advertises the specified shard indices
- Auto-manage prioritizes downloading missing shards in the range (100x scoring bonus)
- Smart pruning never removes shards in the configured range

## Multi-Node Example

Run a 7B model across two machines:

```bash
# Machine A (shards 0-3, layers 0-13):
./swarmllm run --shards "0-3" --bootstrap "/ip4/MACHINE_B_IP/udp/8800/quic-v1/p2p/PEER_ID"

# Machine B (shards 4-7, layers 14-27):
./swarmllm run --shards "4-7" --bootstrap "/ip4/MACHINE_A_IP/udp/8800/quic-v1/p2p/PEER_ID"
```

Both nodes discover each other, assemble a distributed pipeline, and forward hidden-state activations between them. The pipeline is assembled automatically by the InferenceRouter.

## Without --shards

If you don't specify `--shards`, the node auto-detects and advertises **all** local shards. This is the normal mode for most users — `--shards` is only needed when you want explicit control over which layers a node handles.
