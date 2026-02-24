@0xc2f00d2a3b8e4f1a;

# Cap'n Proto schema for SwarmLLM high-bandwidth network messages.
#
# Only tensor data (activation forwarding) and shard chunk transfers use
# Cap'n Proto for zero-copy deserialization. All other protocol messages
# (health, governance, credits, discovery) remain serde_json over GossipSub.

struct TensorEnvelope {
  requestId   @0 :Data;      # UUID as 16 raw bytes
  sequenceNum @1 :UInt32;    # Token position in generation sequence
  format      @2 :TensorFmt;
  data        @3 :Data;      # Raw tensor bytes (zero-copy on read)

  enum TensorFmt {
    fp16 @0;
    fp32 @1;
    int8 @2;
  }
}

struct ShardChunk {
  modelId   @0 :Text;
  shardIdx  @1 :UInt32;
  offset    @2 :UInt64;
  data      @3 :Data;       # 1MB chunk of shard weight data
  totalSize @4 :UInt64;
  chunkHash @5 :Data;       # BLAKE3 hash of this chunk (32 bytes)
}

struct TensorRequest {
  union {
    forward  @0 :TensorEnvelope;
    shardReq @1 :ShardChunkRequest;
  }
}

struct TensorResponse {
  union {
    result   @0 :LayerResultMsg;
    chunk    @1 :ShardChunk;
    ack      @2 :Void;
  }
}

struct LayerResultMsg {
  requestId    @0 :Data;     # UUID as 16 raw bytes
  tokenIds     @1 :List(UInt32);
  finishReason @2 :FinishReason;

  enum FinishReason {
    none      @0;
    stop      @1;
    maxTokens @2;
    error     @3;
  }
}

struct ShardChunkRequest {
  modelId  @0 :Text;
  shardIdx @1 :UInt32;
  offset   @2 :UInt64;
  length   @3 :UInt32;
}
