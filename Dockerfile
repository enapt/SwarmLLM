# ============================================================================
# SwarmLLM — CPU-only Docker Image
# ============================================================================
# Build:  docker build -t swarmllm/swarmllm .
# Run:    docker run -p 8800:8800 swarmllm/swarmllm
# Data:   docker run -p 8800:8800 -v swarmllm-data:/data swarmllm/swarmllm
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM rust:1.75-bookworm AS builder

# Install system dependencies required for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    libclang-dev \
    capnproto \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache dependency compilation: copy manifests first, build a dummy to populate
# the cargo registry and compile deps, then replace with real source.
COPY Cargo.toml Cargo.lock build.rs ./
COPY proto/ proto/
RUN mkdir -p src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs && \
    echo '' > src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Copy full source and build the real binary
COPY src/ src/
COPY frontend/ frontend/
COPY config/ config/
RUN cargo build --release && \
    strip target/release/swarmllm

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash swarmllm

# Copy binary from builder
COPY --from=builder /build/target/release/swarmllm /usr/local/bin/swarmllm

# Create data directory and set ownership
RUN mkdir -p /data && chown swarmllm:swarmllm /data

# Environment configuration
ENV SWARMLLM_NODE_DATA_DIR=/data
ENV SWARMLLM_NODE_LISTEN_PORT=8800
ENV SWARMLLM_LOGGING_LEVEL=info

# Expose port 8800 for both HTTP (TCP) and P2P/QUIC (UDP)
EXPOSE 8800/tcp 8800/udp

# Persistent data volume
VOLUME /data

# Health check against the readiness probe (no auth required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8800/health/ready || exit 1

USER swarmllm
WORKDIR /data

ENTRYPOINT ["swarmllm"]
CMD ["run", "--data-dir", "/data"]
