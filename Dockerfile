# ============================================================================
# SwarmLLM — Multi-stage Docker Image
# ============================================================================
# CPU:   docker build -t swarmllm .
# CUDA:  docker build --build-arg FEATURES="candle-cuda" -t swarmllm:cuda .
# Run:   docker run -p 8800:8800 -p 8810:8810 swarmllm
# Data:  docker run -p 8800:8800 -p 8810:8810 -v swarmllm-data:/data swarmllm
# GPU:   docker run --gpus all -p 8800:8800 -p 8810:8810 swarmllm:cuda
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM rust:1.80-bookworm AS builder

ARG FEATURES=""

# Install system dependencies required for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache dependency compilation: copy manifests first, build a dummy to populate
# the cargo registry and compile deps, then replace with real source.
COPY Cargo.toml Cargo.lock build.rs ./
COPY vendor/ vendor/
RUN mkdir -p src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs && \
    echo '' > src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Copy full source and build the real binary
COPY src/ src/
COPY frontend/ frontend/
COPY config/ config/
RUN if [ -n "$FEATURES" ]; then \
        cargo build --release --features "$FEATURES"; \
    else \
        cargo build --release; \
    fi

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

# Copy binary from builder (already stripped via profile.release.strip = true)
COPY --from=builder /build/target/release/swarmllm /usr/local/bin/swarmllm
COPY config/default.toml /etc/swarmllm/default.toml

# Create data directory and set ownership
RUN mkdir -p /data && chown swarmllm:swarmllm /data

# Environment configuration
ENV SWARMLLM_NODE_DATA_DIR=/data
ENV SWARMLLM_NODE_LISTEN_PORT=8800
ENV SWARMLLM_LOGGING_LEVEL=info
ENV SWARMLLM_UI_OPEN_BROWSER_ON_START=false

# Expose ports:
#   8800/tcp  — HTTP API + admin dashboard
#   8810/tcp  — P2P (Noise+Yamux, primary transport)
#   8800/udp  — P2P (QUIC, optional)
EXPOSE 8800/tcp 8810/tcp 8800/udp

# Persistent data volume
VOLUME /data

# Health check against the readiness probe (no auth required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8800/health/ready || exit 1

USER swarmllm
WORKDIR /data

ENTRYPOINT ["swarmllm"]
CMD ["run"]
