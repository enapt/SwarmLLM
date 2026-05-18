# ============================================================================
# SwarmLLM — Multi-stage Docker Image
# ============================================================================
# CPU:   docker build -t swarmllm .
# Run:   docker run -p 8800:8800 -p 8810:8810 swarmllm
# Data:  docker run -p 8800:8800 -p 8810:8810 -v swarmllm-data:/data swarmllm
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
# R138 (closes R103 deferral): pin base images by digest so a registry-side
# tag move (compromise, accidental retag, supply-chain shift) cannot silently
# change the build inputs. Tag retained alongside `@sha256:` for human
# readability — Docker still pulls by digest. Re-capture the digest with
# `docker buildx imagetools inspect rust:1.94-bookworm` when bumping the tag.
FROM rust:1.94-bookworm@sha256:6ae102bdbf528294bc79ad6e1fae682f6f7c2a6e6621506ba959f9685b308a55 AS builder

# Install system dependencies required for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Override the workspace's `.cargo/config.toml` `target-cpu=native` so the
# image is portable across CPUs other than the build runner. Without this
# the binary SIGILLs on machines whose microarchitecture differs from the
# CI runner that built the image. See gotcha #39 in memory/MEMORY.md and
# the matching `RUSTFLAGS: ""` in `.github/workflows/{ci,release}.yml`.
ENV RUSTFLAGS=""

WORKDIR /build

# Cache dependency compilation: copy manifests first, build a dummy to populate
# the cargo registry and compile deps, then replace with real source.
COPY Cargo.toml Cargo.lock build.rs ./
COPY vendor/ vendor/
COPY crates/swarmllm-frontend/Cargo.toml crates/swarmllm-frontend/Cargo.toml
COPY crates/swarmllm-types/Cargo.toml crates/swarmllm-types/Cargo.toml

# Create dummy sources for dependency caching
RUN mkdir -p src crates/swarmllm-frontend/src crates/swarmllm-types/src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs && \
    echo '' > src/lib.rs && \
    echo '' > crates/swarmllm-frontend/src/lib.rs && \
    echo '' > crates/swarmllm-types/src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src crates/swarmllm-frontend/src crates/swarmllm-types/src

# Copy full source and build the real binary
COPY src/ src/
COPY crates/ crates/
COPY frontend/ frontend/
COPY config/ config/
RUN cargo build --release

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
# R138 (closes R103 deferral): digest-pinned for the same reasons as the
# builder stage above. Re-capture with
# `docker buildx imagetools inspect debian:bookworm-slim` when bumping.
FROM debian:bookworm-slim@sha256:67b30a61dc87758f0caf819646104f29ecbda97d920aaf5edc834128ac8493d3

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
