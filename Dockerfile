# Pearl Miner — All-in-One Docker Image
# Mines PRL tokens by running LLM inference on H100/H200 GPUs
#
# Usage:
#   docker run --gpus all -e PEARL_WALLET_ADDRESS=prl1... -e HF_TOKEN=hf_... pearl-miner
#
# Build (from this repo root):
#   docker buildx build -t pearl-miner .

ARG CUDA_VERSION=12.9.1

# ============================================================
# Stage 1: Builder — compile CUDA kernels + Python packages
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=docker.io/astral/uv:latest /uv /usr/local/bin/uv

# Install Rust toolchain (required for zk-pow and py-pearl-mining)
COPY --from=rust:latest /usr/local/cargo /usr/local/cargo
COPY --from=rust:latest /usr/local/rustup /usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}" \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
RUN rustup self update && rustup default stable

# uv environment setup
ENV UV_PYTHON_INSTALL_DIR=/usr/local/uv-python \
    UV_PYTHON_BIN_DIR=/usr/local/bin \
    UV_PROJECT_ENVIRONMENT=/usr/local/venv \
    UV_LINK_MODE=copy \
    UV_TORCH_BACKEND=cu129 \
    UV_COMPILE_BYTECODE=true

# Install Python 3.12
RUN uv python install 3.12 --default

WORKDIR /app

# Clone Pearl monorepo
RUN git clone --depth 1 https://github.com/pearl-research-labs/pearl.git .

# Install dependencies (remote packages first for caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    uv sync --package vllm-miner --no-editable --no-dev --no-install-workspace

# Build CUDA kernels (pearl-gemm for sm90 — H100/H200)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    uv sync --package pearl-gemm --no-editable --no-dev

# Install full vllm-miner + workspace deps
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    uv sync --package vllm-miner --no-install-package pearl-gemm --no-editable --no-dev --inexact

# ============================================================
# Stage 2: Download Go binaries (pearld, oyster, prlctl)
# ============================================================
FROM ubuntu:24.04 AS go-bins

RUN apt-get update && apt-get install -y curl tar && rm -rf /var/lib/apt/lists/*

RUN curl -LO https://github.com/pearl-research-labs/pearl/releases/download/pearl-wallet-v1.0.0/go-binaries-linux-amd64-v1.0.2.tar.gz \
    && mkdir -p /go-bins \
    && tar xzf go-binaries-linux-amd64-v1.0.2.tar.gz -C /go-bins \
    && chmod +x /go-bins/*

# ============================================================
# Stage 3: Runtime — slim image with everything needed
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc-dev curl jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python + venv from builder
COPY --from=builder /usr/local/uv-python /usr/local/uv-python
COPY --from=builder /usr/local/venv /usr/local/venv

# Copy Go binaries (pearld, oyster, prlctl)
COPY --from=go-bins /go-bins/ /usr/local/bin/

# Copy entrypoint and mining worker
COPY entrypoint.sh /app/entrypoint.sh
COPY pearl_worker.py /app/pearl_worker.py
RUN chmod +x /app/entrypoint.sh

# Note: 'requests' is already installed as a vLLM dependency
# Install requests-unixsocket for UDS support and transformers for pre-tokenization
RUN /usr/local/venv/bin/pip install --no-cache-dir requests-unixsocket 2>/dev/null || true

# Environment
ENV PATH="/usr/local/venv/bin:${PATH}" \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

# Ports: vLLM API, Pearl P2P, Gateway metrics
EXPOSE 8000 8337 8339

# Volumes for persistent data
VOLUME ["/app/chain-data", "/root/.cache/huggingface"]

ENTRYPOINT ["/app/entrypoint.sh"]
