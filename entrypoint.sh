#!/bin/bash
set -e

# ============================================================
# Pearl Miner — All-in-One Entrypoint
# Starts: pearld (node) → pearl-gateway (miner) → vLLM (inference)
# ============================================================

echo "🐚 Pearl Miner starting up..."
echo "   Wallet: ${PEARL_WALLET_ADDRESS:-NOT SET}"
echo "   GPU Memory Utilization: ${PEARL_GPU_UTIL:-0.9}"
echo "   Max Model Length: ${PEARL_MAX_MODEL_LEN:-8192}"

# Validate required env vars
if [ -z "$PEARL_WALLET_ADDRESS" ]; then
    echo "❌ ERROR: PEARL_WALLET_ADDRESS is required!"
    echo "   Set it to your prl1... address"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN is required!"
    echo "   Get one at https://huggingface.co/settings/tokens"
    exit 1
fi

# Export HF token for model download
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ============================================================
# Step 1: Start Pearl full node
# ============================================================
echo "📦 Starting Pearl node (pearld)..."

# Create chain data directory
mkdir -p /app/chain-data

# Generate random RPC credentials for internal use
RPC_USER="miner_$(cat /proc/sys/kernel/random/uuid | tr -d '-' | head -c 16)"
RPC_PASS="$(cat /proc/sys/kernel/random/uuid | tr -d '-')"

# Start pearld
pearld \
    --datadir=/app/chain-data \
    --rpcuser="$RPC_USER" \
    --rpcpass="$RPC_PASS" \
    --miningaddr="$PEARL_WALLET_ADDRESS" \
    --listen=:44108 \
    --rpclisten=:44107 \
    --notls \
    &
PEARLD_PID=$!

# Wait for node RPC to be ready
echo "⏳ Waiting for Pearl node to start..."
for i in $(seq 1 60); do
    if curl -s --user "$RPC_USER:$RPC_PASS" \
        --data-binary '{"jsonrpc":"1.0","id":"startup","method":"getinfo","params":[]}' \
        -H 'content-type: text/plain;' \
        http://localhost:44107/ > /dev/null 2>&1; then
        echo "✅ Pearl node is running"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ Pearl node failed to start after 60s"
        exit 1
    fi
    sleep 1
done

# Show sync status
BLOCK_COUNT=$(curl -s --user "$RPC_USER:$RPC_PASS" \
    --data-binary '{"jsonrpc":"1.0","id":"sync","method":"getblockcount","params":[]}' \
    -H 'content-type: text/plain;' \
    http://localhost:44107/ | jq -r '.result // "unknown"')
echo "📊 Current block height: $BLOCK_COUNT"

# ============================================================
# Step 1b: Wait for chain to fully sync before proceeding
# The vllm_miner plugin REQUIRES a valid block template on startup.
# If the gateway returns "mining_paused: no block template available",
# vLLM will fatally crash. We must wait for full sync first.
# ============================================================
echo "⏳ Waiting for chain to fully sync (this may take 10-30 min from genesis)..."
SYNC_TIMEOUT=3600  # 1 hour max
SYNC_ELAPSED=0
while [ $SYNC_ELAPSED -lt $SYNC_TIMEOUT ]; do
    # Check if node is still downloading blocks
    SYNC_CHECK=$(curl -s --user "$RPC_USER:$RPC_PASS" \
        --data-binary '{"jsonrpc":"1.0","id":"sync","method":"getblocktemplate","params":[]}' \
        -H 'content-type: text/plain;' \
        http://localhost:44107/ 2>/dev/null)
    
    # If we get error code -10 ("downloading blocks"), keep waiting
    ERROR_CODE=$(echo "$SYNC_CHECK" | jq -r '.error.code // 0')
    if [ "$ERROR_CODE" != "-10" ]; then
        CURRENT_HEIGHT=$(curl -s --user "$RPC_USER:$RPC_PASS" \
            --data-binary '{"jsonrpc":"1.0","id":"h","method":"getblockcount","params":[]}' \
            -H 'content-type: text/plain;' \
            http://localhost:44107/ | jq -r '.result // "?"')
        echo "✅ Chain fully synced at height $CURRENT_HEIGHT!"
        break
    fi
    
    if [ $((SYNC_ELAPSED % 30)) -eq 0 ]; then
        CURRENT_HEIGHT=$(curl -s --user "$RPC_USER:$RPC_PASS" \
            --data-binary '{"jsonrpc":"1.0","id":"h","method":"getblockcount","params":[]}' \
            -H 'content-type: text/plain;' \
            http://localhost:44107/ | jq -r '.result // "?"')
        echo "   Syncing... height=$CURRENT_HEIGHT (${SYNC_ELAPSED}s elapsed)"
    fi
    sleep 5
    SYNC_ELAPSED=$((SYNC_ELAPSED + 5))
done

if [ $SYNC_ELAPSED -ge $SYNC_TIMEOUT ]; then
    echo "❌ Chain sync timed out after ${SYNC_TIMEOUT}s"
    exit 1
fi

# ============================================================
# Step 2: Start Pearl Gateway (the mining bridge)
# ============================================================
echo "⛏️  Starting Pearl Gateway..."

export PEARLD_RPC_URL="http://localhost:44107"
export PEARLD_RPC_USER="$RPC_USER"
export PEARLD_RPC_PASSWORD="$RPC_PASS"
export PEARLD_MINING_ADDRESS="$PEARL_WALLET_ADDRESS"

pearl-gateway start &
GATEWAY_PID=$!

# Wait for gateway metrics endpoint
echo "⏳ Waiting for gateway to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8339/metrics > /dev/null 2>&1; then
        echo "✅ Pearl Gateway is running"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Gateway not responding on :8339 yet, continuing anyway..."
    fi
    sleep 1
done

# ============================================================
# Step 3: Auto-detect GPUs and configure parallelism
# ============================================================
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
        export CUDA_VISIBLE_DEVICES
        echo "🎮 Auto-detected $GPU_COUNT GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
else
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

# Set data parallelism based on GPU count
DP_SIZE="${PEARL_DP_SIZE:-$GPU_COUNT}"

# Critical: disable deep gemm (conflicts with Pearl's NoisyGEMM)
export VLLM_USE_DEEP_GEMM=0

# OPTIMIZATION: Allow CUDA graphs for non-mining layers (0-39)
# Set PEARL_ENFORCE_EAGER=1 to disable CUDA graphs if mining breaks
if [ "${PEARL_ENFORCE_EAGER:-0}" = "1" ]; then
    EAGER_FLAG="--enforce-eager"
    echo "🔧 CUDA graphs disabled (enforce-eager mode)"
else
    EAGER_FLAG=""
    echo "🔧 CUDA graphs enabled (faster non-mining layers)"
fi

# Auto-detect VRAM and adjust max_model_len for H100 (80GB) vs H200 (141GB)
if [ -z "$PEARL_MAX_MODEL_LEN" ]; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$VRAM_MB" ] && [ "$VRAM_MB" -lt 100000 ]; then
        # H100 80GB — tight fit, use shorter context + higher memory util
        PEARL_MAX_MODEL_LEN=4096
        PEARL_GPU_UTIL="${PEARL_GPU_UTIL:-0.95}"
        echo "🔧 H100 detected (${VRAM_MB}MB) — using max_model_len=4096, gpu_util=0.95"
    else
        # H200 141GB — plenty of room
        PEARL_MAX_MODEL_LEN=8192
        PEARL_GPU_UTIL="${PEARL_GPU_UTIL:-0.9}"
        echo "🔧 H200 detected (${VRAM_MB}MB) — using max_model_len=8192, gpu_util=0.9"
    fi
fi

echo "🚀 Starting vLLM inference server..."
echo "   Model: pearl-ai/Llama-3.3-70B-Instruct-pearl"
echo "   Data Parallel Size: $DP_SIZE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Mining will begin once chain is synced + vLLM ready."
echo "  First sync takes ~2 hours from genesis."
echo "  Monitor: http://localhost:8339/metrics"
echo "  Check your stats: https://lordofpearls.xyz"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start vLLM in background (not exec — we need to start the worker loop too)
vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len "$PEARL_MAX_MODEL_LEN" \
    --gpu-memory-utilization "${PEARL_GPU_UTIL:-0.9}" \
    $EAGER_FLAG \
    --data-parallel-size "$DP_SIZE" \
    --no-enable-prefix-caching \
    --max-num-seqs "${PEARL_MAX_SEQS:-64}" \
    &
VLLM_PID=$!

# ============================================================
# Step 4: Wait for vLLM to be ready, then start request worker
# ============================================================
echo "⏳ Waiting for vLLM to load model and become ready..."
for i in $(seq 1 1800); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ vLLM is ready! Starting mining worker..."
        break
    fi
    if [ $i -eq 1800 ]; then
        echo "❌ vLLM failed to start after 30 minutes"
        exit 1
    fi
    # Print progress every 30s
    if [ $((i % 30)) -eq 0 ]; then
        echo "   Still loading model... (${i}s elapsed)"
    fi
    sleep 1
done

# Start the mining request worker
# Mining ONLY happens when inference requests are being processed!
# The NoisyGEMM kernel finds blocks as a by-product of matrix multiplication
echo "⛏️  Starting mining request worker (${PEARL_WORKERS:-32} threads)..."
python3 /app/pearl_worker.py &
WORKER_PID=$!

# ============================================================
# Step 5: Watchdog — restart ALL components if they die
# ============================================================
echo "🔄 Starting watchdog..."
while true; do
    # Check vLLM
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ vLLM died! Exiting container."
        exit 1
    fi
    # Check gateway (CRITICAL — without it, mining proofs go nowhere)
    if ! kill -0 $GATEWAY_PID 2>/dev/null; then
        echo "⚠️  Gateway died! Restarting..."
        pearl-gateway start &
        GATEWAY_PID=$!
        sleep 5
        echo "✅ Gateway restarted (PID=$GATEWAY_PID)"
    fi
    # Check worker
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo "⚠️  Worker died, restarting..."
        python3 /app/pearl_worker.py &
        WORKER_PID=$!
    fi
    sleep 30
done
