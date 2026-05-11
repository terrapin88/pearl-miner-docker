# Pearl Mining Optimization Plan — Remaining Levers

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Maximize Pearl mining hash rate (tiles/second) through software optimizations layered on top of the existing fleet.

**Current baseline:** 16 instances, 26 GPUs (H200/H100), ~97% GPU util, ~$87/hr, `sha-276199c` image.

**Canary in flight:** `sha-e84897b` (optimize-batching branch) — tests vLLM flag changes (max-batched-tokens, max-num-seqs, gpu-memory-utilization, clock locking, connection pooling). Results pending.

---

## Lever Inventory — Prioritized by Impact × Feasibility

### ✅ PHASE 0 — In Canary (building now)
| # | Lever | What | Expected Impact |
|---|-------|------|----------------|
| 0a | `--max-num-batched-tokens 32768` | Directly increases M dimension for NoisyGEMM | **HIGH** — more hash tiles per kernel |
| 0b | `--max-num-seqs 256` | Allow larger concurrent batch | **MEDIUM** |
| 0c | `--gpu-memory-utilization 0.95` | More KV cache headroom | **LOW-MED** |
| 0d | GPU clock locking | `nvidia-smi -pm 1 -lgc 1200,2100` | **MEDIUM** — 5-15% sustained |
| 0e | CUDA env tuning | `CUDA_DEVICE_MAX_CONNECTIONS=8`, `CUDA_MODULE_LOADING=LAZY` | **LOW** |
| 0f | Connection pooling | `requests.Session` with keep-alive | **LOW** — reduces per-req latency |
| 0g | `--disable-log-stats/requests` | Eliminate Python logging overhead | **LOW** |

---

### 🔥 PHASE 1 — Pre-tokenized Prompts (Low effort, Medium impact)
**What:** Send `prompt_token_ids` directly to vLLM instead of text, skipping tokenizer entirely.
**Why:** Each of 64-256 workers currently sends a text prompt that vLLM tokenizes. That's 64-256 tokenizer calls per batch cycle — pure CPU waste.
**Effort:** ~1 hour. Modify `pearl_worker.py` to pre-tokenize once at startup, then send token IDs.
**Risk:** Low — vLLM's `/v1/completions` endpoint supports `prompt_token_ids` natively.

**Implementation:**
1. At worker startup, load tokenizer via `transformers.AutoTokenizer`
2. Pre-generate N prompt strings → tokenize all → store as `List[List[int]]`
3. POST to `/v1/completions` (not chat) with `{"prompt_token_ids": [...], "max_tokens": 1}`
4. Eliminates tokenizer overhead + chat template formatting

---

### 🔥 PHASE 2 — Batch Accumulator Sidecar (Medium effort, High impact)
**What:** A lightweight async service between workers and vLLM that ensures maximum batch saturation every scheduler step.
**Why:** Current workers fire independently — vLLM's scheduler may process partial batches (e.g., 12 requests when 64 are in flight but arrived at different times). A sidecar accumulates requests and fires them in tight bursts, ensuring the scheduler always sees a full batch.
**Effort:** ~3 hours. New Python file `pearl_batch_sidecar.py`, ~150 lines.
**Risk:** Medium — need to tune accumulation window (too long = GPU idle, too short = small batches).

**Architecture:**
```
Workers (256 threads) → Sidecar (:8001) → vLLM (:8000)
                         accumulate 50ms
                         fire burst
                         double-buffer
```

**Implementation:**
1. Create `pearl_batch_sidecar.py` using `aiohttp` (already in vLLM deps)
2. Accept requests on port 8001, queue them
3. Every `BATCH_WINDOW_MS` (default 50ms) or when queue hits `BATCH_TARGET` requests, fire all at vLLM simultaneously using `asyncio.gather`
4. Monitor vLLM `/metrics` for `vllm:num_requests_running` to tune timing
5. Add to entrypoint.sh after vLLM starts, before worker starts

---

### 🔥 PHASE 3 — Tile-Aligned Prompt Lengths (Low effort, Low-Med impact)
**What:** Ensure total batch token count is a multiple of 128 (NoisyGEMM tile_size_m).
**Why:** If M=8200, that's 64 full tiles + 1 partial tile with only 8 tokens wasted. But if prompts vary wildly, waste could be higher. Standardizing prompt length to exact tile boundaries eliminates waste.
**Effort:** ~30 min. Adjust `WORD_LIST_LENGTH` so tokenized prompt is exactly 2048 or 2560 tokens (multiples of 128).
**Risk:** Very low.

**Implementation:**
1. Tokenize a sample prompt, measure exact token count
2. Adjust WORD_LIST_LENGTH to hit 2048 tokens (16 tiles) or 2560 tokens (20 tiles)
3. Bake this calibrated value into the image

---

### ⚡ PHASE 4 — Unix Domain Socket for Worker→vLLM (Low effort, Low-Med impact)
**What:** Replace TCP localhost with UDS for worker→vLLM communication.
**Why:** Eliminates TCP overhead (~30μs per request) — SYN/ACK, Nagle, buffer copies.
**Effort:** ~30 min. Add `--unix-socket /tmp/vllm.sock` to vLLM, update worker URL.
**Risk:** Low — well-supported in vLLM. Need to verify compatible with data-parallel mode.

**Implementation:**
1. Add `--unix-socket /tmp/vllm.sock` to vLLM serve command
2. Update `PEARL_VLLM_URL` to `http+unix:///tmp/vllm.sock/v1/completions`
3. Worker needs `requests_unixsocket` or `httpx` with transport

---

### 🧪 PHASE 5 — OS-Level Tuning (Low effort, Low impact per item but cumulative)
**What:** NUMA pinning, huge pages, IRQ affinity — squeeze the last ~5-10%.
**Effort:** ~1 hour. All entrypoint.sh changes.
**Risk:** Low — all `|| true` guarded, fails silently on unsupported hosts.

**Implementation:**
```bash
# Transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true

# Increase network buffers (faster block submission)
sysctl -w net.core.rmem_max=16777216 2>/dev/null || true
sysctl -w net.core.wmem_max=16777216 2>/dev/null || true

# NUMA auto-binding (if numactl available)
# Pin vLLM to GPU-local NUMA node
```

---

### 🧪 PHASE 6 — Smaller Model Variants (Research, potentially HIGH impact)
**What:** Test whether smaller Pearl-compatible models (e.g., Qwen 9B, Gemma 31B) mine more efficiently per GPU-dollar.
**Why:** A 9B model processes prefill MUCH faster than 70B (fewer layers, smaller matrices). While each GEMM has fewer tiles (smaller K/N), the throughput gain might compensate. If 9B does 4× more requests/sec but 0.5× tiles per request, net is 2× improvement.
**Effort:** Research + 1 canary deploy (~2 hours).
**Risk:** Medium-high — may not work if NoisyGEMM requires minimum K/N dimensions from the 70B architecture. Need to check `min_m=1024, min_n=1024, min_k=1024` thresholds against smaller model hidden dims.

**Key question:** Does Qwen-9B have hidden_dim ≥ 1024? (Likely yes: most 7B+ models have hidden_dim ≥ 4096)

---

### 🔬 PHASE 7 — Prompt Content Optimization (Research, Unknown impact)
**What:** Investigate whether certain token patterns produce more efficient matrix operations.
**Why:** While the GEMM dimensions are fixed by model architecture, different activation patterns could affect memory access patterns and cache hit rates in the GPU.
**Effort:** ~2 hours of benchmarking.
**Risk:** Likely low impact — GEMM performance is generally input-agnostic. But worth a quick test.
**Approach:** Compare tok/s with random tokens vs repeated tokens vs natural text.

---

## Execution Priority

```
PHASE 0 ← IN FLIGHT (canary building)
  ↓ measure delta
PHASE 1 ← NEXT (pre-tokenized prompts, ~1hr, easy win)
PHASE 3 ← BUNDLE WITH 1 (tile alignment, 30min)
PHASE 4 ← BUNDLE WITH 1 (UDS, 30min)
  ↓ ship as sha-XXX canary #2
PHASE 2 ← THEN (batch sidecar, ~3hr, highest potential)
  ↓ ship as sha-XXX canary #3
PHASE 5 ← QUICK ADD (OS tuning, 1hr)
PHASE 6 ← RESEARCH PARALLEL (smaller models)
PHASE 7 ← SKIP UNLESS BORED
```

**Total estimated effort for Phases 1-5:** ~6 hours
**Expected cumulative improvement over current baseline:** 15-40% more tiles/second at same cost

---

## How to Measure

For each canary, compare against a production `sha-276199c` instance on the same GPU type:

1. **Prompt throughput:** `vastai logs <id> | grep "tok/s"` — higher is better
2. **Request rate:** Worker stats line `req/s=X` — higher means less overhead
3. **Block rate:** Blocks found per 24h (statistical, need large sample)
4. **GPU utilization:** Should stay 97%+ — if it drops, we're bottlenecking elsewhere
5. **GPU temperature:** Should not increase significantly (thermal = throttle risk)
