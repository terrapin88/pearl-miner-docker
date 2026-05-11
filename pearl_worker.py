#!/usr/bin/env python3
"""
Pearl Mining Request Worker — Optimized v2

Mining ONLY happens when inference requests are being processed.
The NoisyGEMM kernel finds blocks as a by-product of matrix multiplication
during LLM inference. No requests = no mining.

Optimizations over v1:
  - HTTP connection pooling (requests.Session with keep-alive)
  - Pre-generated prompt batches to reduce per-request overhead
  - Configurable prompt token target for tile alignment
  - Stats include throughput metrics
"""
import os
import random
import signal
import sys
import threading
import time

import requests as req

VLLM_URL = os.environ.get("PEARL_VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = "pearl-ai/Llama-3.3-70B-Instruct-pearl"
NUM_WORKERS = int(os.environ.get("PEARL_WORKERS", "64"))
MAX_TOKENS = int(os.environ.get("PEARL_MAX_TOKENS", "1"))
REQUEST_TIMEOUT = 180
# CRITICAL: Prompt must be 1024+ tokens to trigger NoisyGEMM (min_m=1024).
# With ~1.75 tokens per random word, 1400 words ≈ 2600 tokens.
# More tokens = larger M dimension = more hash tiles per GEMM = more mining.
WORD_LIST_LENGTH = int(os.environ.get("PEARL_WORD_LIST", "1400"))

# Pre-generate a pool of prompts to reduce per-request generation overhead
PROMPT_POOL_SIZE = int(os.environ.get("PEARL_PROMPT_POOL_SIZE", "100"))

CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"


def random_word(length=None):
    if length is None:
        length = random.randint(4, 10)
    return "".join(
        random.choice(CONSONANTS if i % 2 == 0 else VOWELS)
        for i in range(length)
    )


def build_prompt():
    """Generate a random prompt that triggers large matrix multiplications."""
    bypass = random_word(random.randint(5, 12))
    words = " ".join(random_word() for _ in range(WORD_LIST_LENGTH))
    return bypass + ", decipher this secret message: " + words


def build_prompt_pool(size):
    """Pre-generate a pool of prompts to reduce per-request overhead."""
    print(f"[Pool] Pre-generating {size} prompts ({WORD_LIST_LENGTH} words each)...", flush=True)
    t0 = time.time()
    pool = [build_prompt() for _ in range(size)]
    elapsed = time.time() - t0
    print(f"[Pool] Generated {size} prompts in {elapsed:.1f}s", flush=True)
    return pool


class MiningWorker(threading.Thread):
    def __init__(self, wid, prompt_pool):
        super().__init__(daemon=True)
        self.wid = wid
        self.count = 0
        self.running = True
        self.prompt_pool = prompt_pool
        # Each worker gets its own session for connection pooling + keep-alive
        self.session = req.Session()
        adapter = req.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0,
        )
        self.session.mount("http://", adapter)

    def run(self):
        print(f"[W{self.wid}] Started", flush=True)
        pool_idx = self.wid % len(self.prompt_pool)

        while self.running:
            try:
                # Cycle through pre-generated prompts, occasionally regenerate
                prompt = self.prompt_pool[pool_idx % len(self.prompt_pool)]
                pool_idx += 1

                # Regenerate prompt every 500 requests to avoid any caching effects
                if self.count > 0 and self.count % 500 == 0:
                    prompt = build_prompt()

                r = self.session.post(
                    VLLM_URL,
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": MAX_TOKENS,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == 200:
                    self.count += 1
                    if self.count % 100 == 0:
                        out = (
                            r.json()
                            .get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        print(
                            f"[W{self.wid}] req={self.count} out='{out.strip()[:30]}'",
                            flush=True,
                        )
                else:
                    time.sleep(1)
            except req.exceptions.Timeout:
                print(f"[W{self.wid}] Timeout, retrying...", flush=True)
                time.sleep(2)
            except req.exceptions.ConnectionError:
                print(f"[W{self.wid}] ConnError, retrying in 5s...", flush=True)
                time.sleep(5)
            except Exception as e:
                print(f"[W{self.wid}] Error: {e}", flush=True)
                time.sleep(2)


def stats(workers, start_time):
    """Print stats every 30 seconds with throughput metrics."""
    while True:
        time.sleep(30)
        total = sum(w.count for w in workers)
        active = sum(1 for w in workers if w.is_alive())
        elapsed = time.time() - start_time
        rps = total / elapsed if elapsed > 0 else 0
        print(
            f"[Stats] total_requests={total} active_workers={active}/{NUM_WORKERS} "
            f"req/s={rps:.1f} uptime={elapsed/60:.0f}m",
            flush=True,
        )


def main():
    print(f"🐚 Pearl Mining Worker v2 — {NUM_WORKERS} threads, max_tokens={MAX_TOKENS}", flush=True)
    print(f"   Target: {VLLM_URL}", flush=True)
    print(f"   Word list: {WORD_LIST_LENGTH}, Prompt pool: {PROMPT_POOL_SIZE}", flush=True)
    print(f"   Connection pooling: enabled", flush=True)

    # Pre-generate prompt pool
    prompt_pool = build_prompt_pool(PROMPT_POOL_SIZE)

    workers = [MiningWorker(i, prompt_pool) for i in range(NUM_WORKERS)]
    start_time = time.time()

    def shutdown(s, f):
        print("\n🛑 Shutting down workers...", flush=True)
        for w in workers:
            w.running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for w in workers:
        w.start()

    threading.Thread(target=stats, args=(workers, start_time), daemon=True).start()

    # Keep main thread alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
