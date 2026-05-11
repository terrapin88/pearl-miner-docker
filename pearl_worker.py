#!/usr/bin/env python3
"""
Pearl Mining Request Worker — Optimized v3

Mining ONLY happens when inference requests are being processed.
The NoisyGEMM kernel finds blocks as a by-product of matrix multiplication
during LLM inference. No requests = no mining.

Optimizations over v1/v2:
  - Pre-tokenized prompts: sends prompt_token_ids directly (skips tokenizer)
  - Tile-aligned prompt lengths: calibrated to 128-token tile boundaries
  - HTTP connection pooling via requests.Session
  - Pre-generated prompt pool to reduce per-request overhead
  - UDS support (when PEARL_VLLM_SOCKET is set)
  - Uses /v1/completions endpoint (lighter than /v1/chat/completions)
"""
import os
import random
import signal
import sys
import threading
import time
from typing import List

import requests as req

# --- Configuration ---
# Use /v1/completions (not chat) — supports prompt_token_ids, less overhead
VLLM_BASE = os.environ.get("PEARL_VLLM_URL", "http://localhost:8000")
VLLM_URL = f"{VLLM_BASE}/v1/completions"
MODEL = "pearl-ai/Llama-3.3-70B-Instruct-pearl"
NUM_WORKERS = int(os.environ.get("PEARL_WORKERS", "64"))
MAX_TOKENS = int(os.environ.get("PEARL_MAX_TOKENS", "1"))
REQUEST_TIMEOUT = 180

# Prompt sizing — calibrated for tile alignment
# NoisyGEMM tile_size_m = 128. More tiles = more hash checks.
# H200: 1400 words ≈ 4654 tokens ≈ 36 tiles (fits in 8192 context)
# H100: 700 words ≈ 2326 tokens ≈ 18 tiles (fits in 4096 context)
WORD_LIST_LENGTH = int(os.environ.get("PEARL_WORD_LIST", "1400"))

# Pre-tokenization: if enabled, sends token IDs directly to vLLM
# Requires tokenizer to be available. Falls back to text if unavailable.
USE_PRETOKENIZED = os.environ.get("PEARL_PRETOKENIZE", "1") == "1"

# Pool of pre-built prompts (text or token IDs)
PROMPT_POOL_SIZE = int(os.environ.get("PEARL_PROMPT_POOL_SIZE", "200"))

CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"

# Global tokenizer (loaded once)
_tokenizer = None


def load_tokenizer():
    """Load the model's tokenizer for pre-tokenization."""
    global _tokenizer
    try:
        from transformers import AutoTokenizer
        print("[Tokenizer] Loading pearl-ai/Llama-3.3-70B-Instruct-pearl tokenizer...", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(
            "pearl-ai/Llama-3.3-70B-Instruct-pearl",
            use_fast=True,
        )
        print(f"[Tokenizer] Loaded! Vocab size: {_tokenizer.vocab_size}", flush=True)
        return True
    except Exception as e:
        print(f"[Tokenizer] Failed to load: {e}", flush=True)
        print("[Tokenizer] Falling back to text prompts", flush=True)
        return False


def random_word(length=None):
    if length is None:
        length = random.randint(4, 10)
    return "".join(
        random.choice(CONSONANTS if i % 2 == 0 else VOWELS)
        for i in range(length)
    )


def build_prompt_text():
    """Generate a random prompt string."""
    bypass = random_word(random.randint(5, 12))
    words = " ".join(random_word() for _ in range(WORD_LIST_LENGTH))
    return bypass + ", decipher this secret message: " + words


def build_prompt_tokens() -> List[int]:
    """Generate a random prompt and tokenize it, aligned to tile boundaries."""
    text = build_prompt_text()
    token_ids = _tokenizer.encode(text)
    # Align to nearest 128-token boundary (tile_size_m)
    # Truncate to nearest multiple of 128 (wastes at most 127 tokens = <3% for typical prompts)
    aligned_len = (len(token_ids) // 128) * 128
    if aligned_len < 128:
        aligned_len = 128
    return token_ids[:aligned_len]


def build_prompt_pool(size: int, pretokenized: bool):
    """Pre-generate a pool of prompts."""
    print(f"[Pool] Pre-generating {size} prompts (pretokenized={pretokenized}, "
          f"words={WORD_LIST_LENGTH})...", flush=True)
    t0 = time.time()

    if pretokenized and _tokenizer:
        pool = [build_prompt_tokens() for _ in range(size)]
        avg_tokens = sum(len(p) for p in pool) / len(pool)
        avg_tiles = avg_tokens / 128
        print(f"[Pool] Generated {size} token sequences in {time.time()-t0:.1f}s", flush=True)
        print(f"[Pool] Avg length: {avg_tokens:.0f} tokens ({avg_tiles:.0f} tiles per prompt)", flush=True)
    else:
        pool = [build_prompt_text() for _ in range(size)]
        print(f"[Pool] Generated {size} text prompts in {time.time()-t0:.1f}s", flush=True)

    return pool


class MiningWorker(threading.Thread):
    def __init__(self, wid, prompt_pool, pretokenized: bool):
        super().__init__(daemon=True)
        self.wid = wid
        self.count = 0
        self.running = True
        self.prompt_pool = prompt_pool
        self.pretokenized = pretokenized
        # Connection pooling with keep-alive
        self.session = req.Session()
        adapter = req.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0,
        )
        self.session.mount("http://", adapter)
        self.session.mount("http+unix://", adapter)

    def _build_payload(self, prompt):
        """Build the request payload."""
        if self.pretokenized and isinstance(prompt, list):
            # Send token IDs directly — skips tokenizer on vLLM side
            return {
                "model": MODEL,
                "prompt_token_ids": prompt,
                "max_tokens": MAX_TOKENS,
            }
        else:
            # Fallback: send text via completions endpoint
            return {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
            }

    def run(self):
        print(f"[W{self.wid}] Started (pretokenized={self.pretokenized})", flush=True)
        pool_size = len(self.prompt_pool)
        pool_idx = self.wid  # Stagger starting positions

        while self.running:
            try:
                prompt = self.prompt_pool[pool_idx % pool_size]
                pool_idx += 1

                # Occasionally regenerate to avoid any potential caching
                if self.count > 0 and self.count % 1000 == 0:
                    if self.pretokenized and _tokenizer:
                        prompt = build_prompt_tokens()
                    else:
                        prompt = build_prompt_text()

                payload = self._build_payload(prompt)

                r = self.session.post(
                    VLLM_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == 200:
                    self.count += 1
                    if self.count % 200 == 0:
                        print(
                            f"[W{self.wid}] req={self.count}",
                            flush=True,
                        )
                elif r.status_code == 400:
                    # Likely prompt_token_ids not supported — fall back to text
                    error_msg = r.text[:200]
                    if "prompt_token_ids" in error_msg and self.pretokenized:
                        print(f"[W{self.wid}] prompt_token_ids not supported, switching to text", flush=True)
                        self.pretokenized = False
                        self.prompt_pool = [build_prompt_text() for _ in range(50)]
                    else:
                        print(f"[W{self.wid}] 400 error: {error_msg}", flush=True)
                        time.sleep(2)
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
    last_total = 0
    last_time = start_time

    while True:
        time.sleep(30)
        total = sum(w.count for w in workers)
        active = sum(1 for w in workers if w.is_alive())
        now = time.time()
        elapsed = now - start_time
        interval = now - last_time
        rps_overall = total / elapsed if elapsed > 0 else 0
        rps_interval = (total - last_total) / interval if interval > 0 else 0
        last_total = total
        last_time = now
        print(
            f"[Stats] total={total} active={active}/{NUM_WORKERS} "
            f"req/s={rps_interval:.1f} (avg {rps_overall:.1f}) uptime={elapsed/60:.0f}m",
            flush=True,
        )


def main():
    print(f"🐚 Pearl Mining Worker v3 — {NUM_WORKERS} threads", flush=True)
    print(f"   Endpoint: {VLLM_URL}", flush=True)
    print(f"   max_tokens={MAX_TOKENS}, word_list={WORD_LIST_LENGTH}", flush=True)
    print(f"   Pre-tokenization: {'enabled' if USE_PRETOKENIZED else 'disabled'}", flush=True)

    # Load tokenizer if pre-tokenization is enabled
    pretokenized = False
    if USE_PRETOKENIZED:
        pretokenized = load_tokenizer()

    # Pre-generate prompt pool
    prompt_pool = build_prompt_pool(PROMPT_POOL_SIZE, pretokenized)

    workers = [MiningWorker(i, prompt_pool, pretokenized) for i in range(NUM_WORKERS)]
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
