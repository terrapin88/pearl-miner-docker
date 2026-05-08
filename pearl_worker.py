#!/usr/bin/env python3
"""
Pearl Mining Request Worker

Mining ONLY happens when inference requests are being processed.
The NoisyGEMM kernel finds blocks as a by-product of matrix multiplication
during LLM inference. No requests = no mining.

This worker sends a constant stream of requests to the local vLLM server
to keep the GPU busy and mining.
"""
import os
import random
import signal
import sys
import threading
import time

import requests as req

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "pearl-ai/Llama-3.3-70B-Instruct-pearl"
NUM_WORKERS = int(os.environ.get("PEARL_WORKERS", "64"))
MAX_TOKENS = int(os.environ.get("PEARL_MAX_TOKENS", "1"))  # Minimize decode time — mining only happens during prefill
REQUEST_TIMEOUT = 180
# CRITICAL: Prompt must be 1024+ tokens to trigger NoisyGEMM (min_m=1024).
# With ~1.75 tokens per random word, 700 words ≈ 1200 tokens.
# Higher word count = larger M dimension = more hash tiles per GEMM = more mining.
# But too high = slower prefill = fewer requests/sec. Sweet spot TBD.
WORD_LIST_LENGTH = int(os.environ.get("PEARL_WORD_LIST", "1400"))

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


class MiningWorker(threading.Thread):
    def __init__(self, wid):
        super().__init__(daemon=True)
        self.wid = wid
        self.count = 0
        self.running = True

    def run(self):
        print(f"[W{self.wid}] Started", flush=True)
        while self.running:
            try:
                r = req.post(
                    VLLM_URL,
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": build_prompt()}],
                        "max_tokens": MAX_TOKENS,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == 200:
                    self.count += 1
                    if self.count % 50 == 0:
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


def stats(workers):
    """Print stats every 30 seconds."""
    while True:
        time.sleep(30)
        total = sum(w.count for w in workers)
        active = sum(1 for w in workers if w.is_alive())
        print(
            f"[Stats] total_requests={total} active_workers={active}/{NUM_WORKERS}",
            flush=True,
        )


def main():
    print(f"🐚 Pearl Mining Worker — {NUM_WORKERS} threads, max_tokens={MAX_TOKENS}", flush=True)
    print(f"   Target: {VLLM_URL}", flush=True)

    workers = [MiningWorker(i) for i in range(NUM_WORKERS)]

    def shutdown(s, f):
        print("\n🛑 Shutting down workers...", flush=True)
        for w in workers:
            w.running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for w in workers:
        w.start()

    threading.Thread(target=stats, args=(workers,), daemon=True).start()

    # Keep main thread alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
