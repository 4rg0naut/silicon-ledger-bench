#!/usr/bin/env python3
"""L0 lane: Python mlx_lm full-loop generation benchmark.

Emits a JSON row (schema-compatible payload fragment) to stdout or --out.
Per C1: decode tok/s mean/p50/p95 (wall, per-step), prefill @32/512/2048,
peak mem, 1 warmup + 3 measured runs, greedy decoding, internal tps captured
alongside wall to expose the internal-vs-wall gap (issue #124 trap).
"""
import argparse
import json
import statistics
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper


def prefill_tokens(tokenizer, n_target):
    base = "The quick brown fox jumps over the lazy dog. " * 600
    ids = tokenizer.encode(base)[:n_target]
    return tokenizer.decode(ids)


def measure(tokens_per_call, fn):
    out = []
    for _ in range(tokens_per_call):
        out.append(fn())
    return out


def bench_model(model_id, max_tokens=128, runs=3, prefill_sizes=(32, 512, 2048),
                wired_bytes=None):
    if wired_bytes:
        mx.set_wired_limit(int(wired_bytes))
    t0 = time.time()
    model, tokenizer = load(model_id)
    load_s = time.time() - t0

    prompt = "What is the weather like today?"
    row = {
        "model": model_id,
        "max_tokens": max_tokens,
        "runs": runs,
        "load_seconds": round(load_s, 2),
        "decode": {"runs_wall_tps": [], "runs_internal_tps": [], "per_step_ms": []},
        "prefill": {},
    }

    for i in range(runs + 1):  # 1 warmup + 3 measured
        step_ms = []
        n_tok = 0
        t_start = time.time()
        prev = t_start
        gen = stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        final = None
        for g in gen:
            now = time.time()
            if final is not None:
                step_ms.append((now - prev) * 1000.0)
            prev = now
            n_tok += 1
            final = g
        wall = time.time() - t_start
        if i == 0:
            continue
        row["decode"]["runs_wall_tps"].append(round(n_tok / wall, 3))
        row["decode"]["runs_internal_tps"].append(round(getattr(final, "generation_tps", 0.0), 3))
        row["decode"]["per_step_ms"].append([round(x, 3) for x in step_ms])

    steps = [x for r in row["decode"]["per_step_ms"] for x in r]
    steps_sorted = sorted(steps)
    row["decode"]["mean_step_ms"] = round(statistics.fmean(steps), 3)
    row["decode"]["p50_step_ms"] = round(steps_sorted[len(steps_sorted) // 2], 3)
    row["decode"]["p95_step_ms"] = round(steps_sorted[int(len(steps_sorted) * 0.95)], 3)
    row["decode"]["mean_decode_tps_wall"] = round(1000.0 / statistics.fmean(steps), 3)
    row["decode"]["mean_decode_tps_internal"] = round(
        statistics.fmean(row["decode"]["runs_internal_tps"]), 3
    )

    for n in prefill_sizes:
        pp = prefill_tokens(tokenizer, n)
        mx.clear_cache()
        # measured prefill: time to first token of a max_tokens=1 run
        tries = []
        for _ in range(4):
            t = time.time()
            first = None
            for g in stream_generate(model, tokenizer, prompt=pp, max_tokens=1):
                first = g
            dt = time.time() - t
            tries.append((first, dt))
        first, dt = tries[1]  # drop first (warm)
        ptok = getattr(first, "prompt_tokens", n)
        row["prefill"][str(n)] = {
            "prompt_tokens": ptok,
            "wall_tps": round(ptok / dt, 2),
            "internal_tps": round(getattr(first, "prompt_tps", 0.0), 2),
        }

    row["peak_memory_bytes"] = int(mx.get_peak_memory())
    row["peak_memory_GB"] = round(mx.get_peak_memory() / 1e9, 3)
    del model
    mx.clear_cache()
    mx.reset_peak_memory()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--wired-bytes", type=int, default=0)
    a = ap.parse_args()
    row = bench_model(a.model, a.max_tokens, a.runs, wired_bytes=a.wired_bytes or None)
    js = json.dumps(row, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(js)
    print(js)


if __name__ == "__main__":
    main()
