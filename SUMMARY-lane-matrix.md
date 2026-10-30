# Morning summary — LLM host-language lane matrix (night 2026-10-27→28, review round 2)

Session: openfox-e9e19426 · machine bda452aabd60bf26 (M5 Max 128 GB, macOS 27.0.1/26A434)
Ledger: `results/2026-10-28T07-58-42Z-bda452aabd60bf26.json` (schema silicon-ledger/1;
supersedes the 04-07-19Z row file; 67 runs = 19 decode cells + 48 prefill cells)
Lanes: L0 python mlx_lm full loop · L0m python microbench mirror of C graph ·
L1 swift mlx-swift-lm full loop · L2 swift FoundationModels session ·
L3 C mlx-c hand-rolled decode loop · L4 Rust hand-rolled loop, **binds+links
the same mlx.h/Cmlx.o as L1/L3** (post-review, LF-11) ·
L5 ObjC rehost of L3 (same source, `-x objective-c`).
All numbers: greedy, 1 warmup + measured runs, warm process, serialized GPU queue.

## Decode tok/s (wall, per-token step; same-session paired where noted)

| cell | 1.7B | 4B | 30B-A3B |
|---|---|---|---|
| L1 Swift full loop | **348.3** (2.87) | **179.5** (5.57) | **133.5** (7.49) |
| L0m Python mirror | 305.7 (3.27) | 159.7 (6.26) | 135.1 (7.40) |
| L5 ObjC (576-step) | 156.2 (6.40) | — | — |
| L0 Python full loop | 188.9 (5.29) | 118.2 (8.46) | 114.9 (8.70) |
| L3 C | 154.7 (6.46) | 96.1 (10.41) | 47.2* (21.2) |
| L4 Rust (same engine) | 143.0 (6.99) | 92.4* (10.82) | 61.3* (16.3) |
| L2 FoundationModels | 17–34 (session-capped, not comparable) | 23–29 | 9–27 |

Values in the 1.7B/4B L1+L0m rows are SAME-SESSION paired A/B (queue11, two
rounds, reproduced to 0.4%): +14% Swift @1.7B, +11% @4B. Starred cells are
salvage runs (128–576 steps) — native MoE/4B hit the Metal allocator resource
wall (499000 live buffers) mid-run at full length (FAILURES.md LF-06).
Host-environment drift across the night was up to 2× (Swift 1.7B 182–355
t/s on identical bits), which is why ranking cells use paired runs; rows from
different time windows are not directly comparable.

## What the night actually found

1. **Swift is the answer at dense, and it isn't close; at 30B it is the
   safe engine, not the fastest t/s.** Paired decode: 348.3 vs 305.7 (+14%)
   @1.7B, 179.5 vs 159.7 (+11%) @4B over the identical Python graph; Swift
   prefill 14.8k t/s @2048 beats the Python mirror's 14.1k. At 30B MoE Swift
   (133.5) and Python (135.1) are within noise, but Swift completes 768
   steps while hand-rolled C/Rust die at ~300–550 — the product-grade lane.
2. **Hand-rolled native loops are now genuinely apples-to-apples — and still
   lose to Python.** Post-relink, Rust matches C within ~7% (143.0 vs 154.7
   @1.7B), confirming both the C2 fix and that the gap vs the same graph in
   Python (305.7) is per-step host/allocator machinery, not language speed.
   ObjC wrapper cost over C is +1% (156.2 vs 154.7, same-window rebuild) —
   the morning table's "ObjC +70% over C" was pure environment skew: a
   caution against reading language deltas across queues.
3. **The MoE wall for native loops is allocator resources, not FLOPS**
   (LF-06): C/Rust dense finish 768 steps; MoE dies mid-run
   (`metal::malloc Resource limit (499000)`) even with per-50-step
   `mlx_clear_cache()`. Follow-ups: fixed-size expert-window workspaces.
4. **Method traps paid for themselves** (FAILURES.md LF-01…LF-13): the
   Python-wheel metallib is ~4× slower for the C host (LF-02); the Rust lane
   had silently linked a DIFFERENT engine core (mlx-sys vendored 0.32.2) —
   relinking the Swift Cmlx.o changed Rust 1.7B short-run decode 141.9→258.5
   t/s, i.e. version strings are not provenance (LF-11); uninitialized
   `mlx_array` out-params segfault only under -O2 (LF-12); FoundationModels
   intermittently emits empty completions and needed retry-on-empty to
   produce valid 4B/30B cells (LF-13).
5. **Prefill cells are complete**: prompt_tokens exactly 32/512/2048 for
   L0/L0m/L1/L3/L4/L5 (L1 calibrated against measured `promptTokenCount`,
   not tokenizer-estimate). C/Rust/L0m measure prefill after the decode
   session (identical order across those lanes; pessimistic for native hosts
   — allocator pressure after 768 steps: C 4B 2048 → 553 t/s vs 9.9k fresh).
   Highlights: L0m 2048 14.1k, L1 2048 14.8k, L0 11.5k; MoE gather-prefill
   stays ~4.2k (Swift) vs ~0.2–0.9k (naive per-token expert gather, C/Rust).

## Verdict

The lane choice product this night: **Swift (mlx-swift(-lm)) is the viable
custom-loop language** — fastest dense loop by paired A/B (+11–14% over its
own graph in Python), MoE-safe at full length, tokenizer/template/
detokenizer native, and its Cmlx.o is the one engine binary every other
native lane now links to be measured honestly. Python remains the fair
experimental baseline. C/ObjC/Rust are reachable (C3 matrix passed;
gated by allocator/session machinery, ~2× off Python at decode), worth one
follow-up night after an LF-06 fix. C2's guarantee (one engine across lanes)
now holds by construction and re-measurement, not by claim.

## Repro

Queues (serialized; logs under `logs/`): `night_queue{3,4,5}.sh` (first
matrix), `night_queue6.sh` (ObjC), `night_queue{8,9}.sh` (post-review Rust
relink + prefill arms), `lanes/swift/queue11.sh` (paired fairness A/B),
`night_queue10.sh` (L2 retry + prompt-truth S1). Assemble:
`python3 tools/assemble_lane_matrix.py`. C links Swift's Cmlx.o; Rust binds
the same 0.32.3 mlx.h (build.rs) and links the same object; C/Rust greedy
token streams agree on dense; first-mismatch-at-tie parity history in LF-11.
