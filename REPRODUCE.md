# REPRODUCE — claim-by-claim reproduction guide

Target hardware/OS for every expected number: **Mac Studio, Apple M5 Max,
40-core GPU, 18 CPU cores, 128 GB unified memory, macOS 27.0 (build 26A428),
Xcode 27 toolchain**. Machine fingerprint (cross-diff key): `bda452aabd60bf26`
(`bench probe` prints it; it is a SHA-256 prefix over hardware identity —
different chips WILL show different fingerprint and different expected bands).

Reference record: `examples/2026-09-26T02-14-33Z-bda452aabd60bf26.json`
(schema `silicon-ledger/1`, `machine.host` redacted).

## How to reproduce the whole file

```
make build
make all            # ~8 min, needs an unloaded machine
```

Then compare:

```
.build/release/bench diff examples/2026-09-26T02-14-33Z-bda452aabd60bf26.json results/<your-new-run>.json
```

Expected: `bda452aabd60bf26` vs your fingerprint; deltas within the per-row
tolerances below for GPU rows. CPU write/triad deltas are band-dependent (see
traps): the S1 primer guarantees GPU rows are steady-state, but CPU write
placement is external-memory-pressure dependent.

## Pinning (v0 caveat)

This file's `pin` is `file:` — i.e. the commands/expectations below are pinned
to the working-tree copy of this file, **not** to a commit-pinned digest.
Reason: the record's `source_digest` is stored *inside* the record, so a
self-referential digest cannot be committed in the same commit (circular).
First tagged release switches pins to `source_sha:<commit> + path`, which a
tool verifies as `git show <sha>:REPRODUCE.md | sha256`. Until then treat pins
on these rows as `known_trap: self-reference-unsolved-v0`.

## Claim-by-claim

All rows: run from repo root, after `make build`. "Expected" = example-run
value; "Tolerance" = pass criterion applied to your median.

### S1 memory (unit GB/s, 512 MB CPU / 256 MB GPU, steady state after primer)

| Claim | Command | Expected | Tolerance / notes |
|---|---|---|---|
| gpu_read 561.6 | `bench memory` (name `gpu_read`) | 561.6 | ±5%; ≈ packaged LPDDR peak |
| gpu_triad 555.1 | same (`gpu_triad`) | 555.1 | ±5% |
| gpu_copy 380.5 | same (`gpu_copy`) | 380.5 | ±5%; if <340 the primer did not recover — rerun warm (trap R9) |
| gpu_write 196.0 | same (`gpu_write`) | 196.0 | ±10%; placement-sensitive |
| cpu_read 273.5 | same (`cpu_read`) | 273.5 | ±5% |
| cpu_write 101.4 | same (`cpu_write`) | 101-243 | band bimodal: ~100 under external resident memory pressure, ~235 back-to-back warm (R9) |
| cpu_triad 203.7 | same (`cpu_triad`) | 145-315 | same two bands |
| tensor_copy 42.6 | same (`tensor_copy`) | 42.6 | ±20%; MTLTensor copy path is far from buffer path — that gap IS the finding |

### S2 GEMM (unit GFLOPS after validation)

| Claim | Command | Expected | Tolerance |
|---|---|---|---|
| gemm_8192_fp16 mps 60750 | `bench gemm` | 60750 ±5% (median ≥5 ms regime) | n≥2048 fp16 rows are clock-steady |
| gemm_8192_fp32 mps 14619 | `bench gemm` | ±5% | |
| gemm_8192_bf16 msl 12161 | `bench gemm` | ±8% | max-power SIMT band |
| gemm_8192_int8 mps | `bench gemm` | ≈7000-7700 | ±8% |
| tensorops rows | `bench gemm` | excluded (zeros) | **trap R8** — on macOS 27.0 beta GPUCompiler 32023 multi-tile `matmul2d` silently returns zeros; rows appear in `notes` as VALIDATION FAILED, single-tile probe passes |
| fp8/fp4 mps rows | `bench gemm` | skipped | MPS asserts (not NSError) on fp8/fp4 dtypes; filtered by design |

### S3 raw ANE (unit ms/eval, private API)

| Claim | Command | Expected | Tolerance |
|---|---|---|---|
| dispatch_floor 0.119 | `bench spike && bench ane` | 0.116-0.133 | ±15%; this is per-eval XPC+dispatch overhead, not compute |
| peak_int8_512x64_d128 13.1 TOPS | `bench ane` | 13.1 ±5% | int8 weights, fp16 activations (W8A16, not W8A8) |
| peak_256x64_d256 fp16 6.9 TOPS | `bench ane` | 6.9 ±5% | |
| sram sweep 2.6→168 MB | `bench ane` | cliff between 42 MB (100 GB/s weights) and 94 MB | working-set dependent, qualitative |
| scale_4096 26.8 ms/eval | `bench ane` | ±10% | single-op regime; compute-dominated |

Requires: `ANE reachability` flags all true in `bench probe` output. If the
spike reports NO-GO (OS update moved private symbols) the S3 rows are not
reproducible on that OS build — **known_trap: version-fragile by design**.
Compile ceiling: ~119 in-memory compiles per process — do not run `bench ane`
twice in one process. **R12**: the in-memory descriptor dict accepts exactly
ONE weights entry; multi-entry dicts fail ANECCompile with InvalidMILProgram
(per-file blob injection of many tensors is NOT available on this path).

### S4 Core AI (unit TOPS, official stack)

| Claim | Command | Expected | Tolerance |
|---|---|---|---|
| coreai_matmul 88.1 | `make coreai` (auto-generates models, ~1 min first time) | 75-95 | ±10%; inter-process DVFS spread |
| coreai_deep_fp16 ~2.0 | same | 1.8-2.1 | **trap R10**: ANECCompile internal error → runtime silently falls back to GPU; all three variants land in the same band — that IS the finding |
| coreai_deep_int8 ~0.8 | same | 0.6-0.8 | same fallback; W8A8 validity confirmed by cosine ≥0.999 vs torch ref |

### S5 Core ML (optional)

| Claim | Command | Expected | Tolerance |
|---|---|---|---|
| coreml_conv_512x64_d8 ≈1.4 TOPS @0.2 ms | `make coreml` | ±20% | needs coremltools 9.0 in `tools/.venv` (`make venv`) |

## Cross-cutting traps (read before blaming the hardware)

1. **R8 — matmul2d silent zeros**: only single 64x32 tile / static K=64 is
   correct on the 27.0 beta GPUCompiler; every multi-tile row must be read as
   validation_failed, never as hardware slowness.
2. **R9 — primer/power-allocation**: never benchmark cold from deep idle. A
   process launched cold stays in a degraded state for its whole lifetime and
   anti-correlates CPU vs GPU by up to 2x. Long cooldowns between runs HURT
   under memory pressure (degraded page placement). Run warm, back-to-back.
3. **R10 — silent GPU fallback**: Core AI deep graphs do NOT prove ANE
   execution on this build. Fallback detection is via the equal-band across
   `default/ane-preferred/gpu-preferred` variants.
4. **R12 — single weights entry**: multi-blob in-memory weight injection is
   rejected by ANECCompile; only one weights region per descriptor.
5. **Dispatch floor ≈ 0.12 ms**: any per-call ANE latency below ~0.116 ms is
   not reachable via the in-memory-MIL route on this machine regardless of
   model size; report sub-0.1 ms/eval numbers from other routes with a route
   comparison (e.g. ANEForge's e5rt path claims ~70 µs on M4-class).
6. **119-compiles/session**: in-memory compiles are session-capped; suite
   design reuses one program per config. Re-running `bench ane` inside one
   process past the cap yields spurious compile failures.

## Validation contract

No reported number was eyeballed: every throughput row was produced only if a
CPU/torch reference validated it first (`rel_err`/`cosine` fields alongside).
Failed validation excludes the row and writes a note into the section — grep
`notes` in the report JSON for the full audit trail. Machine-verified only:
no human review has been applied to any row (by design; see README Provenance).
