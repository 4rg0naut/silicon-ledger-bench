# silicon-ledger/1 — canonical record schema (unified bench, agreed 2026-09-26)

One JSON document per run. Top level:

```json
{
  "schema":    "silicon-ledger/1",
  "generated": "2026-09-26T03:22:23Z",            // ISO-8601 UTC
  "machine":   { ...MachineInfo... },
  "sections":  [ ...SuiteSection... ],
  "status":    { "s1_memory": "done", "spike": "GO" }   // per-suite state:
                                                        // done | ERROR <msg>
                                                        // | skipped (<reason>) | GO | NO-GO
}
```

File name: `<generated, ":"->"-">-<machine.fingerprint>.json` in `results/`.

## machine (MachineInfo)

| field | type | notes |
|---|---|---|
| `fingerprint` | string | 16-hex; THE diff key |
| `host` | string | `sysctl`/ProcessInfo hostName |
| `model` | string | e.g. "Mac Studio" / "Mac mini" |
| `chip` | string | marketing name, e.g. "Apple M5 Max" |
| `osVersion` | string | e.g. "27.0" |
| `osBuild` | string | e.g. "26A428" |
| `memoryGB` | int | |
| `cpuCores` | {total, perflevel0, perflevel1: int} | |
| `gpuName` | string | MTLCreateSystemDefaultDevice().name |
| `gpuCores` | int | |
| `metalFamilies` | [string] | supported MTLGPUFamily names, e.g. ["apple7",...,"apple10"] |
| `metalTensor` | {dtype: bool} | fp32/fp16/bf16/int8/int4/fp8e4m3/fp8e5m2/fp4e2m1 support |
| `ane` | {capability: bool} | framework_loadable / inmemory_classes / request_class / ane_compiler_xpc / aned_daemon / coreai_framework / coreml_mlmodel |
| `tools` | {tool: version} | swift/clang/coremltools/... |

**ANE chip id**: M4 session adds `chipId` here — use the framework's ANE class
name/subtype string (`h16g`, `h17`) as `machine.chipId` (string). Consumers
match cross-machine on `fingerprint`, read `chipId` for interpretation.

## sections[] (SuiteSection)

```json
{ "id": "s2_gemm", "title": "human readable", "runs": [ ...RunResult... ],
  "notes": ["free-form strings, one line each"] }
```

`id` ∈ { s0 handled by machine block, s1_memory, s2_gemm, s3_ane_raw,
s4_coreai, s5_coreml, s6_* (external, e.g. "s6_ane_dma" for the
budget/notch arms) }.

## runs[] (RunResult) — one arm

```json
{
  "name": "gemm_512_fp16",          // stable arm key
  "variant": "tensorops",           // path/config axis (mps|msl|tensorops|mps_fp32...)
  "config": { "n": "512" },         // strings; full arm description
  "unit": "ms",                     // ms | ms/run | ms/eval | GB/s
  "iters": 100,
  "median": 0.4722, "mean": 0.51, "p95": 0.61, "min": 0.44, "max": 1.2,  // ALWAYS ms
  "derived": { "GFLOPS": 9.1, "max_abs_err": 1.1e-5 },
  "status": "ok",                   // see taxonomy; missing on old files == "ok"
  "error": "max_abs_err=19.4 rel_err=1.0 tol=1e-3"   // only when status != ok
}
```

Rules (the convergence-critical ones):

1. `median/mean/p95/min/max` are **always milliseconds**, regardless of `unit`;
   human-facing quantities (GB/s, GFLOPS, TOPS) go in `derived` as doubles.
   Consumers that display GB/s read `derived` (or recompute from the byte
   count in `config`), never the ms median. Delta% is computed per the
   displayed quantity.
2. **Timing is within-run**: stats over the timed iterations of one process
   (warmup excluded by construction: `warmup` + timed loop, time-based
   warmups where the platform needs them). No cross-run means.
3. Every arm has exactly one row, success or failure. Failures are NOT
   omitted and NOT only in `notes`.
4. Row status taxonomy (`status`; `error` required when not "ok"; timing
   fields 0 and `iters: 0` on failed rows):

| status | meaning |
|---|---|
| `ok` | ran and validated (default; also the implicit value for pre-status files) |
| `compile_failed` | could not build/encode/compile the program — nothing ran |
| `validation_failed` | it ran and returned, but output failed the numeric gate |
| `skipped` | deliberately not attempted (env/capability gated); reason in `error` |

   Notes may still carry the detail; the row is the record of truth.
5. `derived` keys are conventional per suite, open-ended: `GFLOPS`, `TOPS`,
   `GBs`, `p95_GBs`, `max_abs_err`, `rel_err`, `cosine_sim`, `pointwise_rel_err`,
   `bytes_per_core`, `mib_per_core`, `compile_ms`, `specialize_ms`,
   `observed`, `expected`.

## Correctness probe (required for a row to claim `ok` on generated models)

All-ones weights, all-ones input, one 1x1 conv: output must equal `in_dim`
exactly (fp16-representable for Cin <= 2048). ~20 lines; catches
locatable-wrong BLOBFILE offsets that compile and return plausible garbage.
SpikeA on M5 Max runs the equivalent gate (deterministic weights, CPU fp32
reference within fp16 tolerance); external s6 sections should ship their own
probe arm with `expected`/`observed` in `derived`.

## Diff semantics

`bench diff A.json B.json`: for each section present in both, join rows on
(name, variant); print A/B medians (derived GBs for GB/s units), delta% of
the displayed quantity; rows present in only one side listed; non-`ok` rows
print their status with delta `n/a`. Machine fingerprint mismatch prints a
warning.

## DMA-notch cross-check (RESOLVED on H17 — 2026-09-26)

The M4 measured shared-blob-vs-per-file at equal conv count: 0.175 → 0.111 ms
(1.58x, non-overlapping p95s). H17 re-run (`bench ane --subset blob_probe`
/ `blob_ab`, arms `blob_*/shared` vs `blob_*/per_file`): single-entry dicts
under **any** file name compile and produce byte-identical output
(`blob_probe_d1`: shared 0.116 ms / per_file 0.105 ms), but **2+ entries
fail ANECCompile with InvalidMILProgram** on the in-memory
`_ANEInMemoryModelDescriptor` path — 16 per-file names, correct headers and
offsets, kept-bundle inspection shows the framework materializes exactly one
weights region. Conclusion: the M4 1.58x penalty is path-specific (its
arms used the on-disk bundle / offline-compile route); on H17's in-memory
route the per-file A/B is not expressible, and the shared-blob arm remains
the canonical number (d16 0.141 ms / 3.81 TOPS, d128 0.554 ms / 7.75 TOPS,
per_file rows report status `compile_failed`). Details: research-notes
R11 + R12.
