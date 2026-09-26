# silicon-ledger-bench

Low-level, burst-mode benchmark suite for Apple silicon local AI hardware.
It is the **instrument** of the [silicon-ledger](https://github.com/4rg0naut/silicon-ledger)
project: every run emits a machine-verifiable measurement record (schema
`silicon-ledger/1`) that the ledger consumes as a first-class claim. Built and
calibrated on a **Mac Studio (M5 Max, 40-core GPU, 128 GB, macOS 27.0)** —
other Apple silicon can run it; expected bands and known traps are documented
in [REPRODUCE.md](REPRODUCE.md).

It measures:

- **ANE** (Apple Neural Engine) — via the reverse-engineered private API
  (`_ANEInMemoryModel`) and via the official **Core AI** `.aimodel` stack.
- **GPU NE accelerators** — Metal 4 TensorOps (`tensor_handle` matmul), plus
  MPS and custom `simdgroup_matrix` MSL.
- **Memory bandwidth** — CPU (Accelerate/vDSP) and GPU (Metal buffer + MTLTensor copy).

Burst mode only: no thermal soak, no fan control, no sudo.

> **Verification status:** every number below was validated against a CPU
> reference inside the harness and reproduced by machine across repeated runs.
> **No human has signed any row yet** — that is by design; the ledger's review
> facet accepts only separately-stamped human events.

## TL;DR (from the committed [example run](examples/2026-09-26T02-14-33Z-bda452aabd60bf26.json), M5 Max / macOS 27.0)

| Measurement | Value |
|---|---|
| GPU read (256 MB, Metal timestamps) | **561.6 GB/s** |
| GPU triad | **555.1 GB/s** (≈ packaged peak) |
| GPU copy | **380.5 GB/s** |
| CPU read (vDSP, 18 cores) | **273.5 GB/s** |
| GEMM fp16 8192² (MPS) | **60.7 TFLOPS** |
| Core AI single 2048² matmul | **88.1 TOPS** |
| ANE raw peak int8 W8A16 (1x1-conv chain d128) | **13.1 TOPS** |
| ANE raw peak fp16 | **6.9 TOPS** |
| ANE dispatch floor (single small conv eval) | **0.119 ms** |
| Unattended `bench all` wall time | **~8 min** |

## Quickstart

```
git clone https://github.com/4rg0naut/silicon-ledger-bench
cd silicon-ledger-bench
make build
.build/release/bench probe        # S0 fingerprint
make all                          # full suite -> results/<timestamp>-<fingerprint>.json
```

**Requires Xcode 27.** Command-line tools alone are *not* enough: the macOS 27
SDK, `CoreAI.framework`, `MTLTensor` and the Metal 4 APIs come from the Xcode
SDK. The Makefile invokes the Xcode-default toolchain's `swift` directly with
`SDKROOT` set to the Xcode SDK — deliberately avoiding `DEVELOPER_DIR` (which
triggers the Xcode license gate; no sudo is used or required) and
`xcode-select`, which on the calibration machine points at CommandLineTools.

**Version-fragile by nature.** Everything ANE-adjacent leans on private,
undocumented APIs that Apple can change or remove in *any* OS update. A NO-GO
spike result after an OS bump is expected behavior, not a broken benchmark —
see [macOS 27 caveats](#macos-27-caveats-version-fragile).

For the Python model generators (S4/S5):

```
make venv             # tools/.venv from tools/requirements.txt (needs python3.11)
```

Known environment quirk: `swift-tools-version` must be `6.4` so the manifest
can declare `platforms: [.macOS(.v27)]`.

## Layout

```
Sources/
  ANEBridge/       C bridge to ANE private APIs (runtime-resolved, no link-time dep)
  bench/           Swift package target (CLI: `bench`)
    Fingerprint.swift   S0 machine fingerprint (JSON key for cross-machine diffing)
    S1_Memory.swift     S1 CPU/GPU/tensor memory bandwidth
    S2_GEMM.swift       S2 GEMM: MPS / custom MSL / TensorOps, 6 dtypes
    SpikeA.swift        S-A ANE direct-path feasibility spike (GO/NO-GO gate)
    S3_ANEDirect.swift  S3 raw ANE throughput via _ANEInMemoryModel
    S4_CoreAI.swift     S4 Core AI .aimodel inference (official stack)
    S5_CoreML.swift     S5 legacy Core ML baseline (optional, --include-coreml)
tools/
  gen_models.py    generates models/coreai/*.aimodel via coreai-torch + coreai-opt
  gen_coreml.py    generates models/coreml/*.mlpackage via coremltools (S5)
  .venv/           python3.11 venv (torch, coreai-torch, coreai-opt; gitignored)
reference/
  maderix-ane/     MIT-licensed source of the ANE bridge (attribution)
examples/          one curated full-run report (host redacted), referenced by TL;DR
results/           report JSON + ane_spike.json (gitignored)
models/            generated model assets (gitignored)
```

## Run

```
make probe            # S0 fingerprint
make memory           # S1
make gemm             # S2 (default: all sizes/dtypes/paths)
make spike            # S-A ANE spike -> results/ane_spike.json (gates S3)
make ane              # S3 raw ANE
make coreai           # S4 (auto-runs tools/gen_models.py on first use)
make coreml           # S5 (optional)
make all              # probe + spike + s1 + s2 + s3(if GO) + s4 (+s5 w/ flag)
make stability        # S1 3-run stability series (gated + cooldowns, ~25-45min)

.build/release/bench diff results/a.json results/b.json   # cross-machine delta table
.build/release/bench diff examples/2026-09-26T02-14-33Z-bda452aabd60bf26.json results/<new>.json
```

Run the binary from the repo root: `models/` and `results/` are resolved
relative to the current working directory (fine for a local bench tool,
but a `cd`-away will look like missing models / an empty results dir).

Every run writes `results/<timestamp>-<fingerprint>.json` (schema
`silicon-ledger/1`) containing the machine fingerprint, per-run
median/mean/p95/min/max, and derived GFLOPS/TOPS/GB/s. Reports from the
pre-rename era carry schema id `m5max-bench/1`; the fields are identical and
still load (with a note on stderr).

An unattended `bench all` completes in ~8 minutes on the calibration machine
(excluding the one-time ~minute model generation by `tools/gen_models.py`).

## Cross-repo contract

This producer emits `silicon-ledger/1` records; the ledger attaches a review
facet keyed by producer. Join keys: **schema `silicon-ledger/1` · `_producer`
= bare repo name (`silicon-ledger-bench`) · facet key = producer name · facet
schema at `schemas/facets/<producer>/1.json` in the producer's own repo ·
machines may stamp `model_check` only · unverifiable evidence is declared,
never asserted** (`evidence_unlocatable` there, `evidence_stale` here).

## Methodology

- Timing: `mach_absolute_time()` scaled by `mach_timebase_info`, per-iteration
  samples, **median** reported (mean/p95/min/max alongside). Warmup iterations
  precede the timed loop (2-5 warmup, 15-200 timed iters per subtest).
- Specialization/AOT times (Core AI `AIModel.specialize`) are measured and reported
  separately, excluded from inference timing.
- GEMM/ANE results are validated against an Accelerate (cBLAS) or CPU fp32
  reference before timing; failing runs are excluded and noted.
- ANE direct path reuses one compiled program per config (in-memory compiles are
  limited to ~119 per process).

## Suites

### S0 — probe
Machine fingerprint: chip, GPU core count, Metal families, MTLTensor dtype support
(fp32/fp16/bf16/int8/int4/fp8e4m3/fp8e5m2/fp4e2m1), ANE reachability flags
(framework dlopen, in-memory classes, request class, aned daemon,
ANECompilerService XPC, Core AI framework), unified memory size, tool versions.
The 16-hex-char SHA-256 prefix is the cross-machine diff key.

### S1 — memory
CPU: 512 MB vDSP read/write/triad, one chunk per physical core
(`DispatchQueue.concurrentPerform`), buffers 2 MB-aligned (`posix_memalign`)
with first-touch spread across all cores. Timing: warm 1.5 s, then three
independent 3 s windows; the per-window medians are reported (a background
daemon can contaminate at most one window; the cross-window median lands in
the stable band). A pre-flight load check aborts the suite when the SoC is
already busy so daemon interference can never silently skew a run. GPU:
256 MB shared-storage Metal buffer read/write/copy/triad kernels, float4
grid-stride reads, timed with Metal GPU timestamps (`gpuEndTime-gpuStartTime`,
8 dispatches per command buffer) so CPU commit/wait overhead — which
dominates single 0.5 ms dispatches — is excluded. Tensor: MTLTensor 256 MB
fp32 copy via an MTL4 encoder (`copy(sourceTensor:...)`). Before any
subtest, a self-calibrating mixed CPU+GPU primer runs (20 s minimum, 120 s
cap) until measured GPU-copy bandwidth plateaus: a process launched after
idle starts in a cold power-allocation state that persists for the whole
process and anti-correlates CPU vs GPU results by up to 2x, and the
recovery time out of deep idle is stochastic, so the primer converges on
measured bandwidth rather than a fixed duration. GPU+CPU share the unified
memory pool, so GPU read+write ≈ CPU numbers is the sanity check. Expected
steady state on an unloaded-memory machine: GPU read/triad ≈ 550 GB/s
(packaged peak), copy ≈ 380 GB/s, CPU read ≈ 265 GB/s, CPU write
≈ 230-240 GB/s. Under heavy external resident memory (low band) CPU write
settles at ≈ 100 GB/s and triad ≈ 145; GPU rows are unchanged. Stability
protocol: `tools/stability_run.sh <prefix>` collects 3 GOOD runs — a run
counts good when the GPU primer recovered sustained copy (>=340 GB/s) and
the CPU write is in a consistent band — at short (30-120 s) cooldowns.
Long idle cooldowns HURT: under sustained external resident-memory
pressure, freshly allocated pages routed into degraded physical placement
after idle (CPU write 101 GB/s plateau), while runs issued back-to-back
keep healthy placement (CPU write ~235 GB/s); see R9. Gate:
`tools/check_stability.py --paths-file <good.txt> --suite=s1` (<10% per
subtest). See docs/research-notes.md R9 for the full measurement study
behind these design choices.

### S2 — GEMM (GPU)
Sizes 512-8192, dtypes fp32/fp16/bf16/int8(fp32 accum via int32)/fp8 e4m3/
fp8 e5m2/fp4 e2m1.
Paths: `MPSMatrixMultiplication` (reference), custom 32x32-tile MSL
(`simdgroup_matrix`, 4 simdgroups; fp16/fp8/fp4 additionally use a 16x128
float-accumulate SIMT tile where `simdgroup_matrix` precision or int types fall
short), Metal 4 TensorOps `mpp::tensor_ops::matmul2d` (64x32 tiles,
`execution_simdgroups<4>`) which uses the GPU neural accelerators.
Validated vs cBLAS `sgemm` (relative error: fp32 1e-3, fp16 1e-2, int8 exact,
fp8/fp4 1e-2). Timing uses a 2.5s sustained-load primer plus >=1s of
per-kernel time-based warmup: back-to-back small-n measurements otherwise
drift up to 2x between processes as the GPU DVFS ramps between run states.

Stability protocol (regime-split, measured across 3-run studies):
**<5% 3-run median spread for configs with median >=5ms** (clock-steady);
**<8%** for the max-power SIMT rows (msl int8/fp16/fp8/fp4 at n>=4096, where
sustained-load throttling dominates); configs **<5ms** report spread as
informational (DVFS + MPS per-process autotune make <5% unattainable on
macOS 27 beta without a user-accessible clock lock; observed up to ~25%).
`tools/spread.py <a.json> <b.json> <c.json>` applies these rules. TensorOps results are three-state: **passed** (single-tile
capability probe), **validation_failed (excluded)** (multi-tile matmul2d
silently returns zeros on the macOS 27.0 beta toolchain — see
docs/research-notes.md R8; the row is *not* reported as skipped because the op
ran and produced wrong output), or **skipped** (dtype/path unsupported).

### S-A — ANE spike
Minimal program: dlopen `AppleNeuralEngine.framework`, resolve the private classes,
compile a trivial 1x1 conv (512ch, 64x64 spatial) MIL text via
`_ANEInMemoryModelDescriptor` + in-memory weight blob, run it, compare vs CPU fp32
within fp16 tolerance. Writes `results/ane_spike.json` with GO/NO-GO + ms.
**This gates S3.**

### S3 — raw ANE
MIL 1x1-conv chains through `_ANEInMemoryModel` (see [maderix/ANE](https://github.com/maderix/ANE),
MIT; see also arXiv:2606.22283 for the format background):
- dispatch floor (single small conv) — the per-eval overhead regime;
- single-op scaling 256->4096 — where compute starts dominating dispatch;
- deep-graph peak FP16 and INT8 W8A16 TOPS (int8 weights, fp16 activations;
  16-256 chained 1x1 convs, fp16 I/O) — genuine W8A8 (quantized activations)
  is measured in S4 via Core AI PTQ;
- SRAM working-set sweep 2.6 MB -> 168 MB to locate the in-chip memory cliff.

On this OS the in-memory path accepts **exactly one** weights-region entry in
the descriptor dict (multi-entry dicts fail compile — see
docs/research-notes.md R12), which bounds multi-shape model injection for now.

### S4 — Core AI (official)
Python generator (`tools/gen_models.py`, coreai-torch 0.4.3 + coreai-opt 0.2.1)
emits `.aimodel` assets with deterministic weights matching S3's chains:
- `deep_fp16`: 64x Conv2d(512->512, 1x1) — same graph family as S3;
- `deep_int8`: depth 8 (not 64): 64 chained quantized layers with random
  calibration weights decorrelate from the torch fake-quant reference
  (cosine -0.13); depth 8 keeps the same shapes while validating cleanly
  (W8A8 PTQ via `coreai_opt.quantization.Quantizer`, default int8 config,
  calibration on the deterministic input);
- `matmul`: single 2048x2048 GEMM.

The Swift runner loads with `AIModel.specialize` (AOT; time reported separately),
then times `InferenceFunction.run` across `default` / ANE-preferred /
GPU-preferred specialization variants. Outputs are validated against the torch
CPU reference stored next to each asset (`<name>.ref.bin`): pointwise rel for
fp16 (tol 0.1 / matmul 0.02), normalized rel + cosine for the quantized chain
(tol 0.15 / 0.999). TOPS are derived from the `flops` field in
`<name>.verify.json`. Note: on this OS the deep graphs hit an ANECCompile
internal error (MLIR MPS->ANEC conversion failed) and the runtime falls back
to GPU, which is why all three variants land at the same ~0.6-2.1 TOPS; the
single-GEMM graph reaches 75-95 TOPS.

### S5 — legacy Core ML (optional)
Same conv chain (512ch 1x1, depth 8) converted with **coremltools 9.0**
(`tools/gen_coreml.py`, torch.export + run_decompositions -> mlprogram,
fp16 compute precision incl. fp16 I/O). The `coremlc`-on-MIL-text path is
broken on this build (protobuf wireType 7 + unrecognized
`--minimum-deployment-target`), so the legacy baseline goes through
coremltools instead; `MLModel.compileModel(at:)` + `.cpuAndNeuralEngine`.
Validated vs fp64 torch reference (normalized rel 0.15 / cosine 0.999 —
fp16 end-to-end error is large pointwise at near-zero outputs).
Quantifies CoreML-vs-CoreAI-vs-raw overhead on this OS: ~1.4 TOPS at 0.2 ms/run,
vs 75-95 TOPS for the Core AI single GEMM. Off by default (`--include-coreml`).

## macOS 27 caveats (version-fragile!)

Everything ANE-adjacent here leans on **private, undocumented APIs** that Apple
can change or remove in any OS update:

- `AppleNeuralEngine.framework` is a **stub** on-disk (DSC in the OS cryptex is a
  753 KB stub); the real implementation is installed at runtime. The bridge
  dlopens it and resolves `_ANEInMemoryModel*` / `_ANERequest` /
  `_ANEIOSurfaceObject` at runtime — no compile-time dependency, so the binary
  survives even if symbols move (it just reports NO-GO).
- The in-process compile path used here (`_ANEInMemoryModelDescriptor` ->
  in-memory MIL) is one of at least two routes on 27; `ANECompilerService.xpc`
  (and `ANELargeModelCompilerService`) also exist as XPC alternatives.
- `eval` goes through `_ANERequest` against the `aned`/`aneuserd` daemons.
- In-memory program compilation is limited to ~119 per process.
- The in-memory descriptor dict accepts a single weights-region entry only (R12).
- E5/HWX kernel format cracking is deliberately **out of scope** (documented in
  maderix/ANE and arXiv:2606.22283).

Metal 4 / SDK notes discovered during this build (Xcode 27, macOS 27 SDK):

- `MTLCompileOptions.headerSearchPath` is **gone** — TensorOps kernels inline the
  MPS headers textually (`ShaderLib.tensorOpsSource()`).
- Legacy argument buffers (`MTLFunction.argumentBufferLayout`,
  `makeArgumentEncoder`) are removed; use `MTL4ArgumentTable` +
  `device.makeMTL4CommandQueue()` + `event.wait(untilSignaledValue:)` for
  CPU-side completion on M4 queues (no `waitUntilCompleted()` on M4 buffers).
- `_Float16` was removed from Swift — use `Float16`.
- `MTLBuffer.contents()` now returns a typed `UnsafeMutablePointer<UInt8>`.
- Core AI: `MLModel`-style APIs are async; the old synchronous
  `prediction(fromFeatures:)` is unavailable — use `prediction(from:)`.
- Swift concurrency (Swift 6 language mode is enforced: `@Sendable` closures
  cannot capture mutable state (the vDSP chunk loop uses raw pointers).
- `matmul2d` (Metal 4 tensor ops, user MSL via inlined MPS headers) compiles,
  dispatches, and passes argument validation but **silently writes zeros for
  every multi-threadgroup or multi-K-chunk dispatch** on the macOS 27.0 beta
  GPUCompiler (32023); only a single 64x32 tile with static K=64 is correct.
  `execution_simdgroups<1>` is broken even single-tile. See R8.
- `MPSMatrixMultiplication` **aborts the process** (assert, not an `NSError`)
  for fp8/fp4 `MPSDataType`s — these must be filtered before construction.
- M4 queues have no `waitUntilCompleted()`; with `signalEvent`/`wait`, never
  reuse an event value — a second `wait(untilSignaledValue: v)` on an already
  signaled value returns immediately and races the GPU (this fake-measured
  "hardware failures" during development).

## Related work

- [ANEForge](https://github.com/sbryngelson/ANEForge) (MIT; arXiv:2606.17090)
  — the closest prior system: reverse-engineers the ANE E5 kernel format and
  ships a Python/C++ compiler + roofline suite (pip `aneforge`). Their route
  dispatches a pre-compiled kernel per call (~70 µs floor on M4-class); ours
  compiles MIL in-process and dispatches through `aned` (~119-133 µs floor on
  this M5 Max). Same-chip A/B of the two routes is planned as a cross-check
  (`make floorcheck`, and a row contribution for their leaderboard). This repo
  is the ledger-side instrument; ANEForge is the compiler route.
- [maderix/ANE](https://github.com/maderix/ANE) (MIT) — reverse-engineering
  write-up and reference code for `_ANEInMemoryModel`; vendored under
  `reference/maderix-ane/` with attribution.
- arXiv:2606.22283 — ANE/MIL format background used to build the bridge.
- Rigel, arXiv:2606.12765 (MIT code) — Metal 4 tensor compute-path reverse
  engineering (`simdgroup_matrix` lowering, fp8 emulation semantics),
  cross-checked against our R8 findings.
- [Luvia-AB/openclaims](https://github.com/Luvia-AB/openclaims) — the
  span-rejection discipline adopted here: evidence that cannot be located is
  *declared* (`evidence_unlocatable`), never asserted. The openclaims-ai/openclaims
  claim standard (v0.1) is the conformance target for the v0.1.x claims tool.

## Provenance

This suite was developed **AI-assisted** (OpenFox agentic harness driving a
Claude-class model): code, measurement studies (docs/research-notes.md), and
docs were authored in-session and verified by machine on real hardware.
Commits touching AI-assisted work carry an `Assisted-by:` trailer. Conformance
to the openclaims v0.1 machine-checkable format is planned for v0.1.x. The
facet schema is committed at
`schemas/facets/silicon-ledger-bench/1.json` (field set disjoint from the
companion producer `silicon-ledger`; its URL resolves because this repo is
public, while the companion repo's facet URL 404s until that repo is
published — declared, not asserted). The remaining piece is the `claims`
tool that emits facet-annotated records.

## Attribution and license

- MIT — see [LICENSE](LICENSE).
- ANE bridge code adapted from [maderix/ANE](https://github.com/maderix/ANE)
  (MIT) — see `reference/maderix-ane/LICENSE` and the headers in
  `Sources/ANEBridge/`.
- MIL/ANE format background: arXiv:2606.22283.
- Metal 4 tensor compute-path reverse engineering (simdgroup_matrix lowering,
  fp8 emulation semantics): Rigel, arXiv:2606.12765 (MIT code).
- Metal Performance Shaders header text inlined for TensorOps is read at
  runtime from the **user's own** Xcode/macOS 27 SDK installation and inlined
  into the kernel source at compile time; no Apple header files are
  redistributed in this repository (Apple binary tooling may not be
  redistributed — only the fact that it was used is).
- The committed example report under `examples/` has its `machine.host` field
  redacted; hardware/toolchain fingerprint fields are left intact on purpose.
- Everything else here is original code for this benchmark.
