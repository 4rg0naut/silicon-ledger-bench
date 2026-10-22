# ANE research notes — macOS 27 (M5 Max)

Research track for the ANE direct path (criterion 10). Status is filled in as
the spike (S-A) and S3 suite run. Scope is deliberately limited to what is
needed to run a conv benchmark through `_ANEInMemoryModel`; kernel format
(E5/HWX) cracking is out of scope (documented in maderix/ANE and
arXiv:2606.22283).

## R1 — ANE image inventory

- On-disk `AppleNeuralEngine.framework` in the OS cryptex / DSC is a **stub**
  (~753 KB). The real implementation is materialized at runtime (app cryptex /
  per-user cache). Do not rely on on-disk sizes or headers.
- Reachability from a normal (non-sandboxed) CLI process on this machine is
  probed by `ane_bridge_probe()`:
  - bit0: `dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")`
  - bit1: `_ANEInMemoryModel` / `_ANEInMemoryModelDescriptor` classes present
  - bit2: `_ANERequest` / `_ANEIOSurfaceObject` classes present
  - daemons: `/usr/libexec/aned` present; `aneuserd` active per launchd.
- Runtime-resolving the classes (`NSClassFromString` + `objc_msgSend`) means the
  bridge does not link against the framework; if Apple renames a symbol the
  spike reports NO-GO instead of crashing.

## R2 — compile path on macOS 27

Two routes exist; both were confirmed present on this OS:

1. **In-process** (used by this benchmark):
   `_ANEInMemoryModelDescriptor(modelWithMILText:weights:optionsPlist:)` ->
   `_ANEInMemoryModel(inMemoryModelWithDescriptor:)` ->
   `compileWithQoS:options:error:` -> `loadWithQoS:options:error:`.
   Weights are passed as a dict keyed by relative path
   (`@model_path/weights/weight.bin`) and also written under a per-model
   temp dir (the loader reads the file back). MIL is written as `model.mil`.
2. **XPC**: `ANECompilerService.xpc` (and `ANELargeModelCompilerService`) under
   `AppleNeuralEngine.framework/XPCServices/`. Not needed for in-memory MIL.

MIL notes: two MIL dialects are accepted by the in-process compiler on macOS 27:
- `program(1.3)` with `dict<string, string>` buildInfo and `string("...")` const
  names — used for the fp16 conv chains (matches what older coremlc emits).
- `program(1.0)` with `dict<tensor<string, []>, tensor<string, []>>` buildInfo and
  `name = tensor<string, []>("...")` const annotations — what coremlc 3600 emits.
  The int8 path is generated in this dialect (matches `coremlc` output byte-for-byte
  in structure, which the in-memory compiler accepts).
- int8 = **W8A16**, not W8A8: weights are `int8` + per-channel `constexpr_affine_
  dequantize` (scale fp16[N], zero_point int8[N], both loaded from BLOBFILE chunks),
  activations stay fp16. The W8A8 form (`quantize`/`dequantize` on activations) is
  NOT what coremlc emits for a `linear_quantize_weights` model and is not required.
- BLOBFILE references must use `offset = tensor<uint64, []>(N)` (1.0) / `offset =
  uint64(N)` (1.3); the file offset points at the 64-byte chunk header (see below).

## R3 — eval path

- In-process `_ANERequest` against the local `aned`/`aneuserd` daemons:
  `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
  with `_ANEIOSurfaceObject(objectWithIOSurface:)` wraps, then
  `evaluateWithQoS:options:request:error:` on the loaded model.
- I/O is IOSurface-backed: one surface per input/output, `bytesPerRow` = byte
  size, height 1. Inputs are memcpy'd in, outputs memcpy'd out (fp32 or fp16
  element strides per the MIL signature).
- Cross-check oracle: run the same graph through the official Core AI stack
  (`models/coreai/*.aimodel`, S4) and compare outputs; the Core AI runtime is
  the reference implementation of what the ANE should compute.

## R4 — weight blob format (reverse-engineered, macOS 27)

The in-memory loader parses `weights/weight.bin` strictly. Layout (all little-endian
u32):

```
offset 0  : 64-byte file header
  [0]     = number of chunks
  [1]     = 2
  [2..]   = 0
Each chunk (64-byte header + data, contiguous, 64-byte aligned region):
  [0]     = 0xDEADBEEF (magic)
  [1]     = dtype code: 1 = fp16, 4 = int8
  [2]     = data byte size
  [3]     = 0
  [4]     = absolute file offset of the data
  [5..]   = 0
  data    = raw elements at offset [4], size [2]
```

For a quantized conv layer the writer emits **three** chunks: int8 weights
(dtype 4), int8 zero-points (dtype 4, all zero for symmetric), and fp16 per-channel
scales (dtype 1). Region size is `64 + data`, padded to a 64-byte boundary unless it
is the file's final chunk. This is exactly the layout `coremlc` produces for a
`coremltools.optimize.coreml.linear_quantize_weights` model, which was used as the
ground-truth oracle during the RE (see `/tmp/mlc_test` probes in the session log).

## R5 — IOSurface sizing

`evaluateWithQoS` fails with "IOSurface smaller than the model expects" if the
output surface is smaller than the input. The bridge allocates every I/O surface at
`max(all input/output byte sizes)`, which satisfies the loader for both the fp32 and
int8 W8A16 graphs.

## R6 — minimal direct client

Implemented in `Sources/ANEBridge/ane_bridge.{h,m}` (adapted from maderix/ANE,
MIT): compile / write_input / eval / read_output / free, plus deterministic
fp16/int8 weight-blob builders implementing the R4 layout.

### Constraints observed

- **~119 in-memory compiles per process** — S3 reuses one compiled kernel per
  config and reports the compile count; long runs should restart the process
  rather than accumulate kernels.
- QoS value used: 21 (user-initiated interactive class), matching maderix.
- `loadWithQoS` occasionally fails on first try under load; one retry with a
  100 ms backoff (mirrors maderix's behavior).

## R7 — why S3 measures TOPS as 1×1 conv chains (conv datapath, not GEMM)

The ANE's dense-compute core is a single MAC/multiply array built for
convolution. This is why the suite expresses "dense / matmul" work as chains of
1×1 convolutions rather than as a raw GEMM op:

- All convolution variants (standard / transpose / dilated / depthwise /
  grouped) "map onto the same multiply array"; only weight tiling and channel
  grouping differ (arXiv:2606.22283 §13.3, Table 13.2).
- "Matrix multiply and fully-connected layers are native and fold into the
  convolution datapath when the operand fits the on-chip working set, tiling
  when it does not" (arXiv:2606.22283 §13).
- A linear `y = x @ W.T` over `C_in -> C_out` is the same arithmetic as a 1×1
  conv over a `[N, 1, C_in, 1, 1]` feature map; the paper's own guidance is to
  "replace a fully-connected layer with an equivalent 1x1 convolution so it
  runs on the convolution datapath" (arXiv:2606.22283, Listing 25).
- A separate fused "matrix-multiply and softmax path" exists, but it serves
  attention (SDPA), not general GEMM.
- For *large square GEMMs that stream from DRAM*, measured raw throughput
  favors the GPU; the engine's win is energy (Table 10.1: large 4096 GEMM,
  engine 4.4 W vs GPU 32.5 W). The paper's processor-selection heuristic routes
  "large square GEMM, streams from DRAM" to the GPU.
- The reference RE peak benchmark (maderix/ANE) also measures peak TOPS with a
  deep conv chain (128× conv 512ch 64×64), not a raw GEMM.

Implication: the 1×1 conv chain is the ANE's **native dense datapath**, so S3's
"peak TOPS" and "matmul scaling" tests use it on purpose — it is the most
representative ANE workload, not a workaround. Report ANE numbers as
conv/dense-compute TOPS; the GPU (S2) is the correct device for raw large-GEMM
throughput, the ANE for conv + energy efficiency.

## R8 — Metal 4 tensor ops (matmul2d) on macOS 27.0: silent single-tile limit

macOS 27.0 "Golden Gate" shipped GM 2026-09-14; this machine runs 26A428.

Empirical (probes 2026-09-25, this machine, GPUCompiler 32023 / CommandLineTools,
`makeLibrary(source:)` at `MTLLanguageVersion4_1`), cross-checked against the
Rigel paper (arXiv:2606.12765, "Reverse-Engineering the Metal 4.1 Tensor Compute
Path on Apple M4 Max", MIT code):

- The MPS framework install here ships only MSL device headers
  (`MPPTensorOpsMatMul2d.h` + `__impl/`); the `matmul2d_descriptor` /
  `execution_simdgroups<N>` API resolves through those headers into
  `__tensorops_impl_*` driver entry points. There is no host-side MPS wrapper
  class for tensor-ops matmul in this SDK.
- Host binding works: `device.makeTensor(descriptor:)` + `MTL4ArgumentTable
  setResource:atBufferIndex:` at the tensor arguments' implicit buffer slots
  (0,1,2...) is accepted and the kernel observes correct data (validated via
  in-kernel `A[x,y]` reads and a device-side drain kernel).
- Earlier "all-zeros" reports were harness artifacts: (a) buffer-backed
  `makeTensor(descriptor:offset:)` with manually specified strides, and (b) an
  event-value reuse bug (`signalEvent(ev, 1)` a second time is already
  satisfied, so `wait(untilSignaledValue: 1)` returns before completion).
- **Real limitation found:** a user-MSL `matmul2d` kernel computes correctly
  only for a *single* threadgroup processing the *exact* descriptor tile with
  `K == 64` static extent (64x32 tile verified correct). Multi-threadgroup
  dispatch — or a K-loop over multiple 64-chunks, even within one threadgroup —
  compiles, dispatches, validates, and writes **zeros**, with no error anywhere
  (debug layer clean). `execution_simdgroups<1>` is additionally broken even
  single-tile. The deployment-target macro override
  (`__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ = 270000`) changes nothing.
  Interpretation update (M5 architecture): on M4 the tensor API lowers to
  simdgroup_matrix-class SIMD work (Rigel §4) — no dedicated matrix unit.
  On M5 every GPU core carries a dedicated Neural Accelerator (on-die matrix
  engine analogous to NVIDIA tensor cores, arXiv:2607.19438; the accelerators
  are reached ONLY through the Metal 4 tensor API / matmul2d and are not used
  by default). The silent multi-tile zeros therefore read as an incomplete
  beta driver lowering for user-MSL matmul2d on the new matrix hardware, not
  a scheduling gap in a SIMD fallback. Consequences for this suite: the S2
  "msl" simdgroup_matrix rows measure the SIMD FMA path (0.8–1.2 TFLOPS),
  and MPS “Neural Accelerator” rows (1.25 TFLOPS fp16) also still look
  SIMD-path-bound — the dedicated matrix units remain effectively
  unmeasurable through this beta toolchain beyond single tiles.
  Secondary suspect worth revisiting on GM toolchains: cooperative tensor
  register layouts are generation-specific (per pytorch/pytorch PR #182256,
  reviewed 2026-09-25: the layout changed between apple8 and apple10 with
  silently wrong results when mismatched; Rigel recovered the 8x8 fragment
  layout for M4/apple7 only). This
  machine is apple10 (M5 Max, family list in the machine fingerprint) and the
  beta compiler's single-tile-correct/multi-tile-zero signature is consistent
  with a partially-implemented apple10 layout path rather than a usage error
  on our side (dispatch geometry matches the MPS header example exactly).
- Consequence for S2: the tensorops rows are reported `validation_failed
  (excluded)` — not `skipped` — because the op *is* present and *did* run; it
  produced wrong output. A dedicated single-tile capability probe
  (`tensorops/matmul2d probe`) runs first and records the "present and
  functionally correct within the single-tile limit" evidence state.
- fp8/fp4 on this hardware path are additionally covered via MSL scalar SIMD
  decode kernels (`gemm_fp8`, `gemm_fp4`): per Rigel's finding that M4-class
  fp8 is emulated in software on the fp16 datapath (0.94x), the honest
  fp8/fp4 measurement on a pre-M5 tensor unit *is* the decode path
  (~120–340 GFLOPS), not a fictitious 2x fp16 peak.

### R8.1 — macOS 27.0 GM delta check (2026-09-25, verified against the live SDK + runtime)

Cross-checked community findings (tessera PR #749; MLX issue #4533) against
this machine's MacOSX27.0 SDK and the running OS:

- SDK enums DO exist: `MPSDataTypeFloat8e4m3/e5m2/e8m0`,
  `MPSDataTypeFloat4e2m1` in MPSCoreTypes.h (`macos(27.0)`), and matching
  `MTLTensorDataTypeMetalFloat8E4M3/E5M2/UE8M0`, `...Float4E2M1` in
  MTLTensor.h. Multi-plane tensors are real: `MTLTensorPlaneTypeScales`
  auxiliary plane + `MTLTensorAuxiliaryPlaneDescriptor` (default plane dtype
  `MetalFloat8UE8M0` = E8M0 block scales) +
  `MTLTensorBufferAttachments setBuffer:offset:forPlane:`.
- **But `MPSMatrixMultiplication` still rejects them on the runtime:** live
  probe (child process per dtype, n=256) — descriptor + MPSMatrix + op
  construction succeed for e4m3/e5m2/e2m1/e8m0, then
  `MPSMatrixMultiplication.mm:3260` hard-asserts `Input data type must be one
  of MPSDataTypeFloat32, MPSDataTypeFloat16, MPSDataTypeInt8, or
  MPSDataTypeInt16.` (SIGABRT, no NSError path). Conclusion: the "MPS fp8"
  27 feature is the MPSGraph `matmul` dtype path, not the legacy
  MPSMatrix API. S2 keeps fp8/fp4 as MSL-only rows; mpsDtypes filters them.
- 128-byte alignment for sub-byte tensor planes is SDK-documented
  (MTLTensor.h:319-324: Int2/UInt2/Int4/UInt4/Float4E2M1/Float8E4M3/E5M2/
  UE8M0 require 128 B-aligned buffer offsets). Our S2 MTLTensor path only
  uses fp32 (offset 0), so unaffected; any future sub-byte MTLTensor lane
  needs 128 B pre-flight checks, and tessera reports the driver applies
  4-bit offsets at 2x the byte value (256 B granularity) — a silent
  correctness trap, worth re-verifying if an fp4 tensorops lane is added.
- The `matmul2d::get_destination_cooperative_tensor` `enable_if`
  tightening (`__is_tensor_type_v || __is_cooperative_tensor_type_v` on both
  operands, which broke MLX builds) does not affect our code: the S2
  production tensorops kernel takes plain tensor arguments and the R8
  diagnostic's coop-template usage already compiles clean against the 27 SDK.
- On the multi-tile zero: tessera's constraint analysis says tiling
  semantics are unchanged from 26.x (27 only adds types + planes), so this
  is not a 27 regression. Our controlled result — *exact single-tile
  correct, multi-tile or multi-64-K-chunk zero, debug layer clean* — still
  points at the apple10 layout/lowering gap described above rather than a
  stride error (fp16 tiles, extents matching the descriptor, MPS-header
  example geometry).
- 27 MSL surface confirmed in the SDK's MPPTensorOpsMatMul2d.h operand
  table: native `metal_fp8_e4m3/e5m2_format`, `metal_fp4_e2m1_format`,
  `int2b/uint2b_format` (2-bit ints) as matmul2d operands, plus
  `get_left_input_cooperative_tensor()` (coop tensors as direct matmul
  inputs, eliminating the threadgroup-memory round-trip). S2's
  `to_gemm_fp8` (half x fp8-e4m3 -> fp32) and `to_gemm_fp4`
  (fp4 x fp4 -> fp32) kernels exercise these natively under the
  single-tile limit. Caveats accepted from tessera: fp8 ~0.77-0.93x of
  fp16 device time on M1 Max (still emulated there; M5 matrix-unit path
  unmeasurable multi-tile here), fused epilogues 0.5-5% slower than
  decomposed, host-side packing 4-10x matmul cost.
- E8M0 scale planes exist ONLY host-side (MTLTensor multi-plane API); the
  MSL matmul2d header has no scale-plane operand on this SDK — no MSL
  consumer exists to bench yet, so the suggested "fp8 + E8M0 scale-plane
  tensorops lane" is a no-op on this toolchain (tracked as a GM+1 revisit).

## R9 — S1 memory-bandwidth stability on M5 Max / macOS 27 (measurement study)

Goal: 3 runs within 10% per subtest (criterion 3). Raw suite: 35–50%
cross-run spread on every subtest. Systematic study (v2–v6 runs,
`tools/check_stability.py`), causes found and fixed in order:

1. **Per-dispatch CPU overhead dominated GPU subtests.** A 256 MB read is
   ~0.5 ms of GPU work; `commit()+waitUntilCompleted()` adds 1–2 ms of
   variable CPU-side latency, so wall-clock timing measured the OS, not the
   DRAM. Fix: time via Metal `gpuEndTime-gpuStartTime` over batches of 8
   dispatches per command buffer. Effect: GPU copy/write/triad cross-run
   spread collapsed from 36–45% to 0.02–0.35% immediately.
2. **Scalar `mem_read` kernel was 3–5x under DRAM rate.** One float per
   thread with a 67 M-thread grid achieved 103–193 GB/s. Fix: float4
   grid-stride loop over a fixed 4096-threadgroup grid. Now 548–559 GB/s,
   consistent with triad (551–553) and the packaged ~546 GB/s peak.
3. **Idle-launched processes inherit a cold power-allocation state.**
   A process started after idle runs its GPU subtests ~1.8x slow for its
   entire lifetime (CPU keeps boost, GPU stays at idle clocks — CPU and GPU
   subtests anti-correlate). A 3 s primer was insufficient; a 20 s mixed
   CPU+GPU primer (vDSP fills interleaved with GPU copies) before the first
   subtest fixed all GPU rows (worst GPU row now 7% — L2 warm-content
   effect on reads).
4. **CPU store bandwidth is session-state banded.** cpu_write/cpu_read
   alternate between two bands (2.3 ms vs 5.3 ms per 512 MB fill = 235 vs
   103 GB/s) with run1 fast-writes and runs2/3 slow-writes (mirror image on
   reads) in the 120s-cooldown series; within-process spread is <2%. An
   allocation experiment (fresh 512 MB fills, /tmp/allocprobe): malloc
   first-fill 21–44 GB/s drifting up to 84 across reuses, `posix_memalign`
   2 MB-aligned stable 82–88, fresh `mmap` stable 21–24 — page allocation
   and first-touch placement decide the band. Aligned+multi-core-first-touch
   CPU buffers were applied; the cross-run band persisted at 120 s cooldowns.
   The 600 s cooldown series (v6) then put **all three CPU rows in the good
   band** (spread 7.5/9.1/21.1% — the triad row still failed via one
   externally-contaminated run), confirming the band is a power-budget latch
   that a long idle resets.
   v6 caveat: while CPU recovered at 600 s cooldown, the GPU side of v6 run3
   hit a slow episode (copy 1.42→3.99 ms) that the fixed 20 s primer did not
   clear — deep-idle recovery is stochastic, not duration-bounded.
5. **Fixed-length warmups cannot guarantee convergence.** Final S1 design:
   a self-calibrating primer — sustained mixed vDSP-fill + GPU-copy load in
   2 s windows, GPU-timestamp bandwidth, running until copy throughput is
   either within 1.5% of the previous window (plateau) or >=360 GB/s (this
   machine's sustained copy plateau), minimum 20 s, hard cap 120 s.
   CPU subtests additionally switched to three independent 3 s windows with
   per-window medians (v7 run2 showed bimodal in-process samples —
   min 1.87/max 5.43 ms for cpu_write — i.e. daemon interference arriving
   mid-measurement; windowing quarantines it to one window), plus a
   pre-flight host_cpu_load_info check that aborts the suite when the SoC
   is busy. Stability protocol: 3 runs at 600 s cooldowns (the CPU-latch
   reset the platform needs). [v8 series: CPU rows passed all-<10% —
   write/read/triad spreads 0.69/5.15/4.70% — but GPU run3 regressed, so
   the windowing+gate was kept and the primer made self-calibrating.]
6. **The CPU-write band is a physical free-memory effect, not DVFS.**
   Later series (v9–v10, 22:00+) could not reach the good write band at any
   cooldown: cpu_write pinned at 101–151 GB/s. Diagnosis: an external
   long-lived process (omlx-server, a local model server) holds 87 GB
   resident (~74% of RAM), leaving 26% free + active swap. Fresh 1.5 GB
   bench buffers then get scattered/remote physical placement at first
   touch, halving store-stream bandwidth; as the resident set grows during
   the evening, the good band becomes unreachable. Consistent with the
   alloc probe (2 MB-aligned stable vs fresh-mmap slow): placement decides
   the band. Implications: S1 CPU numbers are conditional on machine free
   memory — record machine free-RAM with results when cross-comparing, and
   run on a quiesced machine for the reference numbers. The GPU rows
   (Metal shared buffers, IOKit-pooled) are unaffected: they stayed
   0.5–2% stable across every state. Final S1 protocol: per-run GOOD gate
   (gpu_copy >= 340 GB/s AND cpu_write >= 200 GB/s) + <10% cross-run check;
   CPU consistency then rides on machine quiescence.
7. **Band transitions track idle time, and idle HURTS.** Long cooldown
   series (600 s) kept producing the low write plateau (101-125 GB/s) hour
   after hour, while a first-run-after-prior-series launch measured good
   (236 GB/s) and every 600 s-cooled follow-up measured low. Interpretation
   corrected: under sustained external resident-memory pressure (omlx 87 GB
   + swap), the page allocator routes *freshly* allocated 512 MB buffers
   into degraded physical placement when memory has been quiesced long
   enough for reclaim/compression to fragment the free pool; back-to-back
   launches (30-120 s cooldown) reuse freshly recycled placement and stay
   in the healthy band (~235 GB/s write, GPU 380 GB/s copy — and the v8
   all-CPU-<10% pass was such a series). Final protocol: short cooldowns
   (COOLDOWN=30..120), per-run GPU-convergence gate (copy >= 340),
   same-band trio enforcement, and the <10% cross-run gate as the arbiter.
   mlock on first-touch kept as cheap insurance (no measured placement
   effect). 23:36 worst-of-3 probe (512 MB): 1-thread memset 134,
   1-thread vDSP_vfill 139, 16-thread concurrentPerform vDSP_vfill 242.
   Outcome: the protocol passes deterministically — final trio (band-reset
   to the stable low plateau) measured cpu_read 3.30%, cpu_write 0.03%,
   cpu_triad 1.65%, GPU rows 0.01–2.54% cross-run spread: <10% on all 8
   subtests, FAIL-free.

## Spike outcome

_Written after `bench spike` runs; see `results/ane_spike.json`._

- GO/NO-GO: **GO** (in-process route, macOS 27.0 build 26A428, M5 Max)
- compile+load ms: 57.0 (fp16 512×64 d1), 16.6 (int8 256×64 d2)
- eval median ms: 0.128 (fp16), 0.116 (int8)
- max_abs_err vs CPU fp32 reference:
  - fp16 512×64 d1: **3.6e-05** (tol 1.0e-04) — PASS
  - int8 256×64 d2 (W8A16): **3.0e-05** (tol 1.6e-04) — PASS
- The initial NO-GO (max_rel_err≈13186) was a **weight-blob format mismatch**, not a
  missing API: maderix's 64-byte-header layout is missing the `0xDEADBEEF` magic, the
  dtype/size/offset fields, and the per-layer quantization chunk triplet that macOS 27
  requires. Once the R4 layout was implemented, both fp16 and int8 conv chains verify
  against a CPU fp32 reference within fp16/quantization tolerance.

## Verdict

Criterion 5 (S3 raw ANE) is **satisfiable on macOS 27** via the in-process
`_ANEInMemoryModel` route (no XPC, no sudo). Requirements that were previously
unknown and are now implemented:

1. Strict `weight.bin` layout (R4) — file header + magic chunk headers + per-layer
   int8/zp/fp16-scale chunk triplet, 64-byte aligned.
2. int8 as W8A16 with per-channel `constexpr_affine_dequantize` (R2), using the
   `program(1.0)` MIL dialect.
3. IOSurface I/O sized to the max input/output byte size (R5).

The ~119 in-memory compiles/process limit (R6) is handled by reusing one compiled
kernel per S3 config. Spike A gates criterion 5 → **gate passed**.

## R10 — S4/S5 deep-graph findings (Core AI + legacy Core ML, macOS 27 GM)

1. **ANECCompile internal error on deep conv chains.** Any `.aimodel` with 8+
   chained 512ch Conv2d fails the MLIR `mps-to-anec` conversion
   (`ANECCompile: internal error`), and the Core AI runtime transparently
   falls back to GPU. All three specialization variants (default /
   ane-preferred / gpu-preferred) then land within noise of each other
   (deep_fp16 1.39–1.47 TOPS, deep_int8 0.64–0.68 TOPS at iters=100). A
   single 2048x2048 GEMM graph *does* compile for ANE: 87–95 TOPS, with
   ane-preferred fastest (95.4). So on this OS the "official stack ANE peak"
   is reachable for matmul graphs but not for long conv chains.
2. **W8A8 PTQ depth limit for validation.** 64 chained quantized layers with
   the deterministic-but-oscillating weight pattern decorrelate from the
   torch fake-quant reference (cosine -0.13, outputs ~1e37) — accumulation
   noise dominates. `deep_int8` is therefore generated at depth 8 (same
   shapes); validation then passes cleanly (normalized_rel 0.0, cosine 1.0
   on the official run). The Swift gate uses normalized_rel<=0.15 /
   cosine>=0.999 for quantized models (standard PTQ metrics) and reads the
   per-graph `flops` from `<name>.verify.json` instead of hardcoding.
3. **coremlc-on-MIL-text is broken on 27 GM** (`Field number 13 has
   wireType 7`, plus `--minimum-deployment-target` rejected). The legacy
   Core ML baseline therefore goes through **coremltools 9.0** in the venv:
   `torch.export` + `run_decompositions({})` (9.0 rejects undecomposed
   TRAINING dialect), `ct.precision.FLOAT16` (module-level
   `ct.compute_precision` no longer exists; `TensorType(dtype=torch.float32)`
   no longer accepted — omit dtype). mlprogram conversion makes **I/O fp16
   end-to-end**, so the Swift side feeds/reads `Float16` MLMultiArrays.
   Result: 0.19–0.20 ms/run → 1.32–1.39 TOPS validated (normalized_rel
   5.0e-2, cosine 0.9993; pointwise rel is 2.5e+1 at near-zero outputs —
   expected for fp16 chains, same acceptance rule as S4 quantized).
   `MLModel.load(contentsOf:)` needs `MLModel.compileModel(at:)` first on
   an .mlpackage (27 runtime requires the compiled .mlmodelc path).
4. **Overhead ladder at this graph size (0.2 ms regime):** raw ANE spike
   dispatch floor ~0.11 ms (S3) < CoreML/ANE-preferred 0.193 ms ≈ Core AI
   matmul 0.180 ms << Core AI deep conv chains (GPU fallback) 0.39–1.5 ms.

## R11 — the ANE weight budget counts weights-dict entries (files), not const() nodes

Cross-machine result, confirmed 2026-09-26. The Mac mini M4 session
(EXP-021, `bench/ane-dma-test.m`) ran the decisive A/B: per-layer blob files
compile at N=16 and **fail at N=17**; one shared blob with per-tensor
BLOBFILE offsets compiles at N=17, 32, and **128**, every arm
arithmetically correct (out[0]=64.0, ones-weights x ones-input). The believed
"16 weight tensors per program" ceiling (Orion's constraint #16) is really
"16 entries in the weights dict" — references are unlimited in practice.

This repo's S3 data corroborates independently, on a different chip (M5 Max,
h17) and at larger scale: the bridge passes exactly **one** dict entry
(`ane_bridge.m:94`, `wdict[@"@model_path/weights/weight.bin"]`) and
`MIL.swift` puts every tensor into that file —
- `peak_256x64_d256`: 256 fp16 W() const nodes, 256 references, one entry —
  compiled, 7.08 TOPS (25x past the believed ceiling);
- `peak_int8_512x64_d128`: 128 layers x 3 references (quantized_data +
  scale + zero_point) = **384 BLOBFILE references in one dict entry** —
  compiled, 13.32 TOPS (24x).

Practical consequences for a private ANE engine:
1. Programs are sized by graph/scheduling needs, not weight count. Orion's
   72 programs for a 110M model collapse toward a handful.
2. The ~119 compiles/process limit (R6) stops binding: fewer programs =>
   fewer compiles; a per-program cache covers the rest.
3. Every deleted program boundary was an ANE<->host (or ANE<->ANE) transfer
   plus a request round-trip — the dispatch floor (0.133 ms here) is paid
   once per boundary, so boundary removal is a direct latency win.
4. Layout discipline: one blob per program, header at 0, tensor blocks
   addressed by uint64 offsets; per-tensor block = 64 B header + payload
   (matches R4 on M5 Max and the M4 "reference 64 / payload 128" reading —
   same layout described from two anchors).
5. Compile != correct: unlocatable offsets fail loudly, but locatable-wrong
   offsets silently read wrong bytes (M4 EXP-020). A deep-chain arm is only
   trustworthy with an arithmetic correctness probe; S3's timing runs lean
   on the SpikeA numeric gate for exactly this reason.

## R12 — H17 in-memory path rejects multi-entry weights dicts (2026-09-26)

Attempted the M4 per-file A/B on M5 Max via `ane_bridge_compile_files`
(multi-entry dict, per-file `w<i>.bin`, correct per-file headers/offsets):

- 1 entry named `w0.bin`: compiles, evals, output byte-identical to the
  shared-blob arm (`blob_probe_d1` — shared 0.116 ms / per_file 0.105 ms;
  names are flexible, single-entry equivalence to `weight.bin` confirmed).
- 2 entries: `ANECCompile FAILED: InvalidMILProgram`.
- 16 entries: same `InvalidMILProgram` (bundle kept, framework rewrites the
  `data` file to a single blob-sized region — it bookkeeps exactly one
  entry on this path).

Conclusion: on H17 the `_ANEInMemoryModelDescriptor` in-memory route is
single-entry — the M4 session's 16-per-file arms (their `ane-dma-test.m`)
must go through the on-disk bundle / Orion offline compile path, where 16
entries work and 17 fails. The M4 shared-vs-per-file speedup
(0.175 → 0.111 ms, 1.58x) is therefore **not reproducible on H17 via the
in-memory path** and stays UNVERIFIED here; on H17 the actionable finding
stands unchanged: one entry with unlimited references (up to 384 measured)
compiles and runs. Re-derivation of the per-file penalty on H17 would
require porting the arms to the bundle-compile path (orion-style
`ANECompiler` XPC) — out of scope for this harness.

## R8.2 — nax_ladder GM retest + poison extension (2026-10-22, macOS 27.0.1 / 26A434)

Context: the NCM campaign needed written-zero vs never-written attribution
for apple10 tensor-op failures. `tools/probes/nax_ladder.m` was extended
(0xDEADDEAD C pre-fill + NCM-geometry arms: fp32 out-of-table, tf32
relaxed=true, device destination, strides third-arg, K=256 dynamic, K=256
4-chunk accumulate loop) and run on the GM toolchain. Ledger:
`results/2026-10-22T12-37-03Z-bda452aabd60bf26.json` (nax_probe_matrix).

Findings:

- **The classic buffer-backed + MTL3-dispatch documented form fails at
  SINGLE TILE**: 1x1 64x32x64 fp16 = 1056 NaN + 992 unwritten(poison), 0
  exact. The PRISTINE Oct-10 probe (git HEAD, re-run today) fails
  identically (1056 bad + 992 zeros-where-nonzero), so this is the platform,
  not the new arm code. Removing the `.version4_0` pin does not change the
  outcome. OS timeline (web-sourced, not guessed): macOS 27.0.1 (26A434)
  shipped 2026-09-28 with UNSPECIFIED bug fixes and no release-notes page
  (developer.apple.com/news/releases 09282026c; osxdaily 2026-09-28), and
  the Oct-21 bench ledger + today's sw_vers both read 26A434 - so the
  Oct-10 run was already on 27.0.1 and its header-comment "single-tile PASS"
  was either mis-recorded or run against different inputs/flags we can no
  longer reproduce. As of today the classic form is NOT-KNOWN-GOOD at any
  tiling.
- M5-specific context for why M4-era notes don't transfer: on M5/A19 the
  tensor path targets the new Neural Accelerators (WWDC26 session 330),
  whereas Rigel (arXiv:2606.12765) proved matmul2d lowers to
  simdgroup_matrix/ALU on M4 - a different lowering, and Rigel's own
  documented law there ("tensor_inline slice() does not offset reads; only
  the (0,0) tile computes correctly") matches our multi-tile/multi-K zero
  family regardless of generation. Also SDK-side: the MPP tensor-ops
  headers ship with macOS, not Xcode (ComfyUI-AppleSilicon-FP8 README), so
  point releases can silently move this ground.
- TRIVIAL non-MPP control (same buffers/dispatch) writes every cell finite:
  the fault is inside matmul2d lowering/dispatch for this form, not harness
  plumbing.
- Device-destination arm: all-NaN, confirming the Oct-10 note.
- fp32 out-of-table and tf32 arms on this path produce **partial miscompute**
  (976/2048 bad), not the clean written-zero the MTL4 tensor-object path
  produces (NCM results/2026-10-22T11-31-39Z): the silent-failure MECHANISM
  is invocation-form dependent. FB package updated accordingly.
- strides third-arg to `tensor_inline` is now **compile-rejected**
  ("no matching constructor") — the silent-misread mine in older notes is
  fixed at compile time by the SDK; the misread-NaN anecdote is historical.
- Multi-K (dynamic K=256 and 4-chunk accumulate loop): unattributable on a
  path whose single-tile baseline is already corrupt; clean multi-K evidence
  remains the MTL4 arms (written-zero 2048).
- Cross-path summary for 27.0.1: MTL4 tensor-object path fp16 single-tile
  ok 4.7e-07 (NCM probe, same day) — the trainer lane is unaffected; the
  classic buffer form is quarantined pending a radar answer (FB pending).

Build note: the extended probe needs `clang++ ... -x objective-c++ -std=c++17`
(raw-string MSL + the linker's C++ personality); plain `clang` .m builds the
old form only by permissiveness.
