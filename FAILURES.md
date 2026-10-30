# FAILURES — LLM host-language lane matrix (night 2026-10-27/28)

Same discipline as the program FAILURES.md (symptom → wrong assumption → actual
cause → fix → rule). Session: openfox-e9e19426. Raw evidence in `logs/`.

### LF-01 · 30B "Insufficient Memory" that wasn't a memory problem
- **Symptom:** `L0 Qwen3-30B-A3B-4bit` died in `mx.eval([c.state ...])` with METAL
  Insufficient Memory while `peak` showed only ~17 GB of a 130 GB limit; standalone
  load+generate('Hi') passed.
- **Wrongly assumed:** MLX accounting bug or the 128 GB box actually running out.
- **Actual cause:** co-tenant GPU pressure + default wired residency behaviour when
  other queue jobs ran concurrently; the whole matrix must be one-GPU-job-serialized.
- **Fix:** `--wired-bytes 96e9` (`mx.set_wired_limit`) + serialized night queues →
  L0 30B completes 768 steps @ 8.70 ms/step.
- **Rule:** on shared Apple Silicon, MoE-class failures that pass standalone are
  scheduling artifacts; serialize GPU jobs and set the wired limit explicitly
  before writing any "cannot run 30B" verdict.

### LF-02 · C host 4× slower after switching metallibs — the slow-kernel metallib
- **Symptom:** identical C harness: lm_head dequant+matmul 0.935 ms/op vs 0.256
  ms/op on another day; no code change that explained 4×.
- **Wrongly assumed:** GPU throttling or thermal.
- **Actual cause:** the harness picked up the Python wheel's `mlx.metallib`; the
  SPM-built `default.metallib` (from mlx-swift 0.32.3) has the fast quant kernels
  for this OS/toolchain. Same engine, different kernel archive.
- **Fix:** `lanes/c/mlx.metallib -> ../swift/.build/.../default.metallib` symlink;
  A/B saved as `lanes/c/gpu_probe2.c`.
- **Rule:** in a multi-runtime matrix, pin the METALLIB as a first-class version
  field; kernel-archive skew mimics "language is slower" perfectly.

### LF-03 · `Failed to make new command queue` ~250 steps into C decode runs
- **Symptom:** C 1.7B/4B died at step ~250/193 with
  `mlx-c/stream.cpp Failed to make new command queue` — exactly when a 30B job had
  just died OOM in the same queue window (stale driver state), but also on clean
  days.
- **Wrongly assumed:** Metal driver leak; a per-process limit being hit externally.
- **Actual cause:** `S()` created a fresh `mlx_stream_new_device` per call —
  dozens of streams (and GPU command encoders) per decode step.
- **Fix:** cache the GPU stream in a static (one stream for the process);
  queue re-run 1.7B/4B completed 768 steps.
- **Rule:** stream creation is not free in mlx-c; treat `mlx_stream` like a file
  handle — create once, reuse.

### LF-04 · `[slice] Invalid number of indices` — mlx_slice with strides=NULL
- **Symptom:** C and Rust MoE routing died at first MoE layer:
  `[slice] Invalid number of indices or strides for array with dimension 1`.
- **Wrongly assumed:** argsort output shape mismatch; a Rust binding bug.
- **Actual cause:** core `slice()` validates `strides.size() == a.ndim()` — passing
  `NULL/0` strides through `mlx_slice` is rejected for 1-D arrays (unlike Python's
  overload). The dense path never exercised slice, so it hid until 30B.
- **Fix:** pass explicit `strides=[1]` arrays (C + Rust harnesses).
- **Rule:** the C API is stricter than the Python DSL — port op-by-op against the
  C++ signature (`mlx/ops.cpp` validation), not against Python call style.

### LF-05 · `Python positionally gathered the wrong axis` (shapes (8) vs (128))
- **Symptom:** L0m 30B mirror died: `[broadcast_shapes] Shapes (8) and (128)
  cannot be broadcast` inside `mx.gather_qmm`.
- **Wrongly assumed:** gather_qmm indices were malformed in the harness.
- **Actual cause:** `mx.gather_qmm(x, w, s, b, idx, ...)` binds `idx` to
  `lhs_indices`; Qwen3-MoE expert routing must gather the **rhs** (experts are the
  weight matrices). C `mlx_gather_qmm` takes both explicitly, so only the Python
  mirror was affected — easy to misread as "the C lane is slower/odd" if unchecked.
- **Fix:** `rhs_indices=idx` keyword.
- **Rule:** for gather-qmm, name the index side explicitly in every lane;
  positional indices in a dual-index API is a correctness landmine.

### LF-06 · Native loops hit `metal::malloc Resource limit (499000)` on MoE
- **Symptom:** C 30B died at step ~550, Rust 4B/30B died mid-run with
  `[metal::malloc] Resource limit (499000) exceeded` (`allocator.cpp:143`);
  Python lanes (same graph!) and Swift survived. `CLEAR=50` (per-50-step
  `mlx_clear_cache()`) did not rescue.
- **Wrongly assumed:** handle leak in our arena (partially true for dense:
  ~2 MB/step live growth from superseded kcache wrappers — fixed with RAWA-owned
  kcache/vcache — but MoE still died).
- **Actual cause:** the per-token gather set for 128 experts allocates a new live
  buffer set every step; the allocator cache recycles buffers but `num_resources_`
  (the driver resource table, limit 499000 via `iogpu.rsrc_limit`) keeps counting
  live MTLBuffer objects, and MoE step shapes (varying by which experts are hot)
  defeat size-bucket reuse. Swift survives because mlx-swift-lm's LMInput/session
  path reuses a fixed expert-window workspace; Python survives via mlx_lm's
  cached-switch-mlp path.
- **Fix (night):** salvage short runs (96/64, post-review 128×1+prefill)
  recorded as partial cells + this row; 768-step native-MoE is a follow-up:
  reuse fixed-size expert gathers or bump the residency design (mlx
  issue-worthy). Post-relink (LF-11) the Rust dense 4B also died at the
  1024-step full run — same wall, engine-independent.
- **Rule:** hand-rolled native decode loops are fine on dense; on MoE the
  *allocator-resource* cost, not FLOPS, is the wall — measure `num_resources`
  headroom before promising long native generations.

### LF-07 · Swift L1 binary was a stale stub for one whole queue
- **Symptom:** queue5 first run: L1 "completed" in 20 ms, row file absent, log said
  `stub` — the 02:43 binary predated the real L1 implementation.
- **Wrongly assumed:** the harness failed to write `--out`.
- **Actual cause:** `swift build` artifacts under `.build/out/Products/Release`
  weren't refreshed (a second scratch dir `.build/release` was).
- **Fix:** copy the fresh `.build/release/msl` onto the out-path binary; rebuild
  after every source change; `exit(0)` added after emit so queue steps actually
  terminate.
- **Rule:** every lane binary must echo its own build stamp+version in each row
  (`version` field added) — a missing row with a logged "success" is the smell of
  running an old binary.

### LF-08 · dispatchMain() hangs the queue after emit
- **Symptom:** `msl` wrote the complete row but never exited; queue sat on step [4].
- **Cause:** `Task{} + dispatchMain()` idiom keeps the runloop alive after the
  benchmark finishes.
- **Fix:** explicit `exit(0)` after `emitRow`. Killed the stuck pid to unblock.
- **Rule:** batch harness entrypoints exit deterministically; "row written" ≠ done.

### LF-09 · prefill arms missing for microbench lanes (gap, not excuse) — RESOLVED
- **Symptom:** C1 requires prefill@32/512/2k per cell; L0/L1 have it, L0m/L3/L4
  rows didn't (decode loop only harnesses).
- **Fix:** review round added prefill arms to L0m (`prefill_bench` with exact
  filler-id prompts), L3 (C, `decode_step(…, n)` multi-token arm) and L4 (Rust
  mirror); L1 now reports chat-template-calibrated prompts with
  `promptTokenCount` verified exact per row. C/Rust/L0m measure prefill after
  the decode session in identical order (honest, pessimistic for native hosts).
- **Rule:** missing measurement ≠ slow measurement; keep the absence explicit —
  and close it the same night when the harness fix is ~30 lines.

### LF-11 · Rust lane linked a different engine core than it claimed (C2 fail)
- **Symptom:** review verdict: L4 vendored mlx-sys 0.6.0's own mlx-c core
  (v0.32.2) via crates.io CMake build while Swift/L3 use mlx-swift 0.32.3
  (rev 19601207e). Token parity held (62/64 first-mismatch at tie step 19) but
  the same-binary guarantee behind "engine is constant across lanes" was broken.
- **Wrongly assumed:** "same source line" from the vendored crate == same core.
- **Fix:** `build.rs` now bindgens the 0.32.3 `mlx.h` from the Swift checkout
  and links `lanes/swift/.build/out/Products/Release/Cmlx.o` directly (same
  bytes the L1 app embeds); `mlx-sys` dependency dropped; metallib symlink
  installed next to the executable (the core resolves it relative to the
  binary, not cwd — first relink test failed `Failed to load the default
  metallib` until that symlink existed in `target/release/`). Re-measured:
  Rust 1.7B short-run decode jumped 141.9→258.5 t/s — the patch skew was real
  and material, pre-fix rows superseded.
- **Rule:** a lane's "engine" claim must name the exact object/dylib it links;
  version strings from metadata (crate versions) are not provenance.

### LF-12 · uninitialized `mlx_array` outputs: O0 passed, O2 segfaulted
- **Symptom:** C MoE prefill arm SIGBUS'd inside Cmlx `swap<ArrayDesc*>` on the
  first gather; `-O0` build of the same source ran fine. Rust equivalent failed
  differently: `expected a non-empty mlx_array` from `mlx_divide`.
- **Actual cause:** C: `mlx_array x1, xb;` left ctx = stack garbage; the C-API
  setter move-assigns over it and decrefs the garbage pointer — only O2's
  stack layout made it fatal. Rust: `let mut wsel = self.nv();
  ck!(mlx_divide(&mut wsel, wsel, sumw))` shadowed the input with the fresh
  empty output array (same pattern the C mirror doesn't have).
- **Fix:** all outputs go through `NEWA()`/`self.nv()`-distinct names; the
  30B C/Rust rows now pass with exact n-token MoE math (argsort axis -1,
  uint32 row-offset gather indices — note idx dtype int64 regression cost
  2× on expert gather, fixed by staying uint32).
- **Rule:** C-API handle structs are non-trivial C++ in disguise: zero-init or
  arena-allocate every out-param, and re-check shadowed names when porting
  C sequences to Rust.

### LF-10 · spine.py verify fails at session start (pre-existing)
- `python3 tools/spine.py verify` exited nonzero before any change of this
  session (owner-side drift). Not caused by, and not blocking, lane work.
  Record; do not "fix" blind under timebox.

### LF-13 · FoundationModels session intermittently returns empty completions
- **Symptom:** L2 4B/30B cells wrote `wall_tps: 0` rows (0.4 s, 0 tokens);
  1.7B showed the same flake nondeterministically (one run empty, next fine)
  at temp 0.0 and 0.7 — same prompt that produced 19 t/s cells in queue5.
- **Wrongly assumed:** model-specific adapter incompatibility.
- **Actual cause:** the FM session path intermittently emits an immediate-EOS
  empty completion; wall time of the empty call (~0.4 s) is too short to be a
  decode, so it is an adapter sampling glitch, not a slow path.
- **Fix:** measured runs retry up to 5× on empty content (timing excludes the
  empty attempt); `retries` recorded per run in the row for provenance.
- **Rule:** when a framework lane emits zeros, first ask "did it generate at
  all?" — dump the content before tuning any knobs.
