# What is native, and what may be compared

Every "engine A is Nx the engine B" number this repo publishes is only as good as the answer to
two questions: **what operands does each engine accept without emulation, and what arithmetic does
it actually do on them?** This file answers both per engine, anchors every cell, and then states the
rules that follow for our A/Bs.

The rule came from the owner, 2026-10-15:

> "say clearly what are native operations, and compare native ones against others WHEN THERE ARE.
> Otherwise it's not a fair comparison."

The concrete trigger: our flagship GPU A/B ran a **tf32 tensor path** (`relaxed_precision=true`)
against an **exact-fp32 SIMT shader** and printed a single delta. Those two arms do not do the same
arithmetic (see §2, `GPU tensor (NAX/MPP)` vs `GPU shader/SIMT`), so the delta is a *precision*
comparison wearing the label of a *datapath* comparison. That is what this file stops.

---

## 1. Lane topology (read this first or every other section misleads)

```
CPU  ──  ANE  ──  GPU { each core:  shader/SIMT datapath  +  NAX tensor datapath }
```

- The **NAX (Neural Accelerator / MPP tensor) units are not a separate engine** — one GPU core
  carries one NA, so the tensor path and the shader path are two datapaths *inside the same GPU
  core*, sharing its registers, power budget and dispatch. ("every core carries a dedicated Neural
  Accelerator", BaseRT, arXiv:2607.19438, `knowledge/ane/SOURCES.md`; our own lane framing in
  `results/EXP-029-nax-crosslane/README.md` §0.)
- So **"NAX vs SIMT" is *intra-engine***. Only "GPU{NAX,SIMT} vs ANE vs CPU" is an engine choice.
- On this box: M5 Max, macOS 27.0.1 (26A434), ANE family **h17c**, **40 GPU cores = 40 NAX units**.
  The earlier "32 NAX" reading was an ANE-derived count and is withdrawn
  (`knowledge/ane/16-neural-accelerators.md`).

---

## 2. The matrix — matmul / GEMM

`Accum` = the type partial products are summed in. "Emulated" = the engine *accepts* the operand but
does the arithmetic in a different (lower) precision; "Fallback" = the engine refuses and the op
runs elsewhere.

| engine | native operand dtypes (no emulation) | accum | accepted-but-emulated | rejected / fallback | anchor |
|---|---|---|---|---|---|
| **CPU** (Accelerate / vDSP / BNNS) | `fp32`, `fp16` (BNNS) — IEEE scalar fma is the reference | `fp32` | — | — | `compute-units-and-authoring.md:20` (`CPU (BNNS)` precision = fp32/fp16); `README.md` (CPU read via vDSP) |
| **GPU shader / SIMT** (MSL) | `float` (fp32, exact), `half` (fp16); `simdgroup_matrix` over fp16 | `fp32` | — | fp32 is *native here* (that is the whole point) | `tools/probes/nax_load_ab.m` `gemm_shader` (`--dtype float` = exact-fp32 fma control; `--dtype half` = fp16 operands, fp32 fma); `tools/probes/na_tiles.metal` arm B (fp16 `simdgroup_matrix`) |
| **GPU tensor (NAX / MPP `matmul2d`)** | the header enumerates **48 dtype rows**: `half×half`, `half×int8/uint8`, `int8×int8`, bfloat mixes, plus 4-/2-bit (`int4b`, `uint4b`, `int2b`, `uint2b`) and `fp4_e2m1`, `fp8_e4m3`, `fp8_e5m2` | per the header's **dest** column: `half` **or** `float`; `int32` for `int8×int8`-class | **`fp32` operands** — reachable only with `relaxed_precision=true`, which "trades accuracy for speed", i.e. computes in **tf32** | dtype combinations outside the 48 rows | **API-168** (Apple `MPPTensorOpsMatMul2d.h`, 664 lines, macOS 27 SDK: 48-row table + `matmul2d_descriptor(m,n,k,tl,tr,relaxed_precision)`); `tensorops-quantized-kernels.md:149-155`; MLX hard-codes `relaxed_precision=true` at `tools/probes/nax_steel_gemm.metal:1069-1076` |
| **ANE** (Neural Engine) | `fp16`; `int8`; `int16` | `fp32` (the 1×1-Conv engine accumulates in fp32) | `int4`/palettized are **storage**, not an arithmetic lane — a 4-bit value is a palette index into an fp16 codebook | **`fp32`** — any fp32 op/buffer "creates an f32 buffer that Neural Engine cannot execute" and falls back to GPU/CPU | LANDSCAPE-008; GOTCHAS-075(1); `compute-units-and-authoring.md:16,29` (Conv engine accumulates fp32; "No fp32 anywhere"); `ane-region-count-and-fp32-ops.md:83`; `ane-silicon-reference.md:33` |

### Reading the matrix

- **fp32 is native to exactly one of the four: the shader/SIMT path** (and the CPU). It is *not* a
  native tensor-operand dtype on NAX/MPP (it is tf32-via-`relaxed_precision`), and it is *refused*
  by the ANE. That single sentence is the reason the old A/B was unfair.
- The MPP table is **"what the API accepts", not "what is fast"** (API-168 caveat). Native ≠ fast.
- NAX/MPP natively dequantize: `half × int4b_format` or `half × int8_t` is a **single** `matmul2d`,
  no unpacking kernel. On the ANE, the same weights are a *storage* saving only (palette/stream).

### Other operation classes found documented

| op | engine | native? | anchor |
|---|---|---|---|
| conv2d | ANE | **yes** — conv is the ANE's *native* dense datapath; matmul folds into it as 1×1 Conv2d | `docs/research-notes.md:114-139` (R7: "1×1 conv chain is the ANE's native dense datapath"; arXiv:2606.22283 Listing 25) |
| conv2d | GPU tensor | yes — separate header `MPPTensorOpsConv2d.h` ships beside the matmul header | API-167 (framework ships `MPPTensorOpsConv2d.h`) |
| fp8 (e4m3) compute | ANE | **not on this box** — catalogue type exists, datapath is **H18 (A18)-only** | `ane-silicon-reference.md:40`; `ane-region-count-and-fp32-ops.md:83` |
| elementwise fma | GPU shader | yes — `float`/`half` scalar+vector ops | `tools/probes/nax_load_ab.m` `gemm_shader` |

---

## 3. The comparison rules (the point of the file)

### Rule 1 — Native-for-native only earns a *datapath* claim

A datapath delta requires **both arms on the same operand dtype and the same accumulate type**.

| pairing | same arithmetic? | may be called |
|---|---|---|
| NAX `half×half→float` vs SIMT `half` `simdgroup_matrix` (fp32 accum) | **yes** | a **datapath** delta |
| NAX `int8×int8→int32` vs an int8 SIMT path | yes | a datapath delta |
| NAX **tf32** (`fp32` + `relaxed_precision=true`) vs SIMT **exact fp32** fma | **no** | a **cross-precision** comparison — name the precision difference in the row; **do not** report it as a datapath delta |
| **ANE fp16** vs **GPU fp16** | yes (fp16 operands, fp32 accum both sides) | an *engine* delta (not a datapath one — different engines, §1) |

This was the live case, and it has since been fixed in the tool. Until 2026-10-15,
`tools/probes/nax_load_ab.m` ran the tensor arm as **fp32/tf32** and the shader arm as **exact fp32**
and set the oracle tolerance per-arm (`5e-3` for tf32 vs `1e-6` for the shader), so the
`275 µs vs 2561 µs`, `13.8 vs 183.3 mJ` rows (`REPRODUCE.md`, 2026-10-15) are **cross-precision** and
must be labelled that way.

The tool now **defaults to** the Rule-1 "yes" row: both arms take `half` operands and accumulate in
`fp32` (`dtype=half accumulate=fp32 native_match=yes`, oracle `rel ≤ 1e-4`, run at 1024³). Every row
prints `dtype=`, `accumulate=` and `native_match=`; the old fp32/tf32-vs-exact-fp32 arm is retained
only under an explicit `--dtype float`, where the tool prints `native_match=no` plus a line stating
the comparison is cross-precision and that accuracy must not be compared between the arms.

### Rule 2 — When no counterpart exists ("when there are")

Sometimes there is **no** native-for-native partner: the NAX/MPP path has **no exact-fp32 IEEE
GEMM**, and the ANE has **no fp32 GEMM at all** (§2). Then:

1. Compare against the **closest matched task type** (same M/N/K, same dispatch/work unit, same
   operands held fixed as far as possible), and
2. **label the row with the precision difference** (e.g. `tensor=tf32 vs shader=fp32-exact`), and
3. keep the resulting number out of any "datapath speedup" claim — it is a **cross-precision**
   number. "Not native" / "no counterpart exists" is a valid, valuable headline of its own; do not
   manufacture a counterpart to make a ratio.

### Rule 3 — Accuracy vs timing are reported under different conditions

- **Matched-arithmetic arms** (Rule 1 "yes") *may* report a `maxabs` / `rel` delta — and must, as the
  correctness gate. There it is a real error bar: `nax_load_ab`'s matched fp16 pair prints a non-zero
  `maxabs` (steel `2.58e-06`, shader `4.27e-06` at 1024³, `rel ≈ 5e-7–8e-7`) against a shared
  `rel ≤ 1e-4` bound, driven only by fp32 summation order at K=1024.
- **Cross-precision arms** (Rule 1 "no") may report **time and energy only**. Any `maxabs` printed
  beside them is the *precision difference*, not an error of either arm — print it **named as such**
  (the probe already does: the tf32 tolerance `5e-3`), never as "the fast arm is wrong".
- A cross-precision `maxabs` **never** authorises loosening a shared tolerance for a matched-
  precision row.

### Rule 4 — The one line a reader must check first

> **Before quoting any A/B number: check both arms' operand dtype *and* accumulate type. If they
> differ, the number is a precision comparison, not a datapath comparison — quote it with the
> precision difference attached.**

---

## 4. Anchor index

Claim IDs (`API-168`, `LANDSCAPE-008`, `GOTCHAS-075`) are silicon-ledger claim records
(`knowledge/ane/records/*.jsonl`, `knowledge/ane/all.jsonl`); their `source`/`evidence` fields carry
the primary citation. Paths without a repo prefix are relative to the silicon-ledger research tree.

| key | anchor |
|---|---|
| API-168 | Apple `MPPTensorOpsMatMul2d.h` (664 lines, macOS 27 SDK): 48-row dtype table; `matmul2d_descriptor(m,n,k,transpose_left,transpose_right,relaxed_precision)`; `relaxed_precision=true` "trades accuracy for speed". Caveat in-record: API-accepts, not API-is-fast. |
| API-167 / API-130 | the framework ships `MPPTensorOpsMatMul2d.h`/`MPPTensorOpsConv2d.h`; ANE placement dtype whitelist `fp16/f8E4M3/si8/ui8/si16/ui16` read out of a plan manifest. |
| LANDSCAPE-008 | "Core AI supports only fp16, int8 and int16 tensor types for ANE execution, and any fp32 op creates a buffer the Neural Engine cannot execute and hence falls back to GPU or CPU." |
| GOTCHAS-075 | measured placement rule: keep the ANE data path fp16; int32/bool plumbing is placed on CPU. |
| `tools/probes/nax_load_ab.m` | header + row labels: default pair is `dtype=half accumulate=fp32 native_match=yes`, oracle `rel ≤ 1e-4`; `--dtype float` = the old tensor-tf32 vs exact-fp32 arm, prints `native_match=no` and a cross-precision warning. |
| `tools/probes/nax_steel_gemm.metal` | lines 1069-1076: MLX `matmul2d_descriptor(...)` hard-codes `relaxed_precision=true`. |
| `tensorops-quantized-kernels.md:149-155` | MPP dtype table (activations × weights → dest); "native in-kernel dequant". |
| `compute-units-and-authoring.md:16,20,29` | ANE Conv engine accumulates fp32; ANE = fp16 only; CPU(BNNS) = fp32/fp16. |
| `ane-silicon-reference.md:33,40` | no int4 arithmetic lane (palette only); fp8 datapath H18-only. |
| `16-neural-accelerators.md` | NAX = GPU core count = 40 here; matrix fp16 ~1024 FLOPS/core/cycle (TZAKHARKO-NA). |
| `SOURCES.md` | BaseRT arXiv:2607.19438; Apple MLX-M5 post; Tungsten "Metal 4 matmul2d on M5 Max". |
| Third-party (with date) | Tungsten article (per `SOURCES.md:52`); Apple ML Research MLX-M5 post; Zakharko NA benchmark (access 2026-10-06). |

## 5. Cells we could NOT anchor (stated, not guessed)

- **The exact accumulate behaviour of the MPP `float` destination** (`half×half→float`): the header
  gives the *destination* type, and `relaxed_precision` names the tf32 trade — but whether an fp32
  destination accumulates in fp32 vs tf32 remains **INFERENCE** from the destination type. Partial
  measured evidence now exists: `nax_load_ab`'s fp16 pair gates the tensor arm against a double
  oracle at K=1024 and sees `rel ≈ 5e-7` (`maxabs 2.58e-06`) over 4096 sampled outputs, which is
  **inconsistent with a ≤10-bit-mantissa accumulator** (a tf32 or fp16 accumulator would show
  `rel` of order 1e-3–1e-2). That rules *out* tf32/fp16 accumulation; it does **not** prove fp32
  rather than anything wider, and it does not read the accumulate type directly. So the tensor arm's
  `accumulate=fp32` row label is *inference + this bound*, not a header read — do not upgrade the cell
  itself to a measurement.
- **Which of the 48 MPP rows are *fast*** (vs merely accepted): unmeasured. API-168's caveat stands.
- **ANE int16 accumulate**: LANDSCAPE-008 lists int16 as an accepted tensor type; the accumulate
  type for an int16 GEMM is not in any anchor we hold — left blank above rather than guessed.
- **fp8 datapath generation on this box**: the *catalogue* has it; `ane-silicon-reference.md` says
  H18-only and this box is h17c — so on-box fp8 is **not native**, but that is a source-scope claim,
  not our own measurement.

Anything marked INFERENCE above must carry that marking into any table that quotes it.
