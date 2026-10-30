# LLM generation-loop capability matrix per host language (night 2026-10-27/28)

Question this answers (the actual product): *which is the leanest language that
can host a custom generation loop on MLX/Metal here?* Verified on M5 Max,
macOS 27.0.1, mlx-swift 0.32.3 (Cmlx.o exports **677** `mlx_*` C functions),
mlx 0.32.4 / mlx-lm 0.32.2, mlx-sys 0.6.0 (core v0.32.2).

Legend: ✅ reachable natively (used in our harness tonight) · 📜 exists but
hand-rolled work needed · ❌ not reachable (roll your own above the primitive).

| loop function | Python (mlx_lm) | Swift (mlx-swift-lm) | C (mlx-c) | Rust (mlx-sys) | ObjC (=C) |
|---|---|---|---|---|---|
| safetensors load | ✅ mx.load | ✅ Hub + LMModel | ✅ mlx_load_safetensors (CPU stream) | ✅ same via bind | ✅ |
| quantized matmul | ✅ | ✅ | ✅ mlx_quantized_matmul (mode affine) | ✅ | ✅ |
| MoE gather-qmm | ✅ rhs_indices | ✅ | ✅ mlx_gather_qmm (explicit lhs/rhs) | ✅ | ✅ |
| fused rms_norm / rope | ✅ fast.* | ✅ | ✅ fast_rms_norm / fast_rope | ✅ | ✅ |
| fused SDPA + KV update | ✅ | ✅ (session KV reuse) | ✅ fast_sdpa; KV concat hand-rolled | ✅ same | ✅ |
| sampling (argmax/top-p) | ✅ sample utils | ✅ generation params | ✅ argmax_axis (+ roll top-p) | ✅ | ✅ |
| step eval + device mem | ✅ | ✅ | ✅ mlx_eval, peak/active/cache, wired limit | ✅ | ✅ |
| tokenizer (BPE, chat template) | ✅ HF path | ✅ MLXLMCommon (tokenizer+chat+detok native) | ❌ hand-roll or shell out | ❌ same | ❌ same |
| generation loop / EOS / streaming | ✅ | ✅ (generate stream) | 📜 ~200 lines tonight | 📜 ~250 lines | ✅ rehosted free |
| prefill batching | ✅ | ✅ | 📜 not built (LF-09) | 📜 | 📜 |
| long MoE stability (768+ steps) | ✅ | ✅ | ❌ LF-06 | ❌ LF-06 | ❌ (inherits C) |

## Conclusions

- **Everything the loop needs on-GPU is reachable from plain C** (that was the
  C3 claim; it holds — every ✅ cell above executed inside our C harness, and
  the rows carry the op lists). What is *not* reachable from C is text-side:
  tokenizer, chat template, detokenizer — identical gap in Rust and ObjC.
- Swift closes the text-side gap natively (MLXLMCommon) *and* is the only
  hand-rolled-ish full loop that is both fastest and MoE-stable (355/152/146
  t/s). C/Rust loops are tonight ~2× behind the identical Python graph and die
  on long MoE runs (LF-06).
- ObjC adds nothing over C (same ABI, same source compiled with
  `-x objective-c`, ran: 245 t/s @1.7B) — it exists only to prove the C ABI
  rehost is one flag.
