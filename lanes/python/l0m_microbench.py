#!/usr/bin/env python3
"""L0m: Python microbench mirror of lanes/c/qwen3_decode.c.

Runs the hand-rolled Qwen3 decode graph with the SAME op choices the C lane
uses (quantized_matmul transpose=True, dequantized embedding via
mx.take axis=0, fast_rms_norm, fast_rope, fast_sdpa, sigmoid+argsort MoE
routing, concat KV) so the C/Rust numbers can be attributed to the host
language, not the graph. Times per-step eval wall time.
"""
import argparse
import json
import math
import time

import mlx.core as mx
import mlx.nn as nn

def load_params(path):
    cfg = {"shards": [], "model_type": "qwen3"}
    for line in open(path):
        k, _, v = line.strip().partition(" ")
        if not v:
            continue
        if k in ("safetensors", "shard"):
            cfg["shards"].append(v)
        elif k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                   "num_key_value_heads", "head_dim", "intermediate_size",
                   "vocab_size", "num_experts", "num_experts_per_tok",
                   "moe_intermediate_size", "quant_group", "quant_bits"):
            cfg[k] = int(v)
        elif k in ("rope_theta", "rms_eps"):
            cfg[k] = float(v)
        elif k == "model_type":
            cfg["model_type"] = v
    cfg["layers"] = cfg["num_hidden_layers"]
    cfg["hidden"] = cfg["hidden_size"]
    cfg["n_heads"] = cfg["num_attention_heads"]
    cfg["n_kv"] = cfg["num_key_value_heads"]
    cfg["n_experts"] = cfg.get("num_experts", 0)
    cfg["top_k"] = cfg.get("num_experts_per_tok", 0)
    cfg["moe"] = "moe" in cfg["model_type"]
    cfg.setdefault("rms_eps", 1e-5)
    cfg.setdefault("rope_theta", 1_000_000.0)
    cfg.setdefault("quant_group", 64)
    cfg.setdefault("quant_bits", 4)
    return cfg


def load_weights(paths):
    ws = {}
    for p in paths:
        ws.update(dict(mx.load(p)))
    return ws


class Ctx:
    pass


def build(ctx, ws, cfg):
    ew = ws["model.embed_tokens.weight"]
    if "model.embed_tokens.scales" in ws:
        ctx.embed = mx.dequantize(
            ew, ws["model.embed_tokens.scales"],
            ws.get("model.embed_tokens.biases"), cfg["quant_group"],
            cfg["quant_bits"], "affine")
    else:
        ctx.embed = ew
    if "lm_head.weight" in ws and "lm_head.scales" not in ws:
        ctx.headT = ws["lm_head.weight"].transpose(1, 0)
        ctx.head_q = False
    elif "lm_head.weight" in ws:
        ctx.head_q = "qmm"
        ctx.headT = None
    elif "model.embed_tokens.scales" in ws:
        ctx.head_q = "qtie"
        ctx.headT = None
    else:
        ctx.headT = ctx.embed.T
        ctx.head_q = False
    if ctx.headT is not None:
        mx.eval(ctx.embed, ctx.headT)
    else:
        mx.eval(ctx.embed)


def qmm(ctx, ws, x, pre, cfg):
    return mx.quantized_matmul(
        x, ws[pre + ".weight"], ws[pre + ".scales"], ws[pre + ".biases"],
        transpose=True, group_size=cfg["quant_group"], bits=cfg["quant_bits"],
        mode="affine")


def qmm_idx(ctx, ws, x, pre, idx, cfg):
    return mx.gather_qmm(
        x, ws[pre + ".weight"], ws[pre + ".scales"], ws[pre + ".biases"],
        rhs_indices=idx, transpose=True, group_size=cfg["quant_group"],
        bits=cfg["quant_bits"], mode="affine")


def decode_step(ctx, ws, cfg, tok, pos, n=1):
    h = mx.take(ctx.embed, tok, axis=0)
    nq, nkv = cfg["n_heads"] * cfg["head_dim"], cfg["n_kv"] * cfg["head_dim"]
    scale = 1.0 / math.sqrt(cfg["head_dim"])
    for l in range(cfg["layers"]):
        p = f"model.layers.{l}"
        r = mx.fast.rms_norm(h, ws[f"{p}.input_layernorm.weight"], cfg["rms_eps"])
        q = qmm(ctx, ws, r, f"{p}.self_attn.q_proj", cfg).reshape(1, n, cfg["n_heads"], cfg["head_dim"])
        k = qmm(ctx, ws, r, f"{p}.self_attn.k_proj", cfg).reshape(1, n, cfg["n_kv"], cfg["head_dim"])
        v = qmm(ctx, ws, r, f"{p}.self_attn.v_proj", cfg).reshape(1, n, cfg["n_kv"], cfg["head_dim"])
        q = mx.fast.rms_norm(q, ws[f"{p}.self_attn.q_norm.weight"], 1e-6)
        k = mx.fast.rms_norm(k, ws[f"{p}.self_attn.k_norm.weight"], 1e-6)
        q = mx.fast.rope(q, cfg["head_dim"], traditional=False, base=cfg["rope_theta"],
                         scale=1.0, offset=pos)
        k = mx.fast.rope(k, cfg["head_dim"], traditional=False, base=cfg["rope_theta"],
                         scale=1.0, offset=pos)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        if ctx.kcache[l] is None:
            ctx.kcache[l], ctx.vcache[l] = k, v
        else:
            ctx.kcache[l] = mx.concatenate([ctx.kcache[l], k], axis=2)
            ctx.vcache[l] = mx.concatenate([ctx.vcache[l], v], axis=2)
        attn = mx.fast.scaled_dot_product_attention(q, ctx.kcache[l], ctx.vcache[l],
                                                    scale=scale, mask="causal")
        attn = attn.transpose(0, 2, 1, 3).reshape(n, nq)
        h = h + qmm(ctx, ws, attn, f"{p}.self_attn.o_proj", cfg)
        r2 = mx.fast.rms_norm(h, ws[f"{p}.post_attention_layernorm.weight"], cfg["rms_eps"])
        if not cfg["moe"]:
            g = mx.sigmoid(qmm(ctx, ws, r2, f"{p}.mlp.gate_proj", cfg))
            up = qmm(ctx, ws, r2, f"{p}.mlp.up_proj", cfg)
            h = h + qmm(ctx, ws, g * up, f"{p}.mlp.down_proj", cfg)
        else:
            route = qmm(ctx, ws, r2, f"{p}.mlp.gate", cfg).reshape(n, cfg["n_experts"])
            scores = mx.sigmoid(route)
            order = mx.argsort(-scores, axis=-1)
            idx = order[:, : cfg["top_k"]]
            idx_flat = idx.reshape(-1)
            rows = mx.broadcast_to(mx.arange(n).reshape(n, 1),
                                   (n, cfg["top_k"])).reshape(-1) * cfg["n_experts"]
            wsel = mx.take(scores.reshape(-1), idx_flat + rows).reshape(n, cfg["top_k"])
            wsel = wsel / mx.sum(wsel, axis=1, keepdims=True)
            xb = mx.broadcast_to(r2.reshape(n, 1, cfg["hidden"]),
                                 (n, cfg["top_k"], cfg["hidden"]))
            xb = xb.reshape(n * cfg["top_k"], 1, cfg["hidden"])
            g = qmm_idx(ctx, ws, xb, f"{p}.mlp.switch_mlp.gate_proj", idx_flat, cfg)
            u = qmm_idx(ctx, ws, xb, f"{p}.mlp.switch_mlp.up_proj", idx_flat, cfg)
            act = mx.sigmoid(g) * g * u
            e = qmm_idx(ctx, ws, act, f"{p}.mlp.switch_mlp.down_proj", idx_flat, cfg)
            e = e.reshape(n, cfg["top_k"], cfg["hidden"]) * wsel[..., None]
            h = h + mx.sum(e, axis=1)
    hn = mx.fast.rms_norm(h, ws["model.norm.weight"], cfg["rms_eps"])
    if ctx.head_q == "qmm":
        logits = qmm(ctx, ws, hn, "lm_head", cfg)
    elif ctx.head_q == "qtie":
        logits = mx.quantized_matmul(
            hn, ws["model.embed_tokens.weight"], ws["model.embed_tokens.scales"],
            ws["model.embed_tokens.biases"], transpose=True,
            group_size=cfg["quant_group"], bits=cfg["quant_bits"], mode="affine")
    else:
        logits = hn @ ctx.headT
    return mx.argmax(logits[-1:], axis=-1)


PROMPT_SIZES = (32, 512, 2048)


def filler_ids(n, vocab):
    step = 2654435761
    return mx.array([(i * step + 7) % (vocab - 1000) + 10 for i in range(n)])


def prefill_bench(ctx, ws, cfg, sizes=PROMPT_SIZES, runs=3):
    out = {}
    vocab = cfg["vocab_size"]
    for n in sizes:
        ids = filler_ids(n, vocab)
        tps = []
        ms = []
        for run in range(runs + 1):
            ctx.kcache = [None] * cfg["layers"]
            ctx.vcache = [None] * cfg["layers"]
            t0 = time.perf_counter()
            tok = decode_step(ctx, ws, cfg, ids, 0, n=n)
            mx.eval(tok)
            dt = time.perf_counter() - t0
            if run > 0:
                tps.append(n / dt)
                ms.append(dt * 1000)
        out[str(n)] = {"prompt_tokens": n,
                       "wall_tps": round(sum(tps) / len(tps), 2),
                       "mean_ms": round(sum(ms) / len(ms), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_params(args.params)
    ws = load_weights(cfg["shards"])

    ctx = Ctx()
    build(ctx, ws, cfg)
    ctx.kcache = [None] * cfg["layers"]
    ctx.vcache = [None] * cfg["layers"]

    tok = mx.array([151644])
    steps = []
    for run in range(args.runs + 1):
        ctx.kcache = [None] * cfg["layers"]
        ctx.vcache = [None] * cfg["layers"]
        tok = mx.array([151644])
        for t in range(args.tokens):
            t0 = time.perf_counter()
            tok = decode_step(ctx, ws, cfg, tok, t)
            mx.eval(tok)
            dt = (time.perf_counter() - t0) * 1000
            if run > 0:
                steps.append(dt)
    steps.sort()
    mean = sum(steps) / len(steps)
    pf = prefill_bench(ctx, ws, cfg)
    row = {
        "lane": "l0m", "build_mode": "release", "model_kind": cfg["model_type"],
        "hidden": cfg["hidden"], "layers": cfg["layers"],
        "decode_steps": len(steps), "mean_step_ms": round(mean, 3),
        "p50_step_ms": round(steps[len(steps) // 2], 3),
        "p95_step_ms": round(steps[int(len(steps) * 0.95)], 3),
        "mean_decode_tps_wall": round(1000 / mean, 2),
        "prefill": pf,
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }
    line = json.dumps(row)
    print(line)
    if args.out:
        open(args.out, "w").write(line + "\n")


if __name__ == "__main__":
    main()
