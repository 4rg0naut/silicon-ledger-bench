#!/usr/bin/env python3
"""Export MLX models to raw fp16 blobs + C-parseable manifest for C/Rust lanes.

Quantized (4bit) weights are dequantized to fp16. Manifest is line-based:
  <n_tensors>
  <name> <dtype> <dim0> <dim1>... <offset> <nbytes>
so the C lane needs no JSON parser. config.json is copied verbatim; rope
cos/sin caches are precomputed and exported too (keeps rope out of the
per-token op-count question being measured).
"""
import argparse
import json
import os

import mlx.core as mx
from mlx_lm import load

CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def snapshot(model_id):
    org = "models--" + model_id.replace("/", "--")
    snapdir = os.path.join(CACHE, org, "snapshots")
    snaps = os.listdir(snapdir)
    return os.path.join(snapdir, snaps[0])


def flatten(params, prefix=""):
    out = {}
    for k, v in params:
        if isinstance(v, dict):
            out.update(flatten(v.items(), prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rope-positions", type=int, default=1024)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    model, tokenizer = load(a.model)
    flat = flatten(model.parameters().items())

    cfg = json.load(open(os.path.join(snapshot(a.model), "config.json")))

    # rope caches for the given max positions
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    theta = cfg.get("rope_theta", 1000000.0)
    pos = mx.arange(a.rope_positions)
    inv = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = pos[:, None].as_type(mx.float32) * inv[None, :]
    cos = mx.repeat(mx.cos(freqs), 2, axis=-1).as_type(mx.float16)
    sin = mx.repeat(mx.sin(freqs), 2, axis=-1).as_type(mx.float16)
    flat["__cos"] = cos
    flat["__sin"] = sin

    manifest = []
    offset = 0
    blobs = []
    for name, arr in sorted(flat.items()):
        mx.eval(arr)
        fp = arr.as_type(mx.float16)
        raw = fp.tobytes()
        blobs.append(raw)
        manifest.append((name, "f16", list(fp.shape), offset, len(raw)))
        offset += len(raw)

    with open(os.path.join(a.out, "w.bin"), "wb") as f:
        for b in blobs:
            f.write(b)
    with open(os.path.join(a.out, "manifest.txt"), "w") as f:
        f.write(f"{len(manifest)}\n")
        for name, dt, shape, off, nb in manifest:
            f.write(f"{name} {dt} {len(shape)} {' '.join(map(str, shape))} {off} {nb}\n")

    cfg_out = dict(cfg)
    cfg_out["vocab_size_exported"] = int(flat.get("lm_head.weight", mx.zeros((1,))).shape[0])
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=1)

    print(json.dumps({"model": a.model, "tensors": len(manifest), "bytes": offset}))


if __name__ == "__main__":
    main()
