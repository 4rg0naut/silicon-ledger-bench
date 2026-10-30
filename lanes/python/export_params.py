#!/usr/bin/env python3
"""Emit C-parseable params.txt from a model snapshot config.json.

The C/Rust lanes load the ORIGINAL snapshot safetensors (same bytes as
Python), so weight export is not needed; only flat config scalars are.
"""
import argparse
import json
import os

CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def snapshot(model_id):
    org = "models--" + model_id.replace("/", "--")
    snapdir = os.path.join(CACHE, org, "snapshots")
    return os.path.join(snapdir, sorted(os.listdir(snapdir))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    snap = snapshot(a.model)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    keys = [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "intermediate_size",
        "vocab_size", "max_position_embeddings", "decoder_sparse_step",
        "num_experts", "num_experts_per_tok", "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ]
    sharded = any(x.endswith(".safetensors") and "0000" in x for x in os.listdir(snap))
    with open(a.out, "w") as f:
        f.write(f"rope_theta {cfg.get('rope_theta', 1000000.0):.1f}\n")
        f.write(f"rms_eps {cfg.get('rms_norm_eps', 1e-6):.12f}\n")
        f.write(f"model_type {cfg.get('model_type', 'qwen3')}\n")
        quant = cfg.get("quantization") or {}
        f.write(f"quant_group {quant.get('group_size', 64)}\n")
        f.write(f"quant_bits {quant.get('bits', 4)}\n")
        if sharded:
            n = len([x for x in os.listdir(snap) if x.endswith(".safetensors")])
            for i in range(1, n + 1):
                f.write(f"shard {os.path.join(snap, f'model-{i:05d}-of-{n:05d}.safetensors')}\n")
        else:
            f.write(f"safetensors {os.path.join(snap, 'model.safetensors')}\n")
        for k in keys:
            if k in cfg:
                f.write(f"{k} {cfg[k]}\n")
    print(open(a.out).read())


if __name__ == "__main__":
    main()
