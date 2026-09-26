#!/usr/bin/env python3
"""Generate the S5 legacy Core ML baseline model (.mlpackage).

conv_512x64_d8: 8x Conv2d(512->512, 1x1, no bias), input [1,512,1,64] fp16.
Weights/inputs use the deterministic formulas from tools/gen_models.py
(det_weights / det_input), which match Sources/bench/MIL.swift exactly.
Validates that the coremltools path runs on this machine; Swift-side
(S5_CoreML.swift) re-validates against ref.bin with compute units forced
through the neural engine preference.
"""
import json
import pathlib
import struct

import torch
import torch.nn as nn

import coremltools as ct

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "models" / "coreml"

CH = 512
SP = 64
DEPTH = 8
MLPACKAGE = "conv_512x64_d8.mlpackage"


def det_weights(ch: int, depth: int) -> torch.Tensor:
    i = torch.arange(depth, dtype=torch.int64).view(depth, 1, 1)
    n = torch.arange(ch, dtype=torch.int64).view(1, ch, 1)
    c = torch.arange(ch, dtype=torch.int64).view(1, 1, ch)
    v = ((n * 31 + c * 17 + i * 7) % 13 - 6) * 0.01
    return v.to(torch.float64)


def det_input(ch: int, sp: int) -> torch.Tensor:
    c = torch.arange(ch, dtype=torch.int64).view(1, ch, 1, 1)
    s = torch.arange(sp, dtype=torch.int64).view(1, 1, 1, sp)
    return (((c * 13 + s * 7) % 11 - 5) * 0.01).to(torch.float64)


class DeepConv(nn.Module):
    def __init__(self, ch: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(ch, ch, 1, bias=False) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = DeepConv(CH, DEPTH).eval()
    w = det_weights(CH, DEPTH)
    for i, layer in enumerate(model.layers):
        layer.weight = nn.Parameter(w[i].reshape(CH, CH, 1, 1).contiguous(), requires_grad=False)
    x = det_input(CH, SP)
    with torch.no_grad():
        ref = model(x)
    ep = torch.export.export(model, args=(x.to(torch.float32),))
    ep = ep.run_decompositions({})
    mlmodel = ct.convert(
        ep,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS27 if hasattr(ct.target, "macOS27") else ct.target.macOS14,
        inputs=[ct.TensorType(shape=(1, CH, 1, SP), name="x")],
        compute_precision=ct.precision.FLOAT16,
    )
    dest = OUT_DIR / MLPACKAGE
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    mlmodel.save(dest)
    ref32 = ref.to(torch.float32).contiguous()
    (OUT_DIR / "conv_512x64_d8.ref.bin").write_bytes(
        struct.pack("<%df" % ref32.numel(), *ref32.view(-1).tolist()))
    (OUT_DIR / "conv_512x64_d8.verify.json").write_text(json.dumps({
        "ok": True,
        "in_shape": [1, CH, 1, SP],
        "out_shape": list(ref.shape),
        "flops": 2 * CH * CH * SP * DEPTH,
        "layers": DEPTH,
        "ref_file": "conv_512x64_d8.ref.bin",
        "note": "fp16 mlprogram weights (coremltools FLOAT16); CPU fp64 reference",
    }, indent=2))
    print(f"[gen-coreml] {dest} ref magnitude max={ref32.abs().max().item():.3e}")


if __name__ == "__main__":
    main()
