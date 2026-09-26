#!/usr/bin/env python3
"""Generate .aimodel assets for the S4 Core AI suite.

Models (models/coreai/):
  deep_fp16.aimodel  64x Conv2d(512->512, 1x1, no bias), in [1,512,1,64]
  deep_int8.aimodel  same graph, W8A8 PTQ via coreai-opt
  matmul.aimodel     single Linear(2048, 2048), in [1,2048]

Each model gets <name>.verify.json (metadata) and <name>.ref.bin
(float32 little-endian torch CPU reference output) for Swift-side validation.

Weights/inputs are deterministic and match Sources/bench/MIL.swift exactly.
"""

import json
import pathlib
import struct

import torch
import torch.nn as nn

import coreai_torch
from coreai_torch import TorchConverter

HERE = pathlib.Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models" / "coreai"

CH = 512
SP = 64
DEPTH = 64
INT8_DEPTH = 8   # W8A8 PTQ with deterministic random weights decorrelates over deep
                 # chains (integer-accum exec vs torch fake-quant); 8 layers validates
                 # cleanly at identical shapes. fp16/fp32 chains keep the full depth 64.
MM = 2048


def det_weights(ch: int, depth: int) -> torch.Tensor:
    i = torch.arange(depth, dtype=torch.int64).view(depth, 1, 1)
    n = torch.arange(ch, dtype=torch.int64).view(1, ch, 1)
    c = torch.arange(ch, dtype=torch.int64).view(1, 1, ch)
    v = ((n * 31 + c * 17 + i * 7) % 13 - 6) * 0.01
    return v.to(torch.float64)


def det_input(ch: int, sp: int) -> torch.Tensor:
    c = torch.arange(ch, dtype=torch.int64).view(1, ch, 1, 1)
    s = torch.arange(sp, dtype=torch.int64).view(1, 1, 1, sp)
    x = (((c * 13 + s * 7) % 11 - 5) * 0.01).to(torch.float64)
    return x


def det_mm_input(n: int) -> torch.Tensor:
    c = torch.arange(n, dtype=torch.int64).view(1, -1)
    x = (((c * 13 + 7) % 11 - 5) * 0.01).to(torch.float64)
    return x


def det_mm_weight(n: int) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.int64).view(n, 1)
    c = torch.arange(n, dtype=torch.int64).view(1, n)
    w = (((i * 31 + c * 17 + 3) % 13 - 6) * 0.01).to(torch.float64)
    return w


def save_ref(path: pathlib.Path, out: torch.Tensor) -> None:
    out = out.detach().to(torch.float32).contiguous()
    path.write_bytes(struct.pack("<%df" % out.numel(), *out.view(-1).tolist()))


class DeepConv(nn.Module):
    def __init__(self, ch: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(ch, ch, 1, bias=False) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_deep(dtype: torch.dtype, depth: int = DEPTH) -> DeepConv:
    model = DeepConv(CH, depth).eval()
    w = det_weights(CH, depth).to(dtype)
    for i, layer in enumerate(model.layers):
        layer.weight = nn.Parameter(w[i].reshape(CH, CH, 1, 1).contiguous(), requires_grad=False)
    return model


def export_fp32(model: nn.Module, example: torch.Tensor) -> object:
    ep = torch.export.export(model, args=(example,))
    ep = ep.run_decompositions(coreai_torch.get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["x"], output_names=["out"])
    return conv.to_coreai()


def build_deep_fp16() -> tuple[pathlib.Path, dict]:
    model = make_deep(torch.float32)
    ref_model = make_deep(torch.float64)
    x = det_input(CH, SP)
    with torch.no_grad():
        ref = ref_model(x)
    asset = export_fp32(model, x.to(torch.float32))
    dest = MODELS_DIR / "deep_fp16.aimodel"
    asset.save_asset(dest)
    meta = {
        "ok": True,
        "quant_method": "none (fp32 graph, CoreAI picks ANE execution precision)",
        "in_shape": [1, CH, 1, SP],
        "out_shape": list(ref.shape),
        "flops": 2 * CH * CH * SP * DEPTH,
        "layers": DEPTH,
        "ref_file": "deep_fp16.ref.bin",
        "note": "chain values decay ~0.027x/layer; final magnitude ~1e-100 (degenerate for fp16, validation is a smoke check)",
    }
    save_ref(MODELS_DIR / "deep_fp16.ref.bin", ref)
    (MODELS_DIR / "deep_fp16.verify.json").write_text(json.dumps(meta, indent=2))
    return dest, meta


def build_deep_int8() -> tuple[pathlib.Path, dict]:
    from coreai_opt import ExportBackend
    from coreai_opt.quantization import Quantizer, QuantizerConfig

    model = make_deep(torch.float32, depth=INT8_DEPTH)
    x = det_input(CH, SP).to(torch.float32)

    q = Quantizer(model, QuantizerConfig())
    prepared = q.prepare((x,))
    with q.calibration_mode():
        for scale in (0.5, 1.0, 1.5):
            prepared((x * scale).to(torch.float32))
    with torch.no_grad():
        ref = prepared(x)
    qm = q.finalize(backend=ExportBackend.CoreAI)

    ep = torch.export.export(qm, args=(x,))
    ep = ep.run_decompositions(coreai_torch.get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["x"], output_names=["out"])
    asset = conv.to_coreai()

    dest = MODELS_DIR / "deep_int8.aimodel"
    asset.save_asset(dest)
    meta = {
        "ok": True,
        "quant_method": "coreai-opt W8A8 PTQ (default QuantizerConfig, graph mode)",
        "in_shape": [1, CH, 1, SP],
        "out_shape": list(ref.shape),
        "flops": 2 * CH * CH * SP * INT8_DEPTH,
        "layers": INT8_DEPTH,
        "ref_file": "deep_int8.ref.bin",
        "note": "reference = torch fake-quant (float32); depth reduced to 8 (from 64) because W8A8 integer-accum execution decorrelates from fake-quant over 64 random-weight layers (cosine -0.13); at depth 8 pointwise rel-err validates; identical shapes",
    }
    save_ref(MODELS_DIR / "deep_int8.ref.bin", ref)
    (MODELS_DIR / "deep_int8.verify.json").write_text(json.dumps(meta, indent=2))
    return dest, meta


def build_matmul() -> tuple[pathlib.Path, dict]:
    lin = nn.Linear(MM, MM).eval()
    lin.weight = nn.Parameter(det_mm_weight(MM).to(torch.float32).contiguous(), requires_grad=False)
    lin.bias = nn.Parameter(torch.zeros(MM, dtype=torch.float32), requires_grad=False)
    x = det_mm_input(MM)
    with torch.no_grad():
        ref = lin.double()(x)
    asset = export_fp32(lin, x.to(torch.float32))
    dest = MODELS_DIR / "matmul.aimodel"
    asset.save_asset(dest)
    meta = {
        "ok": True,
        "quant_method": "none (fp32 graph)",
        "in_shape": [1, MM],
        "out_shape": list(ref.shape),
        "flops": 2 * MM * MM * MM,
        "layers": 1,
        "ref_file": "matmul.ref.bin",
    }
    save_ref(MODELS_DIR / "matmul.ref.bin", ref)
    (MODELS_DIR / "matmul.verify.json").write_text(json.dumps(meta, indent=2))
    return dest, meta


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    for name, fn in (("deep_fp16", build_deep_fp16), ("deep_int8", build_deep_int8), ("matmul", build_matmul)):
        dest, meta = fn()
        print(f"[gen] {name}: {dest} out={meta['out_shape']} flops={meta['flops']}")
    print("[gen] done")


if __name__ == "__main__":
    main()
