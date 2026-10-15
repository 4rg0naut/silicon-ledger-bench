#!/usr/bin/env python3
"""One latency protocol, applied identically to every model. Emits schema-conformant rows.

Why this exists: the corpus has 142 latency figures measured under at least four different
conventions — different item counts, no stated warm-up, some per-embedding and some per-decision,
none normalised. The numbers were real; they were not comparable. This fixes the protocol so the
NEXT measurement is comparable, and re-measures what we can.

THE PROTOCOL (fixed; do not vary between models):
  * warm-up: 10 discarded calls
  * measured: 100 calls, back to back, one process
  * report:   p50, p95, min, and ms/token
  * the WORK UNIT is stated explicitly on every row
  * tokens are the padded sequence length actually fed to the graph, so ms/token is honest

`ms/token` is the point of the exercise. It is the only axis that means the same thing for an
embedder (S tokens per embedding), a decision model (S tokens per pass) and a reranker (S tokens per
pair), so it is the one number that can legitimately sit in a cross-class table.

requirements: `numpy` and the Core AI runtime (`coreai.runtime`). Neither ships with a stock
python3; if they are missing the script says so and exits non-zero instead of crashing.

usage:
    python tools/probes/latency_protocol.py --bundle <path> --seq-len 256 \
        --model "Laya English" --placement ANE --work-unit "per decision (all options, one pass)" \
        --id laya-en-ane-s256-protocol
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:  # degrade gracefully: reported below, not a traceback
    np = None

WARMUP = 10
MEASURED = 100


# CROSS-REPO-CONTRACT.md: project-wide schema id, stamped on every emitted row
SCHEMA_ID = "silicon-ledger/1"


def machine_identity() -> str:
    """Chip + unified-memory size of the host running this row (never hardcoded)."""
    chip = "unknown"
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, check=True).stdout.strip()
        if out:
            chip = out
    except Exception:
        chip = platform.processor() or chip
    try:
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, check=True).stdout.strip())
        return f"{chip}, {round(mem / (1 << 30))} GB"
    except Exception:
        return chip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--placement", default="ANE")
    ap.add_argument("--work-unit", required=True,
                    help="what ONE call produces, e.g. 'per embedding'")
    ap.add_argument("--id", required=True)
    ap.add_argument("--passes-per-unit", type=int, default=1,
                    help="graph calls per work unit; Von's NLI mapping needs N")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if np is None:
        print("latency_protocol: numpy is required but not installed for this interpreter.\n"
              "  install it (e.g. `python3 -m pip install numpy`) or use one that has it.",
              file=sys.stderr)
        return 2
    try:
        from coreai.runtime import AIModel, NDArray, ComputeUnitKind, SpecializationOptions
    except ImportError as exc:
        print(f"latency_protocol: the Core AI runtime is not importable here ({exc}).\n"
              "  this protocol drives graphs through `coreai.runtime`; install the runtime, "
              "then re-run.", file=sys.stderr)
        return 2

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result()

    unit = {"ANE": ComputeUnitKind.neural_engine, "GPU": ComputeUnitKind.gpu,
            "CPU": ComputeUnitKind.cpu}[args.placement]
    opts = SpecializationOptions.from_preferred_compute_unit_kind(unit())

    t0 = time.perf_counter()
    model = run(AIModel.load(args.bundle, specialization_options=opts))
    fn = model.load_function("main")
    load_s = time.perf_counter() - t0

    # inputs: this protocol measures the GRAPH, so feed it the graph's own shape
    S = args.seq_len
    names = [str(x) for x in getattr(fn, "input_names", [])] or \
            ["input_ids", "attention_mask", "selection", "qtype"]
    ins = {}
    for n in names:
        if n == "input_ids" or n == "attention_mask":
            ins[n] = NDArray(np.ones((1, S), dtype=np.int32))
        elif n == "selection":
            sel = np.zeros((1, 32, S), dtype=np.float16)
            sel[0, 0, 1] = 1.0
            ins[n] = NDArray(sel)
        elif n == "qtype":
            ins[n] = NDArray(np.array([2], dtype=np.int32))

    def call():
        o = run(fn(ins))
        del o

    for _ in range(WARMUP):
        call()
    lat = []
    for _ in range(MEASURED):
        t = time.perf_counter()
        call()
        lat.append((time.perf_counter() - t) * 1000.0)
    lat.sort()

    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    row = {
        "id": args.id,
        "model": args.model,
        "checkpoint": None,
        "params_m": None,
        "runtime": "Core AI",
        "placement": args.placement,
        "dtype": "fp16",
        "seq_len": S,
        "ane_regions": None,
        "benchmark": None,
        "split": None,
        "scope": f"{WARMUP} warm-up + {MEASURED} measured calls",
        "tiers": None,
        "metric": "latency",
        "value": round(p50, 3),
        "unit": "ms (p50)",
        "latency_ms": round(p50, 3),
        "latency_unit": args.work_unit,
        "hardware": machine_identity(),
        "load_factor": None,
        "energy_mw": None,
        "energy_rail": None,
        "energy_baseline": None,
        "provenance": "ours",
        "schema": SCHEMA_ID,
        "date": _dt.date.today().isoformat(),
        "command": "tools/probes/latency_protocol.py",
        # protocol-specific, recorded so a future run is comparable
        "protocol": {"warmup": WARMUP, "measured": MEASURED,
                     "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
                     "min_ms": round(lat[0], 3), "max_ms": round(lat[-1], 3),
                     "load_ms": round(load_s * 1000, 1),
                     "passes_per_unit": args.passes_per_unit,
                     "ms_per_call": round(p50, 3),
                     "ms_per_token_per_call": round(p50 / S, 5),
                     "ms_per_unit": round(p50 * args.passes_per_unit, 3),
                     "ms_per_token_per_unit": round(p50 * args.passes_per_unit / S, 5)},
    }
    print(json.dumps({k: row[k] for k in
                      ("id", "model", "placement", "seq_len", "value", "latency_unit")}, indent=1))
    print(json.dumps(row["protocol"], indent=1))
    if args.out:
        args.out.write_text(json.dumps(row, indent=1))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
