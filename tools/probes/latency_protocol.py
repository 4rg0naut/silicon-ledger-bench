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

requirements: none for the default path — `--self-test` needs only the standard library that ships
with python3. The optional `--bundle` path drives Core AI graphs and needs that runtime and its
array package installed; when either is missing that path prints one clear line and exits non-zero,
never a traceback. The timing protocol itself (`measure`) is identical on both paths.

usage:
    # dependency-free: prove the protocol runs here, emit a valid row
    python tools/probes/latency_protocol.py --self-test

    # the optional Core AI path
    python tools/probes/latency_protocol.py --bundle <path> --seq-len 256 \
        --model "Laya English" --placement ANE --work-unit "per decision (all options, one pass)" \
        --id laya-en-ane-s256-protocol
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

WARMUP = 10
MEASURED = 100

# steps in the built-in self-test workload (see `_self_test_workload`)
SELF_TEST_STEPS = 200_000

# CROSS-REPO-CONTRACT.md: project-wide schema id, stamped on every emitted row
SCHEMA_ID = "silicon-ledger/1"


def _sysctl(key: str) -> str | None:
    """`sysctl -n <key>`, or None if the key/host doesn't answer. Nothing is hardcoded."""
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
    except Exception:
        return None
    return out.stdout.strip() or None


def machine_identity() -> str:
    """Chip + unified-memory size of the host running this row (never hardcoded)."""
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
    mem = _sysctl("hw.memsize")
    if mem and mem.isdigit():
        return f"{chip}, {round(int(mem) / (1 << 30))} GB"
    return chip


def host_facts() -> dict:
    """chip / physical cores / OS version + build, read straight from sysctl (never hardcoded)."""
    cores = _sysctl("hw.physicalcpu")
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        "cores": int(cores) if cores and cores.isdigit() else None,
        "os": _sysctl("kern.osproductversion"),
        "os_build": _sysctl("kern.osversion"),
    }


# ---------------------------------------------------------------------------------------------
# the protocol, in plain Python (no array library anywhere near it)
# ---------------------------------------------------------------------------------------------

def _percentile(sorted_ms: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list — the convention this file always used:
    index = min(n - 1, int(q * n))."""
    return sorted_ms[min(len(sorted_ms) - 1, int(q * len(sorted_ms)))]


def summarize(lat_ms: list[float]) -> dict:
    """p50 / p95 / min / max / mean of a list of per-call times, computed by hand with
    `statistics` + `sorted`. No third-party module is involved (and never was required)."""
    lat = sorted(lat_ms)
    return {
        "n": len(lat),
        "p50_ms": round(statistics.median(lat), 3),
        "p95_ms": round(_percentile(lat, 0.95), 3),
        "min_ms": round(lat[0], 3),
        "max_ms": round(lat[-1], 3),
        "mean_ms": round(statistics.fmean(lat), 3),
    }


def measure(call, *, seq_len: int, passes_per_unit: int = 1,
            warmup: int = WARMUP, measured: int = MEASURED) -> dict:
    """THE PROTOCOL: discard `warmup` calls, then time `measured` back to back in one process.
    `call` is any zero-argument callable; it decides what a "call" is. Returns the summary plus
    the ms/token and ms/unit derivations every row carries."""
    for _ in range(warmup):
        call()
    lat = []
    for _ in range(measured):
        t = time.perf_counter()
        call()
        lat.append((time.perf_counter() - t) * 1000.0)
    stats = summarize(lat)
    p50 = stats["p50_ms"]
    stats.update({
        "ms_per_call": p50,
        "ms_per_token_per_call": round(p50 / seq_len, 5),
        "ms_per_unit": round(p50 * passes_per_unit, 3),
        "ms_per_token_per_unit": round(p50 * passes_per_unit / seq_len, 5),
    })
    return stats


def build_row(*, row_id, model, runtime, placement, dtype, work_unit, seq_len,
              passes_per_unit: int, stats: dict, load_ms: float, command: str) -> dict:
    """Assemble the schema row. Same shape on both paths; only the workload differs."""
    host = host_facts()
    p50 = stats["p50_ms"]
    return {
        "id": row_id,
        "model": model,
        "checkpoint": None,
        "params_m": None,
        "runtime": runtime,
        "placement": placement,
        "dtype": dtype,
        "seq_len": seq_len,
        "ane_regions": None,
        "benchmark": None,
        "split": None,
        "scope": f"{WARMUP} warm-up + {MEASURED} measured calls",
        "tiers": None,
        "metric": "latency",
        "value": p50,
        "unit": "ms (p50)",
        "latency_ms": p50,
        "latency_unit": work_unit,
        "hardware": machine_identity(),
        "chip": host["chip"],
        "cores": host["cores"],
        "os": host["os"],
        "os_build": host["os_build"],
        "load_factor": None,
        "energy_mw": None,
        "energy_rail": None,
        "energy_baseline": None,
        "provenance": "ours",
        "schema": SCHEMA_ID,
        "date": _dt.date.today().isoformat(),
        "command": command,
        # protocol-specific, recorded so a future run is comparable
        "protocol": {"warmup": WARMUP, "measured": MEASURED, "n": stats["n"],
                     "p50_ms": p50, "p95_ms": stats["p95_ms"], "min_ms": stats["min_ms"],
                     "max_ms": stats["max_ms"], "mean_ms": stats["mean_ms"],
                     "load_ms": load_ms,
                     "passes_per_unit": passes_per_unit,
                     "ms_per_call": stats["ms_per_call"],
                     "ms_per_token_per_call": stats["ms_per_token_per_call"],
                     "ms_per_unit": stats["ms_per_unit"],
                     "ms_per_token_per_unit": stats["ms_per_token_per_unit"]},
    }


def _emit(row: dict, out: Path | None) -> None:
    print(json.dumps(row, indent=1))
    if out:
        out.write_text(json.dumps(row, indent=1))
        print(f"  wrote {out}")


# ---------------------------------------------------------------------------------------------
# default path: a fixed workload with no dependencies at all
# ---------------------------------------------------------------------------------------------

def _self_test_workload(steps: int = SELF_TEST_STEPS):
    """A fixed, pure-Python unit of work. It is not a benchmark of anything real; it exists so
    `--self-test` can prove the protocol runs on a bare interpreter and emit a valid row."""
    def call() -> int:
        acc = 0
        for i in range(steps):
            acc = (acc * 31 + i) & 0xFFFFFFFF
        return acc
    return call


def self_test(args) -> int:
    stats = measure(_self_test_workload(), seq_len=1, passes_per_unit=args.passes_per_unit)
    row = build_row(
        row_id=args.id or "self-test",
        model="self-test (built-in, no model)",
        runtime="stdlib (pure python)",
        placement="CPU",
        dtype="int",
        work_unit=f"per self-test call ({SELF_TEST_STEPS}-step pure-Python loop)",
        seq_len=1,
        passes_per_unit=args.passes_per_unit,
        stats=stats,
        load_ms=0.0,
        command="tools/probes/latency_protocol.py --self-test",
    )
    _emit(row, args.out)
    return 0


# ---------------------------------------------------------------------------------------------
# optional path: a Core AI bundle. Every non-stdlib import lives inside this one function.
# ---------------------------------------------------------------------------------------------

def run_coreai(args) -> int:
    missing = [name for name, val in (("--seq-len", args.seq_len), ("--model", args.model),
                                      ("--work-unit", args.work_unit), ("--id", args.id))
               if val is None]
    if missing:
        print("latency_protocol: the Core AI path also needs " + ", ".join(missing) + ".",
              file=sys.stderr)
        return 2

    # `numpy` is imported HERE and not at module scope: the Core AI runtime's NDArray wraps
    # numeric arrays as graph inputs, so this one optional path needs it — nothing else does.
    try:
        import numpy as np
    except ImportError:
        print("latency_protocol: the Core AI path needs numpy (coreai.runtime.NDArray wraps "
              "numeric arrays); it is not installed for this interpreter.\n"
              "  install it, or run the dependency-free protocol with `--self-test`.",
              file=sys.stderr)
        return 2
    try:
        from coreai.runtime import AIModel, NDArray, ComputeUnitKind, SpecializationOptions
    except ImportError as exc:
        print(f"latency_protocol: the Core AI runtime is not importable here ({exc}).\n"
              "  this path drives graphs through `coreai.runtime`; install the runtime, "
              "then re-run — or use `--self-test` for the dependency-free protocol.",
              file=sys.stderr)
        return 2

    import asyncio
    import threading

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

    stats = measure(call, seq_len=S, passes_per_unit=args.passes_per_unit)
    row = build_row(
        row_id=args.id,
        model=args.model,
        runtime="Core AI",
        placement=args.placement,
        dtype="fp16",
        work_unit=args.work_unit,
        seq_len=S,
        passes_per_unit=args.passes_per_unit,
        stats=stats,
        load_ms=round(load_s * 1000, 1),
        command="tools/probes/latency_protocol.py",
    )
    print(json.dumps({k: row[k] for k in
                      ("id", "model", "placement", "seq_len", "value", "latency_unit")}, indent=1))
    print(json.dumps(row["protocol"], indent=1))
    if args.out:
        args.out.write_text(json.dumps(row, indent=1))
        print(f"  wrote {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="One latency protocol (10 warm-up, 100 measured, p50/p95/min + ms/unit).")
    ap.add_argument("--self-test", action="store_true",
                    help="time a fixed built-in workload and print a row; needs nothing installed")
    ap.add_argument("--bundle", type=Path, default=None,
                    help="Core AI bundle to drive (the optional, non-stdlib path)")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--placement", default="ANE")
    ap.add_argument("--work-unit", default=None,
                    help="what ONE call produces, e.g. 'per embedding'")
    ap.add_argument("--id", default=None)
    ap.add_argument("--passes-per-unit", type=int, default=1,
                    help="graph calls per work unit; Von's NLI mapping needs N")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args)
    if args.bundle is not None:
        return run_coreai(args)
    ap.print_usage(sys.stderr)
    print("latency_protocol: nothing to measure — pass --self-test for the dependency-free "
          "protocol, or --bundle <path> for the Core AI path.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
