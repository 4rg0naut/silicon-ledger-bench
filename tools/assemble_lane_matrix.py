#!/usr/bin/env python3
"""Assemble silicon-ledger/1 record from lane rows (night lane matrix)."""
import json
import glob
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = os.path.join(ROOT, "lanes", "rows")
RAW = os.path.join(ROOT, "results", "raw")
MACHINE_SRC = os.path.join(ROOT, "results", "2026-10-22T12-37-03Z-bda452aabd60bf26.json")

LANE_ORDER = ["l0", "l0m", "l1", "l2", "l3", "l4", "l5"]
MODELS = ["qwen3-1.7b-4bit", "qwen3-4b-4bit", "qwen3-30b-a3b-4bit"]


LANE_CORE = {
    "l0": "python mlx 0.32.4 / mlx-lm 0.32.2 (.venv-mlx)",
    "l0m": "python mlx 0.32.4 hand-rolled loop (.venv-mlx)",
    "l1": "mlx-swift 19601207e9a0 (0.32.3) SPM release + mlx-swift-lm 3b339ad6",
    "l2": "FoundationModels (session) via mlx-swift-lm MLXFoundationModels",
    "l3": "links lanes/swift/.build/out/Products/Release/Cmlx.o (0.32.3)",
    "l4": "binds 0.32.3 mlx.h + links same Cmlx.o (same bytes as l3/l1 core)",
    "l5": "same C loop source + same Cmlx.o as l3",
}


def load_row(path):
    with open(path) as f:
        return json.load(f)


def collect():
    rows = {}
    for p in sorted(glob.glob(os.path.join(ROWS, "*.json"))) + \
             sorted(glob.glob(os.path.join(RAW, "*.json"))):
        try:
            r = load_row(p)
        except Exception as e:
            print(f"WARN unreadable {p}: {e}", file=sys.stderr)
            continue
        base = os.path.basename(p).lower()
        prefix_lane = {"o5": "l5", "s1": "l1", "s2": "l2",
                       "c3": "l3", "r4": "l4"}.get(base.split("_")[0])
        lane = prefix_lane or r.get("lane", base.split("_")[0].lower())
        mid = str(r.get("model", "")).lower()
        model = next((m for m in MODELS if m in base or m in mid), None)
        if model is None:
            model = "?"
            print(f"WARN unresolved model id {p}", file=sys.stderr)
        tagm = re.search(r"_q\d+_[a-z]", base)
        tag = tagm.group(0) if tagm else ""
        rows[(lane, model + tag)] = (r, p)
    return rows


def kind_of(r, model):
    k = r.get("model_kind") or r.get("model_type")
    if k:
        return k
    mid = (str(r.get("model", "")) + " " + model).lower()
    if "a3b" in mid or "moe" in mid:
        return "qwen3_moe"
    if "1.7b" in mid or "4b" in mid:
        return "qwen3_dense"
    return "unknown"


def decode_run(lane, model, r, src):
    dec = r.get("decode", r)
    steps = dec.get("decode_steps") or r.get("decode_steps") or 0
    if not steps and isinstance(dec.get("runs"), list):
        steps = sum(x.get("tokens", 0) for x in dec["runs"]
                    if isinstance(x, dict)) or None
    if not steps and isinstance(dec.get("per_step_ms"), list):
        steps = sum(len(rr) for rr in dec["per_step_ms"]
                    if isinstance(rr, list)) or None
    run = {
        "name": f"decode_{model}",
        "variant": lane,
        "config": {
            "model": r.get("model", model),
            "tokens_per_run": str(dec.get("tokens_per_run", "")) or "see logs",
            "runs": f"decode_steps={steps} (as-run; see logs/ + raw row)",
            "build": r.get("build_mode", "release"),
            "kind": kind_of(r, model),
            "core": LANE_CORE.get(lane, "?"),
            "wired_bytes": r.get("wired_bytes", "n/a (host default)"),
        },
        "unit": "ms",
        "iters": steps,
        "median": dec.get("p50_step_ms"),
        "mean": dec.get("mean_step_ms"),
        "p95": dec.get("p95_step_ms"),
        "min": min((x for rr in (dec.get("per_step_ms") or []) for x in
                    (rr if isinstance(rr, list) else [])), default=None),
        "max": max((x for rr in (dec.get("per_step_ms") or []) for x in
                    (rr if isinstance(rr, list) else [])), default=None),
        "derived": {"tps_wall": dec.get("mean_decode_tps_wall")},
        "status": "ok" if dec.get("mean_step_ms") else "incomplete",
    }
    if lane == "l2" and isinstance(dec.get("runs"), list):
        tps_vals = [r["wall_tps"] for r in dec["runs"] if r.get("wall_tps")]
        if tps_vals:
            ms_vals = sorted(1000.0 / t for t in tps_vals)
            run["median"] = round(ms_vals[len(ms_vals) // 2], 3)
            run["mean"] = round(sum(ms_vals) / len(ms_vals), 3)
            run["p95"] = round(ms_vals[-1], 3)
            run["min"] = round(ms_vals[0], 3)
            run["max"] = round(ms_vals[-1], 3)
            run["iters"] = sum(r.get("tokens_retokenized", 0) for r in dec["runs"])
            run["derived"]["tps_wall"] = round(
                sum(tps_vals) / len(tps_vals), 3)
            run["config"]["note"] = "FoundationModels session; short completions"
        return run
    if lane == "l0" and "per_step_ms" in dec and dec["per_step_ms"]:
        flat = [x for rr in dec["per_step_ms"] for x in rr]
        run["min"], run["max"] = min(flat), max(flat)
    if "peak_memory_bytes" in r:
        run["derived"]["peak_GB"] = round(r["peak_memory_bytes"] / 1e9, 3)
    if "load_seconds" in r:
        run["derived"]["load_s"] = r["load_seconds"]
    return run


def prefill_runs(lane, model, r):
    out = []
    for n in ("32", "512", "2048"):
        pf = (r.get("prefill") or {}).get(n)
        if not pf:
            continue
        tps = pf.get("wall_tps") or pf.get("ttfb_tps") or 0
        ptoks = pf.get("prompt_tokens", int(n))
        out.append({
            "name": f"prefill{ptoks}_{model}", "variant": lane,
            "config": {"prompt_tokens": str(ptoks)},
            "unit": "ms", "iters": ptoks,
            "median": round(ptoks / tps * 1000, 3) if tps else None,
            "mean": round(ptoks / tps * 1000, 3) if tps else None,
            "p95": None, "min": None, "max": None,
            "derived": {"prefill_tps": tps},
            "status": "ok" if tps else "incomplete",
        })
    return out


def main():
    machine = load_row(MACHINE_SRC)["machine"]
    rows = collect()
    runs, notes = [], []
    for (lane, model), (r, src) in sorted(rows.items()):
        runs.append(decode_run(lane, model, r, src))
        runs += prefill_runs(lane, model, r)
    def by():
        cell = {(r.get("variant"), r["name"][len("decode_"):]):
                (r.get("derived") or {}).get("tps_wall")
                for r in runs if r.get("name", "").startswith("decode_")}
        lines = []
        for (l, m) in sorted(rows):
            tps = rows[(l, m)][0].get('decode', rows[(l, m)][0]).get('mean_decode_tps_wall')
            if tps is None:
                tps = cell.get((l, m), '?')
            if isinstance(tps, (int, float)):
                tps = f"{tps:.1f}"
            lines.append(f"{l:4s} {m:20s} {tps} t/s")
        return "\n".join(lines)
    notes.append("matrix collected via serialized GPU queues; per-queue "
                 "script/log/row mapping in QUEUES.md (one GPU job at a "
                 "time after the 30B Insufficient-Memory cascade).")
    notes.append("Session: openfox-e9e19426 night experiment 2026-10-27/28; "
                 "base rev 3cadaaf, lane-matrix additions in git history.")
    notes.append("Versions pinned: python lane .venv-mlx (uv) mlx 0.32.4 / "
                 "mlx-lm 0.32.2; Swift mlx-swift 19601207e9a0de51e03ee6ec0c3c5f3784275075 "
                 "(0.32.3) + mlx-swift-lm 3b339ad6e3b3f44c8121ecff5131c7fd55e075e6; "
                 "C/Rust link the same 0.32.3 Cmlx.o -> L3/L4/L5 == L1 engine "
                 "binary-wise (Rust pre-fix 62/64-token rows in FAILURES "
                 "LF-11, kept in results/raw history only).")
    notes.append("All C/OBJC numbers on the SPM default.metallib; wheel "
                 "metallib was ~4x slower on lm_head (A/B: FAILURES.md LF-02, "
                 "evidence lanes/c/gpu_probe2.c).")
    notes.append("Native lane numbers are step-count-sensitive: live buffers "
                 "grow ~2MB/step; decode_steps<768 cells are short-salvage "
                 "runs (long-run crashes: FAILURES.md); prefill for C/Rust/"
                 "L0m is measured after the decode session (pessimistic); "
                 "L1 prefill is best-of-4 ttfb, prompt tokens exact.")
    notes.append("L2 FoundationModels t/s not comparable to raw-loop lanes "
                 "(FM session policy adds prompt overhead).")
    notes.append("Review fixes 7-12 (2026-10-28/29): kind derivation, "
                 "prefill unit=ms, null-when-absent stats, derived iters, "
                 "QUEUES.md manifest, q11 paired rows copied to results/raw.")
    notes.append("Host drift ~2x across the night; rankings use SAME-SESSION "
                 "paired A/B (results/raw/*_q11_*.json): 1.7B S1=348.3 vs "
                 "L0m=305.7 (+14%), 4B S1=179.5 vs L0m=159.7 (+11%), round-B "
                 "reproduced to 0.4%. Single-row t/s across time windows are "
                 "not directly comparable.")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    doc = {
        "schema": "silicon-ledger/1",
        "generated": now,
        "machine": machine,
        "sections": [{
            "id": "s6_llm_lane_matrix",
            "title": "LLM host-language lane matrix (Swift/C/Rust/Python microbench vs mlx_lm) on M5 Max, Qwen3 dense+MoE",
            "runs": runs,
            "notes": notes,
        }],
        "status": {"s6_llm_lane_matrix": "done"},
    }
    out = os.path.join(ROOT, "results", f"{now}-{machine['fingerprint']}.json")
    if "--dry-run" in sys.argv[1:]:
        print("dry-run: no file written")
    else:
        with open(out, "w") as f:
            json.dump(doc, f, indent=1)
        print("wrote", out)
    print(by())


if __name__ == "__main__":
    unknown = [a for a in sys.argv[1:] if a not in ("--dry-run",)]
    if unknown:
        print(f"error: unrecognized arguments: {' '.join(unknown)} (did you mean --dry-run?)",
              file=sys.stderr)
        sys.exit(2)
    main()
