#!/usr/bin/env python3
"""Stability gate for 3-run stability checks.

Usage: python3 tools/check_stability.py results/A.json results/B.json results/C.json [--suite s1|s2|all]

Verdicts:
  PASS  - every evaluated row is under its threshold
  FAIL  - prints each offending row

Thresholds (see docs/research-notes.md R8 + README stability protocol):
  s1_memory:            <10% spread on every subtest
  s2_gemm msl max-power SIMT rows (int8/fp16/fp8/fp8e5m2/fp4, n>=4096): <8%
  s2_gemm clock-steady rows (max median >=5ms):                          <5%
  s2_gemm rows with max median <5ms: informational only (DVFS/MPS
  per-process autotune dominate; expect up to ~25%)
"""
import json, sys
from collections import defaultdict

SUITES = ("s1_memory", "s2_gemm")
SIMT_POWER = ("int8", "fp16", "fp8", "fp8e5m2", "fp4")


def threshold(sec, name, variant, med_max):
    if sec == "s1_memory":
        return 10.0, "memory"
    parts = name.split("_")
    n = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else 0
    dt = parts[-1] if len(parts) > 2 else ""
    if variant == "msl" and dt in SIMT_POWER and n >= 4096:
        return 8.0, "simt-power"
    if med_max >= 5.0:
        return 5.0, "steady"
    return None, "short-informational"


def main():
    args = []
    suite = "all"
    paths_file = None
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--paths-file":
            paths_file = argv[i + 1]
            i += 2
            continue
        if a.startswith("--paths-file="):
            paths_file = a.split("=", 1)[1]
            i += 1
            continue
        if a.startswith("--suite"):
            suite = a.split("=", 1)[1] if "=" in a else argv[i + 1]
            i += 2 if "=" not in a else 1
            continue
        if not a.startswith("--"):
            args.append(a)
        i += 1
    if paths_file:
        args = [l.strip() for l in open(paths_file) if l.strip()]
    if len(args) < 3:
        print(__doc__)
        return 2
    wanted = {"s1": ("s1_memory",), "s2": ("s2_gemm",), "all": SUITES}[suite]
    series = defaultdict(dict)
    for f in args:
        d = json.load(open(f))
        ts = f.split("/")[-1].rsplit("-", 1)[0]
        for sec in d.get("sections", []):
            if sec["id"] not in wanted:
                continue
            for run in sec.get("runs", []):
                if run.get("status", "ok") != "ok":
                    continue
                med = run.get("median")
                if med is not None:
                    series[(sec["id"], run["name"], run.get("variant", ""))][ts] = med
    fails, infos = [], []
    for key in sorted(series):
        vals = list(series[key].values())
        if len(vals) < 2:
            continue
        mx, mn = max(vals), min(vals)
        spread = (mx - mn) / mx * 100 if mx else 0.0
        thr, regime = threshold(key[0], key[1], key[2], mx)
        row = f"{key[0]}/{key[1]}[{key[2]}] {regime}: spread={spread:.2f}%" + \
              (f" (thr {thr:.0f}%)" if thr else "") + f" vals={[round(v, 4) for v in vals]}"
        if thr is None:
            infos.append((spread, row))
        elif spread >= thr:
            fails.append(row)
        else:
            print("ok  " + row)
    for spread, row in sorted(infos, reverse=True):
        print("ii  " + row + (" (>25%!)" if spread > 25 else ""))
    for row in fails:
        print("!!  " + row)
    print(f"\n{len(series)} series, {len(fails)} failures, {len(infos)} informational")
    print("PASS" if not fails else "FAIL")
    return 0 if not fails else 1


sys.exit(main())
