# enginemon — which engine actually ran your work (CPU / GPU-shader / GPU-NA / ANE), no root

A single-file C tool that reads Apple's **IOReport** counters directly and reports
per-engine activity. On **M4**: ANE memory traffic (`AMC Stats` bytes) + interrupt counts.
On **M5** (v2.1): ANE **floor-bin duty** (`PMP0 DCS Floor → ANE-DCS-BW` F≥2 — calibrated
positive signal), GPU engagement (UT histogram, context-only by calibration — "UT" is
PMGR utilization tracking, see below), and the GPU rail that carries both shader and
Neural-Accelerator work, so GPU-NA attribution is a matched-A/B energy delta.

```bash
clang -O2 -o enginemon enginemon.c -framework CoreFoundation

./enginemon --list                      # every engine channel it can see
./enginemon --list --details            # + channel-detail blobs (API absent on macOS 27: prints a note)
./enginemon --interval 500 --duration 5 # sample for 5 s
./enginemon --interval 500 --duration 25 -- <command> [args...]   # sample while a workload runs
./enginemon --interval 1000 --duration 30 --json --out cap.jsonl -- <command> [args...]
```

`--out` implies `--json` (before 2026-10-08 a bare `--out` opened a file no code
path wrote to — 0-byte capture, output silently on stdout).

## Why this exists

`powermetrics` needs **sudo**. `xctrace`'s `ane-hw-intervals` is **blind to Core AI graphs**
(measured: 0 intervals, while a Core ML control in the same session logged 1310). And
`preferredComputeUnitKind: .neuralEngine` turned out to run on the **GPU** anyway. Placement
needs an instrument that is unprivileged, trustworthy, and not tied to one runtime.

IOReport is private but stable, and **an unprivileged process can list, subscribe to and
sample it** (verified: euid 501, macOS 27.0, M4). No helper, no entitlement.

## The two traps it is built to avoid

Both were hit while developing it, and both would have produced a *confident wrong answer*:

1. **`Energy Model → ANE` is frozen.** It reads a constant (10945781) and never changes — not
   under idle, not under a Core ML workload that provably uses the ANE. Reporting its zero as
   "the ANE was not used" is a **fabricated negative**. The tool labels it as frozen and tells
   you to use other channels.
2. **`SoC Stats → ANE_*_TRIG` is a free-running 24 MHz clock.** It advances by exactly
   `elapsed × 24e6` (5.46 s → 131110395 ticks) *regardless of load*. It looks like activity and
   means nothing. The tool excludes these channels.

## What actually works (validated)

Controls run back to back, same machine, same session:

| channel | idle | Core ML `compute_units=ALL` (known ANE) |
| --- | ---: | ---: |
| `AMC Stats → ANE DCS RD` | 0 B | **128–171 GB** |
| `AMC Stats → ANE NRT AF RD` | 0 B | **184 GB** |
| `Interrupt Statistics`, subgroup `ane 0`, 2nd-level count | 0 | **24,538 – 37,368** |
| `PMP → ANE0 RD` | 0 events | 29,252 |
| `Energy Model → GPU Energy` | ~15 mW | ~14 mW (GPU idle — the ANE did the work) |
| `Energy Model → ANE` | 0 | **0** ← dead channel |

**Read `AMC Stats` (bytes moved) and the `ane 0` interrupt counts.** Those are the signals
that separate "the ANE worked" from "the ANE was idle".

## M5 Max (macOS 27, v2.1) — calibration of 2026-10-05

The M4 signals above do **not** survive at user level on M5+ (AMC byte counters are
kernel-blocked per mactop `32d86fd`; `dart-ane0` handler counts stay 0 under real
ANE-direct load). The M5-positive table, from the r3 matrix (idle/cpu/tensorops/msl/mps/
ane/coreai arms, in-repo at `results/EXP-024-engine-attribution/raw/calibration-r3/` —
gzipped streams + MANIFEST.sha256; generation scratch: dev-box `exp024-raw/calibration/`):

| signal | idle | ANE-direct arm (25 MIL, ~9 s) | verdict |
| --- | ---: | ---: | --- |
| `PMP0 DCS Floor → ANE-DCS-BW` F≥2 residency (full window) | 0% | **82.9%** (coreai arm 40.7%, matching its duty) | **CALIBRATED ANE positive** |
| `PMP0 SOC Floor → ANE-LNK*‑AF-BW` VMAX+VOVD residency (full window) | 0% (100% VMIN) | **82.9%** (coreai 51.6%) | **CALIBRATED, independent corroboration** |
| `GPU UT Engagement` / AggD per-perf-state | 112,680/s | 33,629/s — *below* idle | **trick**: peaked on the shader-GEMM arm (295,979/s = 2.6× idle), fails 10× separation |
| `Energy Model` ANE0/GPU0/CPU Energy | frozen 0 | frozen 0 | gated on macOS 27 (rails exist; they advance only alongside an entitled sampler — mactop `53e5f90`) |

- **"UT" is PMGR utilization tracking, not the Neural Accelerator.** The `UT_EXT_THROTTLE_*`
  family spans the SoC (`MGPU03/MGPU12/AFR/PWRS/ACCP/ACCM0/ACCM1/SOC_*`, seen in ioreg under
  `AppleT6050PMGR`). `GPU UT Engagement` = generic per-perf-state GPU engagement (centi-%).
  It is kept with tag `gpu-na` as a TAG ONLY; GPU-NA attribution is the matched-A/B
  `GPU Energy` delta (see `knowledge/ane/10-m5-attribution-signals.md`).
- `sum` records now carry `"ane_floor":{"dcs_f2_ticks":…,"lnk_hi_ticks":…}` — the duty
  numbers above, computed in-tool.
- `--details` finding: macOS 27 exports **no** channel-details symbol (dlsym → NULL in both
  libIOReport and IOKit); histogram bucket bounds ride the state names already captured
  (`F1..F8`, `VMIN/VNOM/VMAX/VOVD*`).
- PMP0 channels are tagged per name (`AGX→gpu`, `PACC/MACC→cpu`, `ANE→ane`, rest `soc`);
  the earlier whole-group `ane` tag mislabeled GPU clusters' floors as ANE.

## Reading the output

```
SUMMARY (over 9.11s)
  CPU Energy            0.0 mW
  ANE                   0.0 mW   (this channel is frozen on M4 -- use AMC/interrupt counters, not this)
  GPU Energy           14.0 mW
  ANE activity   128235228928 B moved, 37368 interrupts
```

- **`ANE activity` > 0** → the ANE moved that much memory and took that many interrupts.
- **`ANE activity` = 0** with high `GPU Energy` → the GPU did the work.
- On **M5** read the `ANE floor` line instead: `F>=2 duty` is the calibrated ANE signal
  (0% when idle by construction; >0 ⇒ ANE active). M5's `ANE handlers`/`DCS traffic` are
  kernel-blocked at user level — zeros there are **gated, not idle**.
- **GPU-NA (M5 tensor units)** has no per-unit counter at user level: `ut` output and the
  `gpu-na` tag are GPU *engagement* context only (calibration demoted them, see M5
  section). To attribute NA work: run the same matmul compiled tensor-ops vs
  `simdgroup_matrix`, checksum-verify both, and diff `GPU Energy` — the delta is the
  NA pJ/token. Details: `knowledge/ane/10-m5-attribution-signals.md`.

## Caveats

- The ANE interrupt rows belong to **subgroup `ane 0`**. The `dart-ane 0` rows stay at 0 for
  ANE compute; the tool tracks both, so read the subgroup column.
- Energy channels are cumulative; deltas come from `IOReportCreateSamplesDelta`, not from
  hand-differencing samples.
- Private API: symbols are resolved with `dlsym` at runtime, so the tool fails softly
  (`IOReport symbols missing`) rather than breaking at link time if Apple moves things.
- Channel availability is SoC/OS specific. Run `--list` first on a new machine; a channel that
  does not exist here may exist there, and vice versa.
