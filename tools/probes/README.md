# probes — the instruments behind the numbers

Each tool builds with `clang` (or runs under stock `python3`) on macOS and needs no
installation, no package manager, and (except where noted) no root. One exception is
called out in its row: `latency_protocol.py` drives graphs through the Core AI runtime and
needs `numpy`, so it is only usable where those are installed — it says so and stops rather
than pretending.

Everything here was written to answer one question honestly: **which engine did the work,
and what did it cost** — on a machine whose OS gives no supported way to ask.

| tool | what it answers | build / run |
| --- | --- | --- |
| `enginemon.c` | which engine ran: CPU / GPU-shader / GPU-NA / ANE, unprivileged | `clang -O2 -o enginemon enginemon.c -framework CoreFoundation` |
| `ane-probe.m` | what the private framework on *this* machine actually exposes (classes, selectors, the constant table) | `clang -O2 -o ane-probe ane-probe.m -framework Foundation` |
| `nax_load_ab.m` | matched A/B between the GPU tensor units and a shader path, per-GEMM time and energy | `clang -fobjc-arc -O2 -o nax_load_ab nax_load_ab.m -framework Metal -framework Foundation` |
| `nax_steel_gemm.metal` | the kernel the tensor arm actually runs (required beside `nax_load_ab`) | — (source input; see its licence header) |
| `nax_ladder.m` | does a given tensor-tile shape compute correctly at all | `clang -fobjc-arc -O2 -o nax_ladder nax_ladder.m -framework Metal -framework Foundation` |
| `canary.c` | a fixed-work CPU + memory canary, so a later session can be compared to this one | `clang -O0 -o canary canary.c` (`-O0` is the contract; `-O2` deletes the loops) |
| `quiet_probe.sh` | the **window gate**: machine identity, suspect daemons, GPU idle floor, canary, and a loud verdict — run before trusting any measurement | `sh quiet_probe.sh [label]` (needs the sibling `enginemon` / `canary` binaries built above) |
| `latency_protocol.py` | one latency convention (10 warm-up, 100 measured, p50/p95/min + ms/token) emitted as schema rows | `python3 latency_protocol.py --bundle <path> --seq-len N --model M --placement ANE --work-unit W --id ID` (requires `numpy` + `coreai.runtime`) |

## The two rules these tools enforce on themselves

Both come from failures that produced *confident wrong answers*, which is worse than no
answer:

1. **A silent no-op must not print a number.** `nax_load_ab` and `nax_ladder` compute a CPU
   oracle and refuse to emit timing when the result is wrong or all-zero. A tensor arm that
   returns zeros is a *result* — it means the shape or the path is unsupported — and printing
   "4.7 ms" beside it hides that.
2. **Prove a counter moves before trusting its zero.** `enginemon` labels frozen channels as
   frozen (`Energy Model → ANE` never changes, on any workload) and excludes the free-running
   24 MHz clocks (`SoC Stats → ANE_*_TRIG`) that look like activity and mean nothing. A zero
   on a dead channel is not evidence of an idle engine.

## Reading them together

The honest measurement unit here is a **matched A/B with a baseline**, not an absolute
reading: run the same work down two paths, checksum both, subtract a measured idle window,
and report the difference. See `../../REPRODUCE.md` for the expected bands and the machine
fingerprint format.

## Licences

Our own code here is MIT (see `../../LICENSE`). One exception, and it carries its notice
in-file: **`nax_steel_gemm.metal` is MLX's Steel GEMM source (MIT, © Apple Inc.)**, extracted
from MLX's `mlx/backend/metal/kernels/steel/gemm/gemm_nax.h`; it is included because the
tensor-unit result cannot be reproduced without it. Do not hand-edit it — regenerate from MLX.
