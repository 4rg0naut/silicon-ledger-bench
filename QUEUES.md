# Night-queue manifest (2026-10-27/28 lane matrix, session openfox-e9e19426)

Maps each queue script to the logs and row files it produced, so ledger
cells can be traced without reading every script. Logs live in `logs/`
plus `lanes/c/logs/` (early C-lane iterations predating the split — kept
in place, committed as provenance). Logs with an empty model segment
(`L0m_.log`, `S2_.log`, `lanes/c/logs/C3_.log`, `lanes/c/logs/c3_1.7b.log`) are debris from
invocations missing the model arg; kept, no ledger cites them. Rows live
in `lanes/rows/` (L0/L0m) and `results/raw/` (S1/S2/C3/R4/O5),
assembled by `tools/assemble_lane_matrix.py`.

| queue | binary path | logs | row outputs | notes |
|---|---|---|---|---|
| `night_queue.sh` (q1) | `.build/out/Products/Release/msl` (abs) | `*_q1.log` | L0/L0m rows + first S1/C3/R4 raw | first night; 30B Insufficient-Memory cascade → serialized-GPU policy (FAILURES.md) |
| q2 | — | — | — | no script kept on disk (same cascade window) |
| `night_queue3.sh` | `.build/out/Products/Release/msl` | `*_q3.log` | S1/C3/R4 raw, L0/L0m rows | |
| `night_queue4.sh` | `.build/out/Products/Release/msl` | `C3_30b_q4.log`, **`L0m_30b_q6.log`** | C3/R4/S1 raw, L0m 30b row | log label q6 written by queue4 (off-by-label; trust this table, not the qN in the name) |
| `night_queue5.sh` | `.build/out/Products/Release/msl` | `*_q5.log` | S1/C3/R4 raw + first S2 raw | |
| `night_queue6.sh` | `.build/out/Products/Release/msl` | `O5/R4 *_q6.log`, **`C3_30b_q7.log`** | O5 raw, C3/R4 raw + `/tmp/R4_30b_salvage.json` | no queue7 script exists; q7 label inside queue6 |
| `night_queue8.sh` | **`.build/release/msl`** (swift-direct, switched mid-night) | `L0m_*_q8.log`, `R4_*_q8.log` | R4 raw, L0m rows | |
| `night_queue9.sh` | `.build/release/msl` | `*_q9.log` | R4/S1/S2 raw, L0m rows | wrote 30B row as `S1_qwen3-30b-4bit.json` (wrong name), patched `mv -f` to `...-a3b...` |
| `night_queue10.sh` | `.build/release/msl` | `S1_*_q10.log`, `S2_*_q10.log` | S1/S2 raw | same 30B name bug + `mv -f` patch |
| `lanes/swift/queue11.sh` | `.build/release/msl` | `S1_30b_q11.log`, `{L0m,S1}_{1.7b,4b}_q11_[AB].log` | paired A/B to **`/tmp/{L0m,S1}_*_q11_[AB].json`** | headline same-session paired A/B; 2026-10-28: 8 rows copied to `results/raw/*_q11_*.json` (review #12) so they survive reboot; assembler keys them as separate `*_q11_a/_q11_b` cells |
| q12 (ad-hoc, no script) | `.build/release/msl` | `C3_1.7b_q12.log`, `O5_1.7b_q12.log` | none persisted | manual C3/O5 1.7B re-runs 2026-10-28 08:52–08:57 (metallib-path re-verification after queue11); logs kept, row outputs not captured — no ledger cites q12 |

Future nights: use one parameterized script (`QNN=$n ...` driving log/row
names from the queue id) instead of forking `night_queueN.sh` copies.
2026-10-29 (ncm side): queue-style launches of long arms go through
`ncm-run/bare/dev/arm.sh` — new arms get autosave + auto-recovery free
(see ncm-run/docs/commands.md §arm.sh); no queue script should re-implement
checkpoint or retry logic again.

`--out` handling in the Swift lane (`lanes/swift/Sources/msl/main.swift`
`emitRow`) writes exactly the path given — the 30B filename issue was the
calling strings in queues 9/10, corrected in queue11; no binary bug.

Regenerate ledger: `python3 tools/assemble_lane_matrix.py` (collects
`lanes/rows/*.json` + `results/raw/*.json`).
