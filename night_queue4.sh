#!/bin/zsh
set -x
cd /Volumes/DEV/OpenFox/AI_dev
V=.venv-mlx/bin/python
LOGS=/Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/logs

echo "=== [1] L0m 30B rhs-fix $(date)"
$V silicon-ledger-bench/lanes/python/l0m_microbench.py \
  --params silicon-ledger-bench/lanes/c/params_30b.txt --tokens 256 --runs 3 \
  --out silicon-ledger-bench/lanes/rows/L0m_qwen3-30b-a3b-4bit.json \
  > $LOGS/L0m_30b_q6.log 2>&1; echo "exit $?"

echo "=== [2] C 30B MoE slice-fix $(date)"
cd silicon-ledger-bench/lanes/c
./qwen3_decode --params params_30b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/C3_qwen3-30b-a3b-4bit.json > $LOGS/C3_30b_q4.log 2>&1; echo "exit $?"

echo "=== [3] Rust 4B retry $(date)"
cd ../rust
./target/release/qwen3_decode --params ../c/params_4b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/R4_qwen3-4b-4bit.json > $LOGS/R4_4b_q4.log 2>&1; echo "exit $?"

echo "=== [4] Rust 30B $(date)"
./target/release/qwen3_decode --params ../c/params_30b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/R4_qwen3-30b-a3b-4bit.json > $LOGS/R4_30b_q4.log 2>&1; echo "exit $?"

echo "=== [5] Swift L1 4B $(date)"
cd ../swift
./.build/out/Products/Release/msl --lane l1 --model mlx-community/Qwen3-4B-4bit \
  --max-tokens 256 --runs 3 --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/S1_qwen3-4b-4bit.json \
  > $LOGS/S1_4b_q4.log 2>&1; echo "exit $?"

echo "=== QUEUE4 DONE $(date)"
