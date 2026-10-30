#!/bin/zsh
# Night queue: GPU jobs strictly serialized (contention caused 30B OOM).
set -x
cd /Volumes/DEV/OpenFox/AI_dev
V=.venv-mlx/bin/python
LOGS=/Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/logs
mkdir -p $LOGS silicon-ledger-bench/lanes/rows silicon-ledger-bench/results/raw

echo "=== [1] L0 30B $(date)"
$V silicon-ledger-bench/lanes/python/l0_full_loop.py \
  --model mlx-community/Qwen3-30B-A3B-4bit --max-tokens 256 --runs 3 \
  --out silicon-ledger-bench/lanes/rows/L0_qwen3-30b-a3b-4bit.json \
  > $LOGS/L0_30B_q1.log 2>&1; echo "exit $?"

echo "=== [2] L0m 1.7B $(date)"
$V silicon-ledger-bench/lanes/python/l0m_microbench.py \
  --params silicon-ledger-bench/lanes/c/params_1.7b.txt --tokens 256 --runs 3 \
  --out silicon-ledger-bench/lanes/rows/L0m_qwen3-1.7b-4bit.json \
  > $LOGS/L0m_1.7b_q1.log 2>&1; echo "exit $?"

echo "=== [3] L0m 4B $(date)"
$V silicon-ledger-bench/lanes/python/l0m_microbench.py \
  --params silicon-ledger-bench/lanes/c/params_4b.txt --tokens 256 --runs 3 \
  --out silicon-ledger-bench/lanes/rows/L0m_qwen3-4b-4bit.json \
  > $LOGS/L0m_4b_q1.log 2>&1; echo "exit $?"

echo "=== [4] L0m 30B $(date)"
$V silicon-ledger-bench/lanes/python/l0m_microbench.py \
  --params silicon-ledger-bench/lanes/c/params_30b.txt --tokens 256 --runs 3 \
  --out silicon-ledger-bench/lanes/rows/L0m_qwen3-30b-a3b-4bit.json \
  > $LOGS/L0m_30b_q1.log 2>&1; echo "exit $?"

echo "=== [5] L3 C 1.7B final $(date)"
cd silicon-ledger-bench/lanes/c
./qwen3_decode --params params_1.7b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/C3_qwen3-1.7b-4bit.json > $LOGS/C3_1.7b_q1.log 2>&1; echo "exit $?"

echo "=== [6] L3 C 4B $(date)"
./qwen3_decode --params params_4b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/C3_qwen3-4b-4bit.json > $LOGS/C3_4b_q1.log 2>&1; echo "exit $?"

echo "=== [7] L3 C 30B MoE $(date)"
./qwen3_decode --params params_30b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/C3_qwen3-30b-a3b-4bit.json > $LOGS/C3_30b_q1.log 2>&1; echo "exit $?"

echo "=== [8] L4 Rust 1.7B $(date)"
cd ../rust
./target/release/qwen3_decode --params ../c/params_1.7b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/R4_qwen3-1.7b-4bit.json > $LOGS/R4_1.7b_q1.log 2>&1; echo "exit $?"

echo "=== [9] L1 Swift 1.7B $(date)"
cd ../swift
./.build/out/Products/Release/msl --lane l1 --model mlx-community/Qwen3-1.7B-4bit \
  --max-tokens 256 --runs 3 --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/S1_qwen3-1.7b-4bit.json \
  > $LOGS/S1_1.7b_q1.log 2>&1; echo "exit $?"

echo "=== NIGHT QUEUE DONE $(date)"
