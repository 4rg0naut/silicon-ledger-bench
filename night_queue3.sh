#!/bin/zsh
set -x
cd /Volumes/DEV/OpenFox/AI_dev
V=.venv-mlx/bin/python
LOGS=/Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/logs

echo "=== [1] C 1.7B 768-steps $(date)"
cd silicon-ledger-bench/lanes/c
./qwen3_decode --params params_1.7b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/C3_qwen3-1.7b-4bit.json > $LOGS/C3_1.7b_q3.log 2>&1; echo "exit $?"

echo "=== [2] C 4B $(date)"
./qwen3_decode --params params_4b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/C3_qwen3-4b-4bit.json > $LOGS/C3_4b_q3.log 2>&1; echo "exit $?"

echo "=== [3] Rust 4B $(date)"
cd ../rust
./target/release/qwen3_decode --params ../c/params_4b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/R4_qwen3-4b-4bit.json > $LOGS/R4_4b_q3.log 2>&1; echo "exit $?"

echo "=== [4] Swift L1 1.7B $(date)"
cd ../swift
./.build/out/Products/Release/msl --lane l1 --model mlx-community/Qwen3-1.7B-4bit \
  --max-tokens 256 --runs 3 --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/S1_qwen3-1.7b-4bit.json \
  > $LOGS/S1_1.7b_q3.log 2>&1; echo "exit $?"

echo "=== [5] L0m 30B params-fixed $(date)"
cd /Volumes/DEV/OpenFox/AI_dev
$V silicon-ledger-bench/lanes/python/l0m_microbench.py \
  --params silicon-ledger-bench/lanes/c/params_30b.txt --tokens 256 --runs 3 \
  --out silicon-ledger-bench/lanes/rows/L0m_qwen3-30b-a3b-4bit.json \
  > $LOGS/L0m_30b_q3.log 2>&1; echo "exit $?"

echo "=== [6] L0 30B wired $(date)"
$V silicon-ledger-bench/lanes/python/l0_full_loop.py \
  --model mlx-community/Qwen3-30B-A3B-4bit --max-tokens 256 --runs 3 --wired-bytes 96000000000 \
  --out silicon-ledger-bench/lanes/rows/L0_qwen3-30b-a3b-4bit.json \
  > $LOGS/L0_30B_q3.log 2>&1; echo "exit $?"

echo "=== QUEUE3 DONE $(date)"
