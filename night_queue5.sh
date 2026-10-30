#!/bin/zsh
set -x
cd /Volumes/DEV/OpenFox/AI_dev
LOGS=/Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/logs

echo "=== [1] C 1.7B FINAL $(date)"
cd silicon-ledger-bench/lanes/c
./qwen3_decode --params params_1.7b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/C3_qwen3-1.7b-4bit.json > $LOGS/C3_1.7b_q5.log 2>&1; echo "exit $?"

echo "=== [2] C 4B FINAL $(date)"
./qwen3_decode --params params_4b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/C3_qwen3-4b-4bit.json > $LOGS/C3_4b_q5.log 2>&1; echo "exit $?"

echo "=== [3] Rust 4B CLEAR=50 $(date)"
cd ../rust
CLEAR=50 ./target/release/qwen3_decode --params ../c/params_4b.txt --tokens 256 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/R4_qwen3-4b-4bit.json > $LOGS/R4_4b_q5.log 2>&1; echo "exit $?"

echo "=== [4] Swift L2 1.7B FoundationModels $(date)"
cd ../swift
./.build/out/Products/Release/msl --lane l2 --model mlx-community/Qwen3-1.7B-4bit \
  --max-tokens 256 --runs 3 --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/S2_qwen3-1.7b-4bit.json \
  > $LOGS/S2_1.7b_q5.log 2>&1; echo "exit $?"

echo "=== [5] Swift L1 30B $(date)"
./.build/out/Products/Release/msl --lane l1 --model mlx-community/Qwen3-30B-A3B-4bit \
  --max-tokens 256 --runs 3 --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/S1_qwen3-30b-a3b-4bit.json \
  > $LOGS/S1_30b_q5.log 2>&1; echo "exit $?"

echo "=== QUEUE5 DONE $(date)"
