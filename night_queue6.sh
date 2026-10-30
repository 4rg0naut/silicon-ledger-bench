#!/bin/zsh
set -x
cd /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench
LOGS=/Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/logs
cd lanes/c

echo "=== [1] L5 ObjC 1.7B $(date)"
TOKLOG=1 ./qwen3_decode_objc --params params_1.7b.txt --tokens 64 --runs 3 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/O5_qwen3-1.7b-4bit.json > $LOGS/O5_1.7b_q6.log 2>&1; echo "exit $?"

echo "=== [2] C 30B 96tok-salvage $(date)"
./qwen3_decode --params params_30b.txt --tokens 96 --runs 2 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/C3_qwen3-30b-a3b-4bit.json > $LOGS/C3_30b_q7.log 2>&1; echo "exit $?"

echo "=== [3] Rust 4B 96tok-salvage $(date)"
cd ../rust
./target/release/qwen3_decode --params ../c/params_4b.txt --tokens 96 --runs 2 \
  --out /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/results/raw/R4_qwen3-4b-4bit.json > $LOGS/R4_4b_q6.log 2>&1; echo "exit $?"

echo "=== [4] Rust 30B 64tok-salvage $(date)"
./target/release/qwen3_decode --params ../c/params_30b.txt --tokens 64 --runs 1 \
  --out /tmp/R4_30b_salvage.json > $LOGS/R4_30b_q6.log 2>&1; echo "exit $?"

echo "=== QUEUE6 DONE $(date)"
