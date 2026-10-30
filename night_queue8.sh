#!/bin/zsh
set -x
cd /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench
LOGS=/Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/logs
PY=/Volumes/DEV/OpenFox/AI_dev/.venv-mlx/bin/python

echo "=== [1] Rust full dense $(date)"
cd lanes/rust
./target/release/qwen3_decode --params ../c/params_1.7b.txt --tokens 256 --runs 3 \
  --out ../../results/raw/R4_qwen3-1.7b-4bit.json > $LOGS/R4_1.7b_q8.log 2>&1; echo "exit $?"
./target/release/qwen3_decode --params ../c/params_4b.txt --tokens 256 --runs 3 \
  --out ../../results/raw/R4_qwen3-4b-4bit.json > $LOGS/R4_4b_q8.log 2>&1; echo "exit $?"

echo "=== [2] Rust 30B 128tok-salvage $(date)"
./target/release/qwen3_decode --params ../c/params_30b.txt --tokens 128 --runs 1 \
  --out ../../results/raw/R4_qwen3-30b-a3b-4bit.json > $LOGS/R4_30b_q8.log 2>&1; echo "exit $?"

echo "=== [3] L0m python full $(date)"
cd ../..
for m in "1.7b:qwen3-1.7b-4bit" "4b:qwen3-4b-4bit" "30b:qwen3-30b-a3b-4bit"; do
  k=${m%%:*}; f=${m#*:}
  $PY lanes/python/l0m_microbench.py --params lanes/c/params_$k.txt --tokens 256 --runs 3 \
    --out lanes/rows/L0m_$f.json > $LOGS/L0m_$k_q8.log 2>&1; echo "exit $?"
done

echo "=== [4] Swift S1 exact-prompt $(date)"
cd lanes/swift
for m in "1.7b:Qwen3-1.7B-4bit" "4b:Qwen3-4B-4bit" "30b:Qwen3-30B-A3B-4bit"; do
  k=${m%%:*}; id=${m#*:}
  ./.build/release/msl --lane l1 --model mlx-community/$id --max-tokens 256 --runs 3 \
    --out ../../results/raw/S1_qwen3-$k-4bit.json > $LOGS/S1_$k_q8.log 2>&1; echo "exit $?"
done

echo "=== [5] Swift L2 4B/30B $(date)"
for m in "4b:Qwen3-4B-4bit" "30b:Qwen3-30B-A3B-4bit"; do
  k=${m%%:*}; id=${m#*:}
  ./.build/release/msl --lane l2 --model mlx-community/$id --max-tokens 64 --runs 2 \
    --out ../../results/raw/S2_qwen3-$k-4bit.json > $LOGS/S2_$k_q8.log 2>&1; echo "exit $?"
done
echo ALL_DONE
