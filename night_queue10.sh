#!/bin/zsh
set -x
cd /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench
LOGS=/Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/logs

echo "=== [1] Swift S1 measured-truth prompt $(date)"
cd lanes/swift
for m in "1.7b:Qwen3-1.7B-4bit" "4b:Qwen3-4B-4bit" "30b-a3b:Qwen3-30B-A3B-4bit"; do
  k=${m%%:*}; id=${m#*:}
  ./.build/release/msl --lane l1 --model mlx-community/$id --max-tokens 256 --runs 3 \
    --out ../../results/raw/S1_qwen3-${k}-4bit.json > $LOGS/S1_${k}_q10.log 2>&1; echo "exit $?"
done

echo "=== [2] Swift L2 retry-on-empty $(date)"
for m in "1.7b:Qwen3-1.7B-4bit" "4b:Qwen3-4B-4bit" "30b-a3b:Qwen3-30B-A3B-4bit"; do
  k=${m%%:*}; id=${m#*:}
  ./.build/release/msl --lane l2 --model mlx-community/$id --max-tokens 256 --runs 3 \
    --out ../../results/raw/S2_qwen3-${k}-4bit.json > $LOGS/S2_${k}_q10.log 2>&1; echo "exit $?"
done
echo ALL_DONE
