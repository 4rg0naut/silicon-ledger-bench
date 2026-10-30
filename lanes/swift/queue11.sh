#!/bin/zsh
set -x
cd /Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench
LOGS=/Volumes/DEV/OpenFox/AI_dev/silicon-ledger-bench/logs
PY=/Volumes/DEV/OpenFox/AI_dev/.venv-mlx/bin/python

echo "=== [1] S1 30B rerun $(date)"
cd lanes/swift
./.build/release/msl --lane l1 --model mlx-community/Qwen3-30B-A3B-4bit --max-tokens 256 --runs 3 \
  --out ../../results/raw/S1_qwen3-30b-a3b-4bit.json > $LOGS/S1_30b_q11.log 2>&1; echo "exit $?"

echo "=== [2] fairness A/B same session $(date)"
for round in A B; do
  for m in "1.7b:Qwen3-1.7B-4bit" "4b:Qwen3-4B-4bit"; do
    k=${m%%:*}; id=${m#*:}
    $PY ../../lanes/python/l0m_microbench.py --params ../../lanes/c/params_$k.txt --tokens 256 --runs 3 \
      --out /tmp/L0m_${k}_q11_$round.json > $LOGS/L0m_${k}_q11_${round}.log 2>&1; echo "exit $?"
    ./.build/release/msl --lane l1 --model mlx-community/$id --max-tokens 256 --runs 3 \
      --out /tmp/S1_${k}_q11_$round.json > $LOGS/S1_${k}_q11_${round}.log 2>&1; echo "exit $?"
  done
done
echo ALL_DONE
