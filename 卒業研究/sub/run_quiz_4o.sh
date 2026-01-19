#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: ./run_quiz_4o.sh <user_id>"
  exit 1
fi

USER_ID="$1"

python -m code.interactive_lpic_quiz \
  --user-id "$USER_ID" \
  --items-csv csv/items_sample_lpic_tier.csv \
  --bkt-params-csv csv/bkt_params_multi.csv \
  --log-csv csv/sim_online_logs.csv \
  --init-log-csv csv/sim_logs.csv \
  --init-include-log \
  --use-llm \
  --llm-model gpt-4o
