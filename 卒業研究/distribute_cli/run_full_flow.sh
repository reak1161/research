#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: ./run_full_flow.sh <user_id> [llm_model]"
  echo "Example: ./run_full_flow.sh user000 gpt-4o-mini"
  exit 1
fi

USER_ID="$1"
LLM_MODEL="${2:-gpt-4o-mini}"

echo "[1/4] Fit BKT params (L2 + L2_tier)"
python -m code.offline_fit_bkt \
  --csv csv/sim_logs.csv \
  --skill-col L2 L2_tier \
  --order-col timestamp \
  --correct-col correct \
  --out-params csv/bkt_params_multi.csv \
  --out-report runs/bkt_metrics.csv

echo "[2/4] Fit IRT params (3PL)"
python -m code.fit_irt_params \
  --log-csv csv/sim_logs.csv \
  --user-col user_id \
  --item-col item_id \
  --domain-col L2 \
  --correct-col correct \
  --out-items csv/irt_items_estimated.csv \
  --out-theta csv/irt_theta_estimated.csv

echo "[3/4] Build offline user-item scores"
python -m code.offline_predict_scores \
  --log-csv csv/sim_logs.csv \
  --params-csv csv/bkt_params_multi.csv \
  --params-main-field L2 \
  --params-tier-field L2_tier \
  --irt-items-csv csv/irt_items_estimated.csv \
  --items-csv csv/items_sample_lpic_tier.csv \
  --items-domain-col L2 \
  --skill-col L2 \
  --tier-skill-col L2_tier \
  --mode hybrid_mean \
  --w-bkt 0.6 \
  --w-tier 0.5 \
  --out runs/sim_user_item_scores.csv

echo "[4/4] Start interactive quiz"
python -m code.interactive_lpic_quiz \
  --user-id "$USER_ID" \
  --items-csv csv/items_sample_lpic_tier.csv \
  --bkt-params-csv csv/bkt_params_multi.csv \
  --log-csv csv/sim_online_logs.csv \
  --init-log-csv csv/sim_logs.csv \
  --init-include-log \
  --irt-items-csv csv/irt_items_estimated.csv \
  --pfinal-mode realtime \
  --use-llm \
  --llm-model "$LLM_MODEL"

