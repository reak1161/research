#!/usr/bin/env bash
# 機能:
#   - BKT + IRT のオフライン正答確率（P_final）を user×item で計算
#
# 使い方:
#   ./run_offline_predict.sh [log_csv] [out_scores_csv]
#   例:
#     ./run_offline_predict.sh csv/sim_logs.csv runs/sim_user_item_scores.csv
set -euo pipefail

LOG_CSV="${1:-csv/sim_logs.csv}"
OUT_SCORES="${2:-runs/sim_user_item_scores.csv}"

python -m code.offline_predict_scores \
  --log-csv "${LOG_CSV}" \
  --params-csv csv/bkt_params_multi.csv \
  --params-main-field L2 \
  --params-tier-field L2_tier \
  --irt-items-csv csv/irt_items_estimated.csv \
  --items-csv csv/items_sample_lpic_tier.csv \
  --items-domain-col L2 \
  --skill-col L2 \
  --tier-skill-col L2_tier \
  --mode hybrid_mean \
  --w-bkt 0.5 \
  --w-tier 0.5 \
  --out "${OUT_SCORES}"
