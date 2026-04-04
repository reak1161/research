#!/usr/bin/env bash
# 機能:
#   - BKT パラメータ推定（L2 / L2_tier）
#   - IRT パラメータ推定（3PL）
#   - ユーザー×問題のオフライン正答確率（P_final）を計算
#   - ※このスクリプトはクイズ開始までは行わない（前処理専用）
#
# 使い方:
#   ./run_full_flow.sh
#   例:
#     ./run_full_flow.sh
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "Usage: ./run_full_flow.sh"
  echo "Example: ./run_full_flow.sh"
  exit 1
fi

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
  --w-bkt 0.5 \
  --w-tier 0.5 \
  --out runs/sim_user_item_scores.csv

echo "[4/4] Done (precompute only)"
echo "Generated files:"
echo "  - csv/bkt_params_multi.csv"
echo "  - csv/irt_items_estimated.csv"
echo "  - csv/irt_theta_estimated.csv"
echo "  - runs/sim_user_item_scores.csv"
echo
echo "To start quiz manually:"
echo "  ./run_quiz.sh <user_id>"
