#!/usr/bin/env bash
# 機能:
#   - オフライン予測スコアの評価（AUC / Logloss）を出力
#
# 使い方:
#   ./run_evaluate.sh [scores_csv] [out_metrics_csv]
#   例:
#     ./run_evaluate.sh runs/sim_user_item_scores.csv runs/sim_metrics_overall.csv
set -euo pipefail

SCORES_CSV="${1:-runs/sim_user_item_scores.csv}"
OUT_METRICS="${2:-runs/sim_metrics_overall.csv}"

python -m code.evaluate_scores \
  --scores-csv "${SCORES_CSV}" \
  --by overall \
  --out "${OUT_METRICS}"

