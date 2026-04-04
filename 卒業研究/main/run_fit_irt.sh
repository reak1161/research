#!/usr/bin/env bash
# 機能:
#   - IRT（3PL）パラメータを推定して item/theta CSV を出力
#   - エポックごとの推定履歴を CSV/PNG/PDF で出力
#
# 使い方:
#   ./run_fit_irt.sh [log_csv] [out_items_csv] [out_theta_csv] [out_history_csv] [plot_history_dir]
#   例:
#     ./run_fit_irt.sh csv/sim_logs.csv csv/irt_items_estimated.csv csv/irt_theta_estimated.csv runs/irt_fit_history.csv runs/irt_fit_plots
set -euo pipefail

LOG_CSV="${1:-csv/sim_logs.csv}"
OUT_ITEMS="${2:-csv/irt_items_estimated.csv}"
OUT_THETA="${3:-csv/irt_theta_estimated.csv}"
OUT_HISTORY="${4:-runs/irt_fit_history.csv}"
PLOT_HISTORY_DIR="${5:-runs/irt_fit_plots}"

python -m code.fit_irt_params \
  --log-csv "${LOG_CSV}" \
  --user-col user_id \
  --item-col item_id \
  --domain-col L2 \
  --correct-col correct \
  --out-items "${OUT_ITEMS}" \
  --out-theta "${OUT_THETA}" \
  --out-history "${OUT_HISTORY}" \
  --plot-history-dir "${PLOT_HISTORY_DIR}"
