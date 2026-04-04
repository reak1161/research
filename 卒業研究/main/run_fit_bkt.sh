#!/usr/bin/env bash
# 機能:
#   - BKT パラメータ（L2 / L2_tier）を推定して CSV 出力
#
# 使い方:
#   ./run_fit_bkt.sh [log_csv] [out_params_csv] [out_report_csv]
#   例:
#     ./run_fit_bkt.sh csv/sim_logs.csv csv/bkt_params_multi.csv runs/bkt_metrics.csv
set -euo pipefail

LOG_CSV="${1:-csv/sim_logs.csv}"
OUT_PARAMS="${2:-csv/bkt_params_multi.csv}"
OUT_REPORT="${3:-runs/bkt_metrics.csv}"

python -m code.offline_fit_bkt \
  --csv "${LOG_CSV}" \
  --skill-col L2 L2_tier \
  --order-col timestamp \
  --correct-col correct \
  --out-params "${OUT_PARAMS}" \
  --out-report "${OUT_REPORT}"

