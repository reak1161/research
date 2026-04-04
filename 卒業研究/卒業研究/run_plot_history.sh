#!/usr/bin/env bash
# 機能:
#   - 指定ユーザー×スキルの学習遷移を CSV / PNG / PDF で出力
#
# 使い方:
#   ./run_plot_history.sh <user_id> <skill_name|overall> [log_csv] [params_csv] [out_dir] [skill_col]
#   例:
#     ./run_plot_history.sh user000 基本的なファイル管理
#     ./run_plot_history.sh user000 overall
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: ./run_plot_history.sh <user_id> <skill_name|overall> [log_csv] [params_csv] [out_dir] [skill_col]"
  exit 1
fi

USER_ID="$1"
SKILL_NAME="$2"
LOG_CSV="${3:-csv/sim_logs.csv}"
PARAMS_CSV="${4:-csv/bkt_params_multi.csv}"
OUT_DIR="${5:-runs/history_plots}"
SKILL_COL="${6:-}"

if [ -z "${SKILL_COL}" ]; then
  HEADER="$(head -n 1 "${LOG_CSV}" 2>/dev/null || true)"
  if echo "${HEADER}" | grep -q '\bL2\b'; then
    SKILL_COL="L2"
  elif echo "${HEADER}" | grep -q '\bdomain\b'; then
    SKILL_COL="domain"
  else
    SKILL_COL="L2"
  fi
fi

if [ "${SKILL_NAME}" = "overall" ] || [ "${SKILL_NAME}" = "OVERALL" ]; then
  python -m code.plot_user_history \
    --log-csv "${LOG_CSV}" \
    --params-csv "${PARAMS_CSV}" \
    --user-id "${USER_ID}" \
    --include-overall \
    --skill-col "${SKILL_COL}" \
    --order-col timestamp \
    --correct-col correct \
    --out-dir "${OUT_DIR}"
else
  python -m code.plot_user_history \
    --log-csv "${LOG_CSV}" \
    --params-csv "${PARAMS_CSV}" \
    --user-id "${USER_ID}" \
    --skills "${SKILL_NAME}" \
    --skill-col "${SKILL_COL}" \
    --order-col timestamp \
    --correct-col correct \
    --out-dir "${OUT_DIR}"
fi
