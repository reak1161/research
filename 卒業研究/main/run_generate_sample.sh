#!/usr/bin/env bash
# 機能:
#   - 問題マスタから疑似学習ログ（sim_logs.csv）を生成
#   - BKT/IRT 推定の入力データを準備
#
# 使い方:
#   ./run_generate_sample.sh [n_users] [interactions_per_user] [seed]
#   例:
#     ./run_generate_sample.sh 80 120 42
set -euo pipefail

N_USERS="${1:-80}"
INTERACTIONS="${2:-120}"
SEED="${3:-42}"

echo "[1/1] Generate sample logs"
python -m code.simulate_user_logs \
  --items-csv csv/items_sample_lpic_tier.csv \
  --items-domain-col L2 \
  --out-csv csv/sim_logs.csv \
  --n-users "${N_USERS}" \
  --interactions-per-user "${INTERACTIONS}" \
  --seed "${SEED}"

echo "Done:"
echo "  - csv/sim_logs.csv"
echo
echo "Usage:"
echo "  ./run_generate_sample.sh [n_users] [interactions_per_user] [seed]"
