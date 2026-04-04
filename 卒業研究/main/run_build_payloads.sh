#!/usr/bin/env bash
# 機能:
#   - オフラインスコアからユーザーごとの LLM 用 JSON ペイロードを生成
#
# 使い方:
#   ./run_build_payloads.sh [scores_csv] [out_dir] [top_k] [max_candidates]
#   例:
#     ./run_build_payloads.sh runs/sim_user_item_scores.csv runs/sim_payloads 8 15
set -euo pipefail

SCORES_CSV="${1:-runs/sim_user_item_scores.csv}"
OUT_DIR="${2:-runs/sim_payloads}"
TOP_K="${3:-8}"
MAX_CANDIDATES="${4:-15}"

python -m code.build_offline_llm_payloads \
  --scores-csv "${SCORES_CSV}" \
  --out-dir "${OUT_DIR}" \
  --top-k "${TOP_K}" \
  --max-candidates "${MAX_CANDIDATES}"

