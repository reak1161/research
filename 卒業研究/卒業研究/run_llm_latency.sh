#!/usr/bin/env bash
# 機能:
#   - json_history の LLM 応答時間 latency_ms を集計し平均を表示
#   - event/model/user 別の統計も表示
#
# 使い方:
#   ./run_llm_latency.sh [json_dir] [out_tsv]
#   例:
#     ./run_llm_latency.sh runs/json_history runs/llm_latency_report.tsv
set -euo pipefail

JSON_DIR="${1:-runs/json_history}"
OUT_TSV="${2:-runs/llm_latency_report.tsv}"

python -m code.report_llm_latency \
  --json-dir "${JSON_DIR}" \
  --out-tsv "${OUT_TSV}"

