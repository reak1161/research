#!/usr/bin/env bash
# 機能:
#   - json_history から問題推薦チェーン（Graphviz 図）を出力
#
# 使い方:
#   ./run_render_selection_chain.sh [json_dir] [user_id] [out_prefix] [fmt]
#   例:
#     ./run_render_selection_chain.sh runs/json_history user000 runs/selection_chain_user000 png
set -euo pipefail

JSON_DIR="${1:-runs/json_history}"
USER_ID="${2:-}"
OUT_PREFIX="${3:-runs/selection_chain}"
FMT="${4:-png}"

if [ -n "${USER_ID}" ]; then
  python -m code.render_selection_chain \
    --json_dir "${JSON_DIR}" \
    --user_id "${USER_ID}" \
    --out "${OUT_PREFIX}" \
    --fmt "${FMT}"
else
  python -m code.render_selection_chain \
    --json_dir "${JSON_DIR}" \
    --out "${OUT_PREFIX}" \
    --fmt "${FMT}"
fi

