#!/usr/bin/env python
"""Construct offline LLM payloads from user×item scores.

機能:
    - offline_predict_scores の出力からユーザーごとの候補問題をソート
    - 分野別の平均 P_bkt を user_state としてまとめる
    - LLM に渡す JSON（トップK候補＋メタ情報）を runs/ 配下へ保存

使い方例:
    python -m code.build_offline_llm_payloads \
        --scores-csv runs/sim_user_item_scores.csv \
        --out-dir runs/sim_payloads \
        --top-k 5 \
        --max-candidates 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from code.data_io import ensure_directory  # type: ignore
    from code.llm_payload import CandidateItem, build_llm_payload  # type: ignore
else:
    from .data_io import ensure_directory
    from .llm_payload import CandidateItem, build_llm_payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create offline JSON payloads for each user.")
    ap.add_argument("--scores-csv", required=True, help="Output of offline_predict_scores.py")
    ap.add_argument("--out-dir", required=True, help="Directory for per-user JSON files")
    ap.add_argument("--top-k", type=int, default=5, help="Top-K candidates to keep")
    ap.add_argument("--max-candidates", type=int, default=20, help="Initial candidate pool per user")
    return ap.parse_args()


def parse_tags(value: object) -> List[str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if value.startswith("["):
        try:
            arr = json.loads(value)
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except json.JSONDecodeError:
            pass
    return [token.strip() for token in value.split(",") if token.strip()]


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.scores_csv)
    required = {"user_id", "item_id", "domain", "P_final", "P_bkt"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Scores CSV missing columns {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_directory(out_dir / "placeholder.json")

    for user_id, group in scores.groupby("user_id"):
        user_state = group.groupby("domain")["P_bkt"].mean().to_dict()
        candidates: List[CandidateItem] = []
        sorted_group = group.sort_values("P_final", ascending=False).head(args.max_candidates)
        for _, row in sorted_group.iterrows():
            candidates.append(
                CandidateItem(
                    item_id=str(row["item_id"]),
                    domain=str(row["domain"]),
                    p_final=float(row["P_final"]),
                    difficulty=str(row["difficulty"]) if "difficulty" in row and not pd.isna(row["difficulty"]) else None,
                    tags=parse_tags(row.get("tags")),
                    question_text=str(row["question_text"]) if "question_text" in row and not pd.isna(row["question_text"]) else None,
                )
            )
        payload = build_llm_payload(user_id=user_id, mode="offline", user_state=user_state, candidates=candidates, k=args.top_k)
        out_path = out_dir / f"offline_payload_{user_id}.json"
        ensure_directory(out_path)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[info] wrote {out_path}")


if __name__ == "__main__":
    main()
