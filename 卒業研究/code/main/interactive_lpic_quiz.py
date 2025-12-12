#!/usr/bin/env python
"""Simple CLI tutor that updates BKT mastery on-the-fly."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT))
    from code.main import bkt_core  # type: ignore
    from code.main.data_io import ensure_directory  # type: ignore
    from code.main.llm_payload import CandidateItem, build_llm_payload  # type: ignore
else:
    from . import bkt_core
    from .data_io import ensure_directory
    from .llm_payload import CandidateItem, build_llm_payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive CLI tutor with BKT updates.")
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--items-csv", required=True)
    ap.add_argument("--item-id-col", default="item_id")
    ap.add_argument("--domain-col", default="L2")
    ap.add_argument("--question-col", default="question_text")
    ap.add_argument("--answer-type-col", default="answer_type")
    ap.add_argument("--choices-col", default="choices")
    ap.add_argument("--correct-col", default="correct_key")
    ap.add_argument("--bkt-params-csv", required=True)
    ap.add_argument("--irt-items-csv", help="Optional items for future use")
    ap.add_argument("--log-csv", required=True)
    ap.add_argument("--max-questions", type=int, default=5)
    ap.add_argument("--emit-llm-payload", action="store_true", help="Print payload preview each step")
    return ap.parse_args()


def load_items(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.items_csv)
    rename_map = {
        args.item_id_col: "item_id",
        args.domain_col: "domain",
        args.question_col: "question_text",
        args.answer_type_col: "answer_type",
        args.choices_col: "choices",
        args.correct_col: "correct_key",
    }
    df = df.rename(columns=rename_map)
    required = ["item_id", "domain", "question_text", "answer_type", "correct_key"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Items CSV missing {missing}")
    return df


def normalize_params(path: Path) -> Dict[str, bkt_core.BKTParams]:
    df = pd.read_csv(path)
    if "skill" in df.columns and "skill_name" not in df.columns:
        df = df.rename(columns={"skill": "skill_name"})
    missing = [c for c in ("skill_name", "L0", "T", "S", "G") if c not in df.columns]
    if missing:
        raise ValueError(f"BKT params missing {missing}")
    params: Dict[str, bkt_core.BKTParams] = {}
    for _, row in df.iterrows():
        params[str(row["skill_name"])] = bkt_core.BKTParams.from_row(row)
    return params


def parse_choices(raw: object) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                arr = json.loads(text)
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in text.split("||") if part.strip()]
    return []


def choose_next_item(items: pd.DataFrame, states: Dict[str, float], asked: set) -> Optional[pd.Series]:
    for domain, _ in sorted(states.items(), key=lambda kv: kv[1]):
        pool = items[(items["domain"] == domain) & (~items["item_id"].isin(asked))]
        if not pool.empty:
            return pool.iloc[0]
    remaining = items[~items["item_id"].isin(asked)]
    if remaining.empty:
        return None
    return remaining.iloc[0]


def log_response(path: Path, row: Dict[str, object]) -> None:
    ensure_directory(path)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_payload_preview(user_id: str, states: Dict[str, float], candidates: List[CandidateItem]) -> None:
    payload = build_llm_payload(user_id=user_id, mode="online", user_state=states, candidates=candidates, k=len(candidates))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    items = load_items(args)
    params = normalize_params(Path(args.bkt_params_csv))
    states: Dict[str, float] = {domain: params[domain].L0 for domain in params}
    asked: set = set()
    log_path = Path(args.log_csv)

    for step in range(args.max_questions):
        item = choose_next_item(items, states, asked)
        if item is None:
            print("[info] No more items available.")
            break
        domain = str(item["domain"])
        if domain not in params:
            print(f"[warn] No params for domain {domain}, skipping item {item['item_id']}.")
            asked.add(item["item_id"])
            continue
        print(f"\n=== Question {step+1} / domain={domain} ===")
        print(item["question_text"])
        choices = parse_choices(item.get("choices"))
        if choices:
            for idx, choice in enumerate(choices, 1):
                print(f"  {idx}. {choice}")
        user_answer = input("Your answer (type 'quit' to exit): ").strip()
        if user_answer.lower() in {"quit", "exit"}:
            break
        correct_key = str(item["correct_key"]).strip()
        answer_type = str(item.get("answer_type", "text")).lower()
        normalized = user_answer.strip()
        is_correct = normalized.lower() == correct_key.lower()
        if answer_type == "mcq" and normalized.isdigit():
            is_correct = normalized == correct_key or (
                correct_key.isdigit() and normalized == correct_key
            )
        print("✅ Correct!" if is_correct else f"❌ Incorrect (correct: {correct_key})")

        state_info = bkt_core.update_state(states.get(domain, params[domain].L0), params[domain], int(is_correct))
        states[domain] = state_info["p_L_after"]
        asked.add(item["item_id"])

        log_response(
            log_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": args.user_id,
                "item_id": item["item_id"],
                "domain": domain,
                "correct": int(is_correct),
            },
        )
        print(
            f"P(L)_after={state_info['p_L_after']:.3f}  P(next)={state_info['p_next']:.3f}"
        )
        if args.emit_llm_payload:
            remaining_candidates = [
                CandidateItem(
                    item_id=str(row["item_id"]),
                    domain=str(row["domain"]),
                    p_final=states.get(str(row["domain"]), 0.5),
                    question_text=str(row.get("question_text", "")),
                )
                for _, row in items[~items["item_id"].isin(asked)].head(3).iterrows()
            ]
            build_payload_preview(str(args.user_id), states, remaining_candidates)


if __name__ == "__main__":
    main()
