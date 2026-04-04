#!/usr/bin/env python
"""Simple CLI tutor that updates BKT mastery on-the-fly.

機能:
    - CSV の問題マスタを読み込み、ドメイン（スキル）単位に出題
    - ユーザー回答を受けて BKT 状態をリアルタイム更新し、ログ CSV に追記
    - 希望時は LLM 連携用ペイロードをその場で表示（ダミー連携にも利用可能）

使い方例:
    python -m code.interactive_lpic_quiz \
        --user-id user000 \
        --items-csv csv/items_sample_lpic_tier.csv \
        --bkt-params-csv csv/bkt_params_multi.csv \
        --log-csv csv/sim_online_logs.csv \
        --init-log-csv csv/sim_logs.csv \
        --init-include-log \
        --irt-items-csv csv/irt_items_estimated.csv \
        --pfinal-mode realtime \
        --max-questions 5 \
        --use-llm \
        --llm-model gpt-4o-mini \
        --llm-top-k 8 \
        --llm-min-interval 1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pandas.errors import EmptyDataError

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from code import bkt_core  # type: ignore
    from code.data_io import ensure_directory  # type: ignore
    from code.llm_payload import CandidateItem, build_llm_payload  # type: ignore
    from code.llm_client import OpenAIClient  # type: ignore
else:
    from . import bkt_core
    from .data_io import ensure_directory
    from .llm_payload import CandidateItem, build_llm_payload
    from .llm_client import OpenAIClient


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
    ap.add_argument("--correct-text-col", default="correct_text")
    ap.add_argument("--scores-csv", help="offline_predict_scores の出力を指定すると P_final を候補に利用")
    ap.add_argument("--bkt-params-csv", required=True)
    ap.add_argument("--irt-items-csv", help="IRT 項目パラメータ CSV（a,b,c がある場合はリアルタイムで P_irt を計算）")
    ap.add_argument("--w-bkt", type=float, default=0.6, help="P_final = w_bkt*P_bkt + (1-w_bkt)*P_irt の重み")
    ap.add_argument(
        "--pfinal-mode",
        choices=["offline", "realtime", "blend"],
        default="offline",
        help="P_final の計算モード: offline=スコアCSV優先, realtime=その場で再計算, blend=両方を重み付け",
    )
    ap.add_argument("--pfinal-blend-offline-weight", type=float, default=0.5, help="blend 時のオフライン重み（1-値がリアルタイム側）")
    ap.add_argument("--log-csv", required=True)
    ap.add_argument("--init-log-csv", action="append", help="初期状態を作るための過去ログ CSV（複数指定可）")
    ap.add_argument("--init-include-log", action="store_true", help="--log-csv も初期化に含める")
    ap.add_argument("--init-order-col", default="timestamp")
    ap.add_argument("--init-user-col", default="user_id")
    ap.add_argument("--init-skill-col", default=None, help="初期化に使うスキル列（未指定なら domain-col）")
    ap.add_argument("--init-correct-col", default="correct")
    ap.add_argument("--max-questions", type=int, default=5)
    ap.add_argument("--emit-llm-payload", action="store_true", help="Print payload preview each step")
    ap.add_argument("--use-llm", action="store_true", help="LLM へ Top-K 候補を送って 1 問選ばせる")
    ap.add_argument("--llm-top-k", type=int, default=8, help="LLM に渡す候補数")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--llm-timeout", type=float, default=30.0)
    ap.add_argument("--llm-max-retries", type=int, default=3)
    ap.add_argument("--llm-min-interval", type=float, default=1.0, help="呼び出し間隔の下限（秒）")
    ap.add_argument("--llm-dry-run", action="store_true", help="LLM を呼ばずにペイロードだけ表示")
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
        args.correct_text_col: "correct_text",
    }
    df = df.rename(columns=rename_map)
    # choice_A, choice_B ... がバラで入っている場合は結合する
    if "choices" not in df.columns:
        choice_cols = [c for c in df.columns if c.lower().startswith("choice_")]
        if choice_cols:
            def _row_choices(row: pd.Series) -> list[str]:
                vals: list[str] = []
                for c in sorted(choice_cols):
                    val = row.get(c)
                    if isinstance(val, str) and val.strip():
                        vals.append(val.strip())
                return vals
            df["choices"] = df.apply(_row_choices, axis=1)

    required = ["item_id", "domain", "question_text", "answer_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Items CSV missing {missing}")
    # item_id を文字列として扱い、後段の比較ぶれを防ぐ
    df["item_id"] = df["item_id"].astype(str)
    return df


def load_irt_items(path: Optional[str]) -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warn] IRT items CSV not found: {path}")
        return {}
    df = pd.read_csv(p)
    required = {"item_id", "a", "b", "c"}
    missing = required - set(df.columns)
    if missing:
        print(f"[warn] IRT items CSV missing {missing}, ignoring.")
        return {}
    df["item_id"] = df["item_id"].astype(str)
    return {row["item_id"]: {"a": float(row["a"]), "b": float(row["b"]), "c": float(row["c"])} for _, row in df.iterrows()}


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


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


def load_scores_map(path: Optional[str], user_id: str) -> Dict[str, float]:
    """Load offline_predict_scores output and filter for the target user."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warn] scores-csv not found: {path}")
        return {}
    df = pd.read_csv(p)
    required = {"user_id", "item_id", "P_final"}
    missing = required - set(df.columns)
    if missing:
        print(f"[warn] scores-csv missing columns {missing}, ignoring {path}")
        return {}
    df = df[df["user_id"].astype(str) == str(user_id)]
    if df.empty:
        print(f"[info] scores-csv has no rows for user_id={user_id}")
        return {}
    result = {str(row["item_id"]): float(row["P_final"]) for _, row in df.iterrows()}
    print(f"[info] loaded {len(result)} scores for user_id={user_id} from {path}")
    return result


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


def normalize_text_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def read_log_csv_robust(path: Path) -> pd.DataFrame:
    """
    Read log CSV robustly.
    - Normal case: header exists.
    - Fallback: if header is missing and file has 5 columns, map to
      timestamp,user_id,item_id,domain,correct.
    """
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()

    required = {"timestamp", "user_id", "item_id", "domain", "correct"}
    if required.issubset(df.columns):
        return df

    # Fallback for headerless online log rows
    try:
        raw = pd.read_csv(path, header=None)
    except EmptyDataError:
        return pd.DataFrame()
    if raw.shape[1] >= 5:
        raw = raw.iloc[:, :5].copy()
        raw.columns = ["timestamp", "user_id", "item_id", "domain", "correct"]
        return raw
    return df


def initialize_states_from_log(
    log_csvs: List[str],
    params: Dict[str, bkt_core.BKTParams],
    *,
    user_id: str,
    user_col: str,
    skill_col: str,
    correct_col: str,
    order_col: str,
) -> tuple[Dict[str, float], set]:
    frames = []
    for log_csv in log_csvs:
        log_path = Path(log_csv)
        if not log_path.exists():
            print(f"[warn] init-log-csv not found: {log_csv}")
            continue
        df = read_log_csv_robust(log_path)
        if df.empty:
            print(f"[warn] init-log-csv is empty; skip {log_csv}.")
            continue
        required = {user_col, skill_col, correct_col}
        missing = required - set(df.columns)
        if missing:
            print(f"[warn] init-log-csv missing columns {missing}; skip {log_csv}.")
            continue
        frames.append(df)
    if not frames:
        return {domain: params[domain].L0 for domain in params}, set()
    df = pd.concat(frames, ignore_index=True)
    df = df[df[user_col].astype(str) == str(user_id)]
    if df.empty:
        return {domain: params[domain].L0 for domain in params}, set()
    if order_col in df.columns:
        df = df.sort_values(order_col)
    states: Dict[str, float] = {domain: params[domain].L0 for domain in params}
    asked: set = set()
    for _, row in df.iterrows():
        skill = str(row[skill_col])
        if skill not in params:
            continue
        correct = int(row[correct_col])
        state_info = bkt_core.update_state(states.get(skill, params[skill].L0), params[skill], correct)
        states[skill] = state_info["p_L_after"]
        if "item_id" in row:
            asked.add(str(row["item_id"]))
    print(f"[info] initialized states from {log_csv} (rows={len(df)})")
    return states, asked


def choose_next_item(
    items: pd.DataFrame,
    states: Dict[str, float],
    asked: set,
    preferred_item_id: Optional[str] = None,
    *,
    history_path: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> Optional[pd.Series]:
    if preferred_item_id is not None:
        preferred_rows = items[items["item_id"] == str(preferred_item_id)]
        if not preferred_rows.empty:
            return preferred_rows.iloc[0]
    for domain, _ in sorted(states.items(), key=lambda kv: kv[1]):
        pool = items[(items["domain"] == domain) & (~items["item_id"].isin(asked))]
        if not pool.empty:
            return pool.iloc[0]
    remaining = items[~items["item_id"].isin(asked)]
    if remaining.empty:
        if history_path and user_id and history_path.exists():
            hist = pd.read_csv(history_path)
            if {"user_id", "item_id", "correct"} <= set(hist.columns):
                hist = hist[hist["user_id"].astype(str) == str(user_id)]
                if not hist.empty:
                    wrong_rates = (
                        hist.groupby("item_id")["correct"]
                        .apply(lambda s: 1.0 - s.astype(int).mean())
                        .sort_values(ascending=False)
                    )
                    for item_id in wrong_rates.index.astype(str).tolist():
                        row = items[items["item_id"] == str(item_id)]
                        if not row.empty:
                            return row.iloc[0]
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


def write_llm_history(
    *,
    user_id: str,
    item_id: str,
    event: str,
    payload: Optional[dict],
    prompt: str,
    response: Optional[dict],
    model: Optional[str],
    latency_ms: Optional[float],
) -> None:
    safe_user = "".join(ch for ch in str(user_id) if ch.isalnum() or ch in ("-", "_")) or "user"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_item = str(item_id).replace(":", "-")
    filename = f"{ts}_item{safe_item}_{event}.json"
    out_path = Path("runs") / "json_history" / safe_user / filename
    ensure_directory(out_path)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": str(user_id),
        "item_id": str(item_id),
        "event": event,
        "model": model,
        "latency_ms": latency_ms,
        "payload": payload,
        "prompt": prompt,
        "response": response,
    }
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def build_payload_preview(user_id: str, states: Dict[str, float], candidates: List[CandidateItem]) -> None:
    payload = build_llm_payload(user_id=user_id, mode="online", user_state=states, candidates=candidates, k=len(candidates))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def load_history(log_path: Path, user_id: str, n: int) -> List[Tuple[int, Optional[float]]]:
    """Load recent (correct, placeholder_score) tuples."""
    if not log_path.exists():
        return []
    df = read_log_csv_robust(log_path)
    if df.empty:
        return []
    if "user_id" not in df.columns or "correct" not in df.columns:
        return []
    df = df[df["user_id"] == user_id]
    if df.empty:
        return []
    df = df.sort_values("timestamp") if "timestamp" in df.columns else df
    history = list(zip(df["correct"].astype(int).tolist(), [None] * len(df)))
    return history[-n:]


def load_user_log_df(log_path: Path, user_id: str) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame()
    df = read_log_csv_robust(log_path)
    if df.empty:
        return pd.DataFrame()
    required = {"user_id", "correct"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = df[df["user_id"].astype(str) == str(user_id)].copy()
    if df.empty:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df


def build_recent_learning_context(log_path: Path, user_id: str) -> dict:
    # LLM選択用の直近学習文脈をオンラインログから構築する。
    # 「全体傾向」と「分野別傾向」を分けて渡すことで、説明理由の一貫性を上げる。
    ctx = {
        "last_answer_correct": False,
        "overall_streak": {"correct": 0, "incorrect": 0},
        "recent_correct_rate_5": None,
        "domain_streak": {},
        "domain_recent_correct_rate_5": {},
        "recent_domains_5": [],
    }
    df = load_user_log_df(log_path, user_id)
    if df.empty:
        return ctx
    if "domain" in df.columns:
        recent_domains = (
            df["domain"]
            .dropna()
            .astype(str)
            .tail(5)
            .tolist()
        )
        ctx["recent_domains_5"] = recent_domains
    last_correct = int(df["correct"].astype(int).iloc[-1])
    ctx["last_answer_correct"] = bool(last_correct)
    all_flags = df["correct"].astype(int).tolist()
    recent5_all = all_flags[-5:]
    if recent5_all:
        ctx["recent_correct_rate_5"] = round(float(sum(recent5_all) / len(recent5_all)), 4)

    overall_correct = 0
    overall_incorrect = 0
    for val in reversed(all_flags):
        if val == 1 and overall_incorrect == 0:
            overall_correct += 1
        elif val == 0 and overall_correct == 0:
            overall_incorrect += 1
        else:
            break
    ctx["overall_streak"] = {
        "correct": int(overall_correct),
        "incorrect": int(overall_incorrect),
    }
    if "domain" not in df.columns:
        return ctx

    for domain, group in df.groupby("domain"):
        domain_str = str(domain)
        flags = group["correct"].astype(int).tolist()
        if not flags:
            continue
        recent5 = flags[-5:]
        ctx["domain_recent_correct_rate_5"][domain_str] = round(float(sum(recent5) / len(recent5)), 4)

        streak_correct = 0
        streak_incorrect = 0
        for val in reversed(flags):
            if val == 1 and streak_incorrect == 0:
                streak_correct += 1
            elif val == 0 and streak_correct == 0:
                streak_incorrect += 1
            else:
                break
        ctx["domain_streak"][domain_str] = {
            "correct": int(streak_correct),
            "incorrect": int(streak_incorrect),
        }
    return ctx


def build_selection_policy(ctx: dict) -> tuple[dict, bool]:
    # ヒューリスティックな方針生成:
    # - 不調時: 当たりやすさ寄り
    # - 好調時: 範囲拡張寄り
    # explore_mode は「分野探索を強めるか」のフラグ。
    overall = ctx.get("overall_streak", {}) or {}
    streak_c = int(overall.get("correct", 0))
    streak_i = int(overall.get("incorrect", 0))
    rate = ctx.get("recent_correct_rate_5")
    if rate is None:
        return (
            {
                "target_p_final": 0.70,
                "range": {"low": 0.55, "high": 0.85},
                "comment_ja": "初期は中程度の難易度で幅広く把握します。",
            },
            False,
        )
    if streak_i >= 2 or float(rate) < 0.4:
        return (
            {
                "target_p_final": 0.72,
                "range": {"low": 0.62, "high": 0.88},
                "comment_ja": "全体不調のため、当たりやすさを優先します。",
            },
            False,
        )
    if streak_c >= 2 or float(rate) > 0.75:
        return (
            {
                "target_p_final": 0.62,
                "range": {"low": 0.48, "high": 0.78},
                "comment_ja": "全体好調のため、範囲拡張を優先します。",
            },
            True,
        )
    return (
        {
            "target_p_final": 0.68,
            "range": {"low": 0.55, "high": 0.82},
            "comment_ja": "通常状態として、難易度と分野のバランスを取ります。",
        },
        False,
    )


def compute_streak(log_path: Path, user_id: str) -> int:
    if not log_path.exists():
        return 0
    df = read_log_csv_robust(log_path)
    if df.empty:
        return 0
    if "user_id" not in df.columns or "correct" not in df.columns:
        return 0
    df = df[df["user_id"] == user_id]
    if df.empty:
        return 0
    df = df.sort_values("timestamp") if "timestamp" in df.columns else df
    streak = 0
    for c in reversed(df["correct"].tolist()):
        if int(c) == 1:
            streak += 1
        else:
            break
    return streak


def is_first_time_correct(log_path: Path, user_id: str, item_id: str) -> bool:
    if not log_path.exists():
        return False
    df = read_log_csv_robust(log_path)
    if df.empty:
        return False
    required = {"user_id", "item_id", "correct"}
    if not required.issubset(df.columns):
        return False
    df = df[df["user_id"] == user_id]
    df = df[df["item_id"].astype(str) == str(item_id)]
    if df.empty:
        return False
    # True if no prior correct before the latest attempt
    corrects = df["correct"].astype(int).tolist()
    return sum(corrects) == 1 and corrects[-1] == 1


def get_candidate_pool(
    items: pd.DataFrame,
    asked: set,
    *,
    log_path: Path,
    user_id: str,
    limit: int,
    states: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    remaining = items[~items["item_id"].isin(asked)]
    if not remaining.empty:
        if "domain" not in remaining.columns:
            return remaining.head(limit)
        # 未出題がある間は、分野が偏らないようにラウンドロビンで候補を作る。
        # states がある場合は習得度が低い分野を優先順にする。
        domains = remaining["domain"].astype(str).tolist()
        unique_domains = list(dict.fromkeys(domains))
        if states:
            unique_domains = sorted(unique_domains, key=lambda d: states.get(d, 0.5))
        buckets = {
            d: remaining[remaining["domain"].astype(str) == d].reset_index(drop=True)
            for d in unique_domains
        }
        picked_rows: list[pd.Series] = []
        row_idx = 0
        while len(picked_rows) < limit:
            progressed = False
            for d in unique_domains:
                bucket = buckets[d]
                if row_idx < len(bucket):
                    picked_rows.append(bucket.iloc[row_idx])
                    progressed = True
                    if len(picked_rows) >= limit:
                        break
            if not progressed:
                break
            row_idx += 1
        if picked_rows:
            return pd.DataFrame(picked_rows).reset_index(drop=True)
        return remaining.head(limit)
    # 未出題が尽きたら再出題フェーズ。
    # 直近出題が古い順に並べ、復習ローテーションを作る。
    if not log_path.exists():
        return items.head(limit)
    hist = read_log_csv_robust(log_path)
    if hist.empty:
        return items.head(limit)
    if {"user_id", "item_id", "timestamp"} <= set(hist.columns):
        hist = hist[hist["user_id"].astype(str) == str(user_id)].copy()
        if not hist.empty:
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
            last_seen = hist.groupby("item_id")["timestamp"].max()
            order = last_seen.sort_values(ascending=True).index.astype(str).tolist()
            ordered = items.set_index("item_id").reindex(order).reset_index()
            return ordered.head(limit)
    return items.head(limit)


def build_selection_prompt(payload: dict) -> str:
    return (
        "[system]\n"
        "あなたは学習支援システムのために「出題候補から1問だけ選ぶアシスタント」です。\n"
        "回答形式は必ず JSON オブジェクト 1つだけにしてください。\n"
        "JSON の前後にコメントを書いたり、日本語だけの文章を出力したりしないでください。\n\n"
        "出力形式:\n"
        '{"choice_index": 0, "reason_ja": "...", "motivation_ja": "..."}\n\n'
        "- choice_index: candidates_indexed 配列の index（整数）\n\n"
        "- reason_ja:\n"
        "  教員・開発者向けの選定理由を書いてください。次を守ってください。\n"
        "  【矛盾防止（必須）】\n"
        "  - reason_ja に書く domain/tier/p_final は必ず candidates_indexed[choice_index] と一致させてください。\n"
        "  - reason_ja の冒頭に必ず書いてください：\n"
        "    「選択: index=..., domain=..., tier=..., p_final=...」\n"
        "  - 「分野を変えた/復習/範囲拡張」の根拠は、overall か domain か recent_domains_5 のどれか1つだけ添えてください（盛りすぎない）。\n"
        "  【内容の目安】\n"
        "  - 選んだ設問の domain（分野名）\n"
        "  - tier（可能なら difficulty_tag も）\n"
        "  - p_final（正答確率）と、全体/分野の直近成績との関係（不調なら当たりやすさ、好調なら範囲拡張など）\n"
        "  - 注意: P(L)_current は習得度です。正答確率の説明は p_final（または P(correct_now)）で書いてください。\n\n"
        "- motivation_ja:\n"
        "  学習者向けの短いコメント(1〜2文)を書いてください。解答前なのでネタバレしないでください。\n"
        "  【文体】\n"
        "  - 口語で書いてください（「〜だよ」「〜だね」「〜しよう」など）。\n"
        "  - 「あなた」「君」など二人称は禁止です。\n"
        "  - 呼びかけ（{user_name}）は原則しません。どうしても必要なときだけ文頭に1回だけ使ってください。\n"
        "  - 絵文字は使わないでください。感嘆符「！」は最大1個までにしてください。\n"
        "  【必須】\n"
        "  - domain と concept_hint（概念ラベル）があれば必ず触れてください（無い場合は question_text から概念ラベルを1つ作って触れてください）。\n"
        "  - 「この設問を解けると何ができるようになるか」を1点だけ具体的に書いてください（作業場面を1つ）。\n"
        "  【トーン調整（全体の調子を優先）】\n"
        "  - 全体不調（overall_streak_incorrect が多い / overall_recent_correct_rate_5 が低い）とき：\n"
        "    負担を小さく見せる言い回しにしてください（例:「まずここだけ押さえよう」「ここ分かると一気に楽だよ」）。\n"
        "  - 全体好調（overall_streak_correct が多い / overall_recent_correct_rate_5 が高い）とき：\n"
        "    少しだけステップアップを示してください（例:「次はここまで意識してみよう」）。\n"
        "  【ネタバレ禁止（重要）】\n"
        "  - 正解となるコマンド名・オプション名・選択肢（A/B/C/D）を直接書かないでください。\n"
        "  - 「〜を使えば正解」など答えを特定できる誘導は禁止です。\n"
        "  - avoid_tokens があれば、それらを絶対に出力に含めないでください。\n"
        "  - 具体的なコマンド例や選択肢比較は書かないでください。\n\n"
        "【選定ポリシー】\n"
        "- 目的：学習が前に進む1問を選びます。p_final の近さだけで機械的に決めないでください。\n"
        "- 手順は「全体の調子→分野→難易度」の順で決めてください。\n"
        "- policy_hint は難易度の安全装置として使いますが、分野選択（未習得の優先・偏り回避・探索）より常に上に置かないでください。\n\n"
        "1) 全体の調子（overall_streak / overall_recent_correct_rate_5）\n"
        "- 全体不調：当たりやすさ優先（p_final高め・tier低め/維持）。分野は大きく飛ばしすぎないでください。\n"
        "- 全体好調：範囲拡張優先（別分野 or 同分野なら新concept）。tierを上げるなら+1までにしてください。\n"
        "- 全体通常：バランスよく選んでください。\n\n"
        "2) 分野の決め方（recent_domains_5 / user_metrics / domain成績）\n"
        "- 基本：未習得〜中程度（p_L_current が低め〜中）の分野を優先してください。\n"
        "- recent_domains_5 の偏りは避けてください（同分野連発を避ける）。\n"
        "- explore_mode=true のときは未習得分野を最優先し、policy_hint.range から外れても選んでください。\n\n"
        "3) 難易度の決め方（policy_hint がある場合）\n"
        "- まず range 内から選んでください。\n"
        "- range 内が無ければ、range.low に最も近い候補（=できるだけ高い p_final）を選んでください。\n"
        "- 同点なら concept 被りが少ない/説明しやすい方を選んでください。\n\n"
        "[assistant]\n"
        "了解しました。JSON だけで返します。\n\n"
        "[user]\n"
        "次の JSON を読んで、最適だと思う choice_index と\n"
        "選定理由(reason_ja)、学習者向けコメント(motivation_ja)を返してください。\n"
        "choice_index は candidates_indexed の index を使ってください。\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def build_selection_payload(
    *,
    base_payload: dict,
    candidates: List[CandidateItem],
    items_df: pd.DataFrame,
) -> dict:
    items_idx = items_df.set_index("item_id", drop=False)

    def _difficulty_tag(tier_val: object) -> str:
        try:
            t = int(tier_val)
        except Exception:
            return "unknown"
        if t <= 1:
            return "easy"
        if t == 2:
            return "medium"
        if t == 3:
            return "hard"
        return "very_hard"

    def _concept_hint(row: pd.Series, domain: str, question_text: str) -> str:
        l3 = normalize_text_cell(row.get("L3", ""))
        if l3:
            return l3
        if "、" in question_text:
            return question_text.split("、", 1)[0][:24]
        return (question_text[:24] if question_text else domain)

    def _avoid_tokens(row: pd.Series) -> list[str]:
        toks: list[str] = []
        ans_type = normalize_text_cell(row.get("answer_type", "")).lower()
        correct_key = normalize_text_cell(row.get("correct_key", ""))
        correct_text = normalize_text_cell(row.get("correct_text", ""))
        if correct_key:
            toks.append(correct_key)
        if ans_type == "mcq" and correct_key:
            letter = correct_key.upper()
            if len(letter) == 1 and "A" <= letter <= "D":
                choice_col = f"choice_{letter}"
                choice_text = normalize_text_cell(row.get(choice_col, ""))
                if choice_text:
                    toks.append(choice_text)
        if correct_text:
            toks.append(correct_text)
        # 重複排除
        seen: set[str] = set()
        uniq: list[str] = []
        for t in toks:
            if t and t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    indexed = []
    for idx, cand in enumerate(candidates):
        row = items_idx.loc[str(cand.item_id)] if str(cand.item_id) in items_idx.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        tier_val = row.get("tier") if row is not None else None
        q_text = cand.question_text or (normalize_text_cell(row.get("question_text", "")) if row is not None else "")
        concept_hint = _concept_hint(row, cand.domain, q_text) if row is not None else cand.domain
        indexed.append(
            {
                "index": idx,
                "item_id": cand.item_id,
                "domain": cand.domain,
                "domain_ja": cand.domain,
                "tier": int(tier_val) if pd.notna(tier_val) else None,
                "difficulty_tag": _difficulty_tag(tier_val),
                "p_final": round(float(cand.p_final), 4),
                "question_text": q_text,
                "concept_hint": concept_hint,
                "avoid_tokens": _avoid_tokens(row) if row is not None else [],
            }
        )
    base_payload = dict(base_payload)
    base_payload["candidates_indexed"] = indexed
    return base_payload


def build_feedback_prompt(
    *,
    question_text: str,
    user_answer: str,
    correct_answer: str,
    is_correct: bool,
    explanation: str,
    tier: str | None,
    recent_correct_rate: float | None,
    choices: List[str],
    user_choice_text: str | None,
    correct_choice_text: str | None,
    domain_ja: str | None,
    tag: str | None,
    streak_correct: int,
    is_first_time_correct: bool,
    question_type: str,
) -> str:
    outcome = "正解" if is_correct else "不正解"
    choice_lines = ""
    if choices:
        labeled = []
        for idx, choice in enumerate(choices, 1):
            letter = chr(ord("A") + idx - 1)
            labeled.append(f"{letter}. {choice}")
        choice_lines = "選択肢:\n" + "\n".join(labeled) + "\n\n"
    user_choice_line = ""
    if user_choice_text is not None:
        user_choice_line = f"学習者の回答: {user_answer} ({user_choice_text})\n"
    else:
        user_choice_line = f"学習者の回答: {user_answer}\n"
    correct_choice_line = ""
    if correct_choice_text is not None:
        correct_choice_line = f"正解: {correct_answer} ({correct_choice_text})\n"
    else:
        correct_choice_line = f"正解: {correct_answer}\n"

    return (
        "あなたは学習者にフィードバックを返すチューターです。\n"
        "必ず JSON オブジェクト 1つだけを出力してください。\n\n"
        "出力形式:\n"
        '{"feedback_ja": "...", "extra_ja": "..."}\n\n'
        "【文体ルール（重要）】\n"
        "- ため口（口語）で書いてください（「〜だよ」「〜だね」「〜しよう」など会話っぽく）。\n"
        "- 「あなた」「君」など二人称は禁止です。\n"
        "- 呼びかけ（{user_name}）は原則しません。どうしても必要なときだけ文頭に1回だけ使ってください。\n"
        "- 上から目線・命令口調は禁止です（例：「〜しろ」「当然」「簡単」「常識」）。\n"
        "- 空疎な一般論の慰めは禁止です（例：「誰にでもある」「大丈夫」「落ち込むな」「頑張れ」）。\n"
        "- 絵文字は使わないでください。\n"
        "- 感嘆符「！」は使ってよいですが多用しないでください（feedback_ja は最大1個、extra_ja は原則0個）。\n\n"
        "【評価の対象】\n"
        "- 評価対象はコマンドではなく理解や判断に置いてください。\n"
        "- 出力では主語は省略してOKですが、「〜できてるね」「ここで〜が混ざったね」など、理解に焦点を当ててください。\n\n"
        "【feedback_ja（1〜2文）】\n"
        "- まず短い受け止めを入れてください（例：「いいね！」「惜しいね！」）。\n"
        "- 1〜2文の中で入れてよい学習ポイントは1つだけにしてください（言いすぎないでください）。\n"
        "- 正解時：\n"
        "  * できている点を1つ述べてください。\n"
        "  * 追加で「ひっかかりやすい点／他の選択肢で間違えやすいポイント」を1点だけ軽く触れてください（理由は深掘りしないでください）。\n"
        "- 不正解時：\n"
        "  * 「選んだ答え」と「正解」の違いを要点だけで1対比してください（差分は1つだけにしてください）。\n"
        "  * 次に見るポイントを一言添えてください（正解の用語を出しても構いません）。\n\n"
        "【extra_ja（2〜4文）】\n"
        "- 解答後の解説なので、正解のコマンド名・選択肢・オプションを明示して構いません。\n"
        "- 2〜4文で、学習に役立つ説明を書いてください。\n"
        "- 次の要素から状況に合うものを「1〜2個」選んで入れてください（全部入れないでください）:\n"
        "  A) 正解の理由\n"
        "  B) 誤答との差分（誤答時は優先）\n"
        "  C) ひっかかりやすい勘違い（他の選択肢がなぜ違うか）\n"
        "  D) 覚え方・語源（例: -r=recursive, -p=preserve）\n"
        "  E) 実務の短い使用例（コマンド例は最大1つ）\n"
        "- 不正解時は、B（差分）を必ず含めてください。\n"
        "- コマンド例を出す場合は1つだけにしてください。\n\n"
        "【入力情報】\n"
        f"質問: {question_text}\n"
        f"問題タイプ: {question_type}\n"
        f"{choice_lines}"
        f"{user_choice_line}"
        f"{correct_choice_line}"
        f"判定: {outcome}\n"
        f"分野: {domain_ja or ''}\n"
        f"タグ: {tag or ''}\n"
        f"連続正解数: {streak_correct}\n"
        f"is_first_time_correct: {str(is_first_time_correct).lower()}\n"
        f"システム側の簡単な解説: {explanation}\n\n"
        "必ず {\"feedback_ja\": \"...\", \"extra_ja\": \"...\"} の形の JSON だけを返す。\n"
    )


def call_llm_with_timing(llm_client, prompt: str) -> tuple[Optional[dict], Optional[float]]:
    start = time.monotonic()
    resp = llm_client.generate_json(prompt, expect_json=True)
    elapsed_ms = (time.monotonic() - start) * 1000.0
    return resp if isinstance(resp, dict) else None, elapsed_ms


def main() -> None:
    args = parse_args()
    items = load_items(args)
    params = normalize_params(Path(args.bkt_params_csv))
    init_logs = list(args.init_log_csv or [])
    if args.init_include_log and args.log_csv:
        init_logs.append(args.log_csv)
    if init_logs:
        init_skill_col = args.init_skill_col or "domain"
        states, asked = initialize_states_from_log(
            init_logs,
            params,
            user_id=args.user_id,
            user_col=args.init_user_col,
            skill_col=init_skill_col,
            correct_col=args.init_correct_col,
            order_col=args.init_order_col,
        )
    else:
        states = {domain: params[domain].L0 for domain in params}
        asked = set()
    scores_map = load_scores_map(args.scores_csv, args.user_id)
    irt_items = load_irt_items(args.irt_items_csv)
    log_path = Path(args.log_csv)
    llm_client = None
    if args.use_llm:
        try:
            llm_client = OpenAIClient(
                model=args.llm_model,
                timeout=args.llm_timeout,
                max_retries=args.llm_max_retries,
                min_interval=args.llm_min_interval,
                dry_run=args.llm_dry_run,
            )
        except Exception as e:  # pragma: no cover - optional dependency/credential issues
            print(f"[warn] LLM client initialization failed: {e}")
            llm_client = None

    preferred_item_id: Optional[str] = None

    for step in range(args.max_questions):
        # ---- 1) 次問の事前選定（必要時のみLLM） ----
        if args.use_llm and llm_client and preferred_item_id is None and step < (args.max_questions - 1):
            remaining_candidates: List[CandidateItem] = []
            for _, row in get_candidate_pool(
                items,
                asked,
                log_path=log_path,
                user_id=args.user_id,
                limit=15,
                states=states,
            ).iterrows():
                item_id = str(row["item_id"])
                domain = str(row["domain"])
                p_bkt = states.get(domain, 0.7)
                p_offline = scores_map.get(item_id)
                p_rt = None
                if item_id in irt_items:
                    theta = logit(p_bkt)
                    itm = irt_items[item_id]
                    p_irt = itm["c"] + (1.0 - itm["c"]) / (1.0 + math.exp(-itm["a"] * (theta - itm["b"])))
                    p_rt = args.w_bkt * p_bkt + (1.0 - args.w_bkt) * p_irt
                else:
                    p_rt = p_bkt

                if args.pfinal_mode == "realtime":
                    p_final = p_rt if p_rt is not None else p_offline if p_offline is not None else p_bkt
                elif args.pfinal_mode == "blend":
                    if p_rt is None and p_offline is None:
                        p_final = p_bkt
                    elif p_rt is None:
                        p_final = p_offline
                    elif p_offline is None:
                        p_final = p_rt
                    else:
                        w_off = args.pfinal_blend_offline_weight
                        p_final = w_off * p_offline + (1 - w_off) * p_rt
                else:
                    p_final = p_offline if p_offline is not None else p_rt if p_rt is not None else p_bkt
                remaining_candidates.append(
                    CandidateItem(
                        item_id=item_id,
                        domain=domain,
                        p_final=p_final,
                        question_text=str(row.get("question_text", "")),
                    )
                )
            selected_items = remaining_candidates[: args.llm_top_k]
            base_payload = build_llm_payload(
                user_id=str(args.user_id),
                mode="online",
                user_state=states,
                candidates=selected_items,
                k=len(selected_items) if selected_items else args.llm_top_k,
            )
            base_payload["metrics_note_ja"] = "P(L)_current は習得度、P(correct_now) が正答確率です。"
            base_payload["user_metrics"] = {}
            for dom, p_L in states.items():
                params_row = params.get(dom)
                if not params_row:
                    continue
                p_correct = p_L * (1.0 - params_row.S) + (1.0 - p_L) * params_row.G
                base_payload["user_metrics"][dom] = {
                    "p_L_current": round(float(p_L), 4),
                    "p_correct_now": round(float(p_correct), 4),
                }
            payload = build_selection_payload(
                base_payload=base_payload,
                candidates=selected_items,
                items_df=items,
            )
            payload.pop("candidates", None)
            # 直近成績・方針ヒントを後付けで追加して LLM 判断材料を揃える。
            ctx = build_recent_learning_context(log_path, args.user_id)
            payload.update(ctx)
            policy_hint, explore_mode = build_selection_policy(ctx)
            payload["policy_hint"] = policy_hint
            payload["explore_mode"] = bool(explore_mode)
            # prompt 側の期待名に合わせたエイリアス
            payload["overall_recent_correct_rate_5"] = payload.get("recent_correct_rate_5")
            overall_streak = payload.get("overall_streak", {}) or {}
            payload["overall_streak_correct"] = int(overall_streak.get("correct", 0))
            payload["overall_streak_incorrect"] = int(overall_streak.get("incorrect", 0))
            if args.emit_llm_payload:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            prompt = build_selection_prompt(payload)
            try:
                resp, latency_ms = call_llm_with_timing(llm_client, prompt)
                if isinstance(resp, dict) and (("choice_index" in resp) or ("item_id" in resp)):
                    chosen = None
                    if "choice_index" in resp:
                        try:
                            idx = int(resp["choice_index"])
                            if 0 <= idx < len(selected_items):
                                chosen = selected_items[idx]
                        except (TypeError, ValueError):
                            pass
                    if chosen is None and "item_id" in resp:
                        raw_item_id = str(resp["item_id"])
                        candidate_ids = {c.item_id for c in selected_items}
                        if raw_item_id not in candidate_ids:
                            print(f"[warn] LLM item_id not in candidates: {raw_item_id}")
                        else:
                            chosen = next((c for c in selected_items if c.item_id == raw_item_id), None)
                    if chosen is None and selected_items:
                        chosen = selected_items[0]
                    if chosen:
                        preferred_item_id = chosen.item_id
                    reason = resp.get("reason_ja")
                    motivation = resp.get("motivation_ja")
                    comment = resp.get("comment_ja")
                    if reason:
                        print(f"🧠 {reason}")
                    if motivation:
                        print(f"🔥 {motivation}")
                    if comment and not (reason or motivation):
                        print(f"[info] LLM comment: {comment}")
                    if preferred_item_id:
                        tier_val = items.loc[items["item_id"] == str(preferred_item_id), "tier"]
                        tier_text = ""
                        if not tier_val.empty:
                            tier_text = f" tier={int(tier_val.iloc[0])}"
                        print(f"[info] LLM suggested item_id={preferred_item_id} domain={chosen.domain}{tier_text}")
                        if latency_ms is not None:
                            print(f"[info] LLM selection latency={latency_ms:.0f}ms")
                        write_llm_history(
                            user_id=args.user_id,
                            item_id=preferred_item_id,
                            event="selection",
                            payload=payload,
                            prompt=prompt,
                            response=resp,
                            model=args.llm_model,
                            latency_ms=latency_ms,
                        )
            except Exception as e:  # pragma: no cover - network errors
                print(f"[warn] LLM call failed: {e}")

        item = choose_next_item(
            items,
            states,
            asked,
            preferred_item_id,
            history_path=log_path,
            user_id=args.user_id,
        )
        if item is None:
            print("[info] No more items available.")
            break
        preferred_item_id = None
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
                letter = chr(ord("A") + idx - 1)
                print(f"  {idx} ({letter}) {choice}")
        # ---- 2) 出題・回答受け取り ----
        # 現在の習得度と、同ドメイン問題に対する現時点予測正答確率を表示。
        p_L_prior = states.get(domain, params[domain].L0)
        p_correct_now = p_L_prior * (1.0 - params[domain].S) + (1.0 - p_L_prior) * params[domain].G
        print(f"P(L)_current={p_L_prior:.3f}  P(correct_now)={p_correct_now:.3f}")
        user_answer = input("Your answer (type 'quit' to exit): ").strip()
        if user_answer.lower() in {"quit", "exit"}:
            break
        correct_key = normalize_text_cell(item.get("correct_key", ""))
        if not correct_key:
            correct_key = normalize_text_cell(item.get("correct_text", ""))
        answer_type = str(item.get("answer_type", "text")).lower()
        normalized = user_answer.strip()
        is_correct = normalized.lower() == correct_key.lower()
        if answer_type == "mcq":
            # map digits to letters (1->A, 2->B, ...) if choices are enumerated
            def to_letter(token: str) -> str:
                if token.isdigit():
                    try:
                        val = int(token)
                        if 1 <= val <= len(choices):
                            return chr(ord("A") + val - 1)
                    except ValueError:
                        pass
                return token.upper()

            normalized_letter = to_letter(normalized)
            correct_letter = to_letter(correct_key)
            is_correct = normalized_letter == correct_letter
        print("✅ Correct!" if is_correct else f"❌ Incorrect (correct: {correct_key})")

        # ---- 3) 解答後フィードバック（任意でLLM） ----
        if args.use_llm and llm_client:
            explanation = str(item.get("explanation", "")).strip()
            history = load_history(log_path, args.user_id, 5)
            recent_correct_rate = None
            if history:
                recent_correct_rate = sum(int(c) for c, _ in history) / len(history)
            tier_val = item.get("tier")
            streak_correct = compute_streak(log_path, args.user_id)
            first_correct = is_first_time_correct(log_path, args.user_id, str(item.get("item_id")))
            user_choice_text = None
            correct_choice_text = None
            if choices:
                def to_letter(token: str) -> str:
                    if token.isdigit():
                        try:
                            val = int(token)
                            if 1 <= val <= len(choices):
                                return chr(ord("A") + val - 1)
                        except ValueError:
                            pass
                    return token.upper()

                user_letter = to_letter(normalized)
                correct_letter = to_letter(correct_key)
                if len(user_letter) == 1 and "A" <= user_letter <= "Z":
                    idx = ord(user_letter) - ord("A")
                    if 0 <= idx < len(choices):
                        user_choice_text = choices[idx]
                if len(correct_letter) == 1 and "A" <= correct_letter <= "Z":
                    idx = ord(correct_letter) - ord("A")
                    if 0 <= idx < len(choices):
                        correct_choice_text = choices[idx]
            prompt = build_feedback_prompt(
                question_text=str(item.get("question_text", "")),
                user_answer=normalized,
                correct_answer=correct_key,
                is_correct=is_correct,
                explanation=explanation,
                tier=str(tier_val) if tier_val is not None else None,
                recent_correct_rate=recent_correct_rate,
                choices=choices,
                user_choice_text=user_choice_text,
                correct_choice_text=correct_choice_text,
                domain_ja=domain,
                tag=str(item.get("L3", "")) if "L3" in item else None,
                streak_correct=streak_correct,
                is_first_time_correct=bool(first_correct),
                question_type=str(item.get("answer_type", "text")),
            )
            try:
                resp, latency_ms = call_llm_with_timing(llm_client, prompt)
                if isinstance(resp, dict) and resp.get("feedback_ja"):
                    print(f"💬 {resp['feedback_ja']}")
                    extra = resp.get("extra_ja")
                    if extra:
                        print(f"📝 {extra}")
                elif isinstance(resp, dict) and resp.get("_raw_text"):
                    print(f"💬 {resp['_raw_text']}")
                if latency_ms is not None:
                    print(f"[info] LLM feedback latency={latency_ms:.0f}ms")
                write_llm_history(
                    user_id=args.user_id,
                    item_id=item.get("item_id"),
                    event="feedback",
                    payload={
                        "question_text": str(item.get("question_text", "")),
                        "user_answer": normalized,
                        "correct_answer": correct_key,
                        "is_correct": bool(is_correct),
                        "explanation": explanation,
                    },
                    prompt=prompt,
                    response=resp if isinstance(resp, dict) else None,
                    model=args.llm_model,
                    latency_ms=latency_ms,
                )
            except Exception as e:  # pragma: no cover - network errors
                print(f"[warn] LLM feedback failed: {e}")

        # ---- 4) BKT状態更新とログ保存 ----
        state_info = bkt_core.update_state(states.get(domain, params[domain].L0), params[domain], int(is_correct))
        states[domain] = state_info["p_L_after"]
        asked.add(str(item["item_id"]))

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
        if (args.emit_llm_payload or args.use_llm) and step < (args.max_questions - 1):
            remaining_candidates: List[CandidateItem] = []
            for _, row in get_candidate_pool(
                items,
                asked,
                log_path=log_path,
                user_id=args.user_id,
                limit=15,
                states=states,
            ).iterrows():
                item_id = str(row["item_id"])
                domain = str(row["domain"])
                p_bkt = states.get(domain, 0.7)
                p_offline = scores_map.get(item_id)
                p_rt = None
                if item_id in irt_items:
                    theta = logit(p_bkt)
                    itm = irt_items[item_id]
                    p_irt = itm["c"] + (1.0 - itm["c"]) / (1.0 + math.exp(-itm["a"] * (theta - itm["b"])))
                    p_rt = args.w_bkt * p_bkt + (1.0 - args.w_bkt) * p_irt
                else:
                    p_rt = p_bkt

                if args.pfinal_mode == "realtime":
                    p_final = p_rt if p_rt is not None else p_offline if p_offline is not None else p_bkt
                elif args.pfinal_mode == "blend":
                    if p_rt is None and p_offline is None:
                        p_final = p_bkt
                    elif p_rt is None:
                        p_final = p_offline
                    elif p_offline is None:
                        p_final = p_rt
                    else:
                        w_off = args.pfinal_blend_offline_weight
                        p_final = w_off * p_offline + (1.0 - w_off) * p_rt
                else:  # offline
                    p_final = p_offline if p_offline is not None else p_rt if p_rt is not None else p_bkt
                remaining_candidates.append(
                    CandidateItem(
                        item_id=item_id,
                        domain=domain,
                        p_final=p_final,
                        question_text=str(row.get("question_text", "")),
                    )
                )
            selected_items = remaining_candidates[: args.llm_top_k]
            base_payload = build_llm_payload(
                user_id=str(args.user_id),
                mode="online",
                user_state=states,
                candidates=selected_items,
                k=len(selected_items) if selected_items else args.llm_top_k,
            )
            payload = build_selection_payload(
                base_payload=base_payload,
                candidates=selected_items,
                items_df=items,
            )
            payload.pop("candidates", None)
            ctx = build_recent_learning_context(log_path, args.user_id)
            payload.update(ctx)
            policy_hint, explore_mode = build_selection_policy(ctx)
            payload["policy_hint"] = policy_hint
            payload["explore_mode"] = bool(explore_mode)
            payload["overall_recent_correct_rate_5"] = payload.get("recent_correct_rate_5")
            overall_streak = payload.get("overall_streak", {}) or {}
            payload["overall_streak_correct"] = int(overall_streak.get("correct", 0))
            payload["overall_streak_incorrect"] = int(overall_streak.get("incorrect", 0))
            if args.emit_llm_payload:
                print(json.dumps(payload, ensure_ascii=False, indent=2))

            if args.use_llm and llm_client:
                prompt = build_selection_prompt(payload)
                try:
                    resp, latency_ms = call_llm_with_timing(llm_client, prompt)
                    if isinstance(resp, dict) and resp.get("_raw_text"):
                        print("[warn] LLM returned non-JSON response; fallback to first candidate.")
                        if selected_items:
                            preferred_item_id = selected_items[0].item_id
                    elif isinstance(resp, dict) and (("choice_index" in resp) or ("item_id" in resp)):
                        chosen = None
                        if "choice_index" in resp:
                            try:
                                idx = int(resp["choice_index"])
                                if 0 <= idx < len(selected_items):
                                    chosen = selected_items[idx]
                            except (TypeError, ValueError):
                                pass
                        if chosen is None and "item_id" in resp:
                            raw_item_id = str(resp["item_id"])
                            candidate_ids = {c.item_id for c in selected_items}
                            if raw_item_id not in candidate_ids:
                                print(f"[warn] LLM item_id not in candidates: {raw_item_id}")
                            else:
                                chosen = next((c for c in selected_items if c.item_id == raw_item_id), None)
                        if chosen is None and selected_items:
                            chosen = selected_items[0]
                        if chosen:
                            preferred_item_id = chosen.item_id
                        reason = resp.get("reason_ja")
                        motivation = resp.get("motivation_ja")
                        comment = resp.get("comment_ja")
                        if reason:
                            print(f"🧠 {reason}")
                        if motivation:
                            print(f"🔥 {motivation}")
                        if comment and not (reason or motivation):
                            print(f"[info] LLM comment: {comment}")
                        if preferred_item_id:
                            tier_val = items.loc[items["item_id"] == str(preferred_item_id), "tier"]
                            tier_text = ""
                            if not tier_val.empty:
                                tier_text = f" tier={int(tier_val.iloc[0])}"
                            print(f"[info] LLM suggested item_id={preferred_item_id} domain={chosen.domain}{tier_text}")
                            if latency_ms is not None:
                                print(f"[info] LLM selection latency={latency_ms:.0f}ms")
                            write_llm_history(
                                user_id=args.user_id,
                                item_id=preferred_item_id,
                                event="selection",
                                payload=payload,
                                prompt=prompt,
                                response=resp,
                                model=args.llm_model,
                                latency_ms=latency_ms,
                            )
                        else:
                            print("[warn] LLM response was not a JSON with item_id; ignoring.")
                            if selected_items:
                                preferred_item_id = selected_items[0].item_id
                    else:
                        print("[warn] LLM response was not a JSON with choice_index/item_id; ignoring.")
                        if selected_items:
                            preferred_item_id = selected_items[0].item_id
                except Exception as e:  # pragma: no cover - network errors
                    print(f"[warn] LLM call failed: {e}")
                    if selected_items:
                        preferred_item_id = selected_items[0].item_id
                        preferred_item_id = selected_items[0].item_id


if __name__ == "__main__":
    main()
