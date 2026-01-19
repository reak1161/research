#!/usr/bin/env python
"""Simple CLI tutor that updates BKT mastery on-the-fly.

機能:
    - CSV の問題マスタを読み込み、ドメイン（スキル）単位に出題
    - ユーザー回答を受けて BKT 状態をリアルタイム更新し、ログ CSV に追記
    - 希望時は LLM 連携用ペイロードをその場で表示（ダミー連携にも利用可能）

使い方例:
    python -m code.interactive_lpic_quiz \
        --user-id u001 \
        --items-csv csv/items_sample_lpic_tier.csv \
        --bkt-params-csv csv/bkt_params_multi.csv \
        --log-csv csv/sim_online_logs.csv \
        --init-log-csv csv/sim_logs.csv \
        --init-log-csv csv/sim_online_logs.csv \
        --max-questions 5 \
        --emit-llm-payload \
        --use-llm \
        --llm-model gemini-2.5-flash \
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

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from code import bkt_core  # type: ignore
    from code.data_io import ensure_directory  # type: ignore
    from code.llm_payload import CandidateItem, build_llm_payload  # type: ignore
    from code.llm_client import OpenAIClient  # type: ignore
    from code.mood import MoodThresholds, get_mood_state, mood_policy, select_candidates_by_mood  # type: ignore
else:
    from . import bkt_core
    from .data_io import ensure_directory
    from .llm_payload import CandidateItem, build_llm_payload
    from .llm_client import OpenAIClient
    from .mood import MoodThresholds, get_mood_state, mood_policy, select_candidates_by_mood


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
    ap.add_argument("--llm-model", default="gemini-2.5-flash")
    ap.add_argument("--llm-timeout", type=float, default=30.0)
    ap.add_argument("--llm-max-retries", type=int, default=3)
    ap.add_argument("--llm-min-interval", type=float, default=1.0, help="呼び出し間隔の下限（秒）")
    ap.add_argument("--llm-dry-run", action="store_true", help="LLM を呼ばずにペイロードだけ表示")
    ap.add_argument("--mood-window", type=int, default=5, help="直近何問で mood を判定するか")
    ap.add_argument("--mood-thr-low", type=float, default=0.3)
    ap.add_argument("--mood-thr-high", type=float, default=0.8)
    ap.add_argument("--mood-consec-wrong", type=int, default=3)
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
        df = pd.read_csv(log_path)
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
    filename = f"{ts}_item{item_id}_{event}.json"
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
    """Load recent history for mood estimation."""
    if not log_path.exists():
        return []
    df = pd.read_csv(log_path)
    if "user_id" not in df.columns or "correct" not in df.columns:
        return []
    df = df[df["user_id"] == user_id]
    if df.empty:
        return []
    df = df.sort_values("timestamp") if "timestamp" in df.columns else df
    history = list(zip(df["correct"].astype(int).tolist(), [None] * len(df)))
    return history[-n:]


def get_candidate_pool(
    items: pd.DataFrame,
    asked: set,
    *,
    log_path: Path,
    user_id: str,
    limit: int,
) -> pd.DataFrame:
    remaining = items[~items["item_id"].isin(asked)]
    if not remaining.empty:
        return remaining.head(limit)
    if not log_path.exists():
        return items.head(limit)
    hist = pd.read_csv(log_path)
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
        "あなたは学習支援システムのために\n"
        "「出題候補から1問だけ選ぶアシスタント」です。\n\n"
        "回答形式は必ず JSON オブジェクト 1つだけにしてください。\n"
        "日本語の文章だけの出力や、JSON の前後にコメントを書いてはいけません。\n\n"
        "出力形式:\n"
        '{"choice_index": 0, "reason_ja": "...", "motivation_ja": "..."}\n\n'
        "- choice_index: candidates_indexed 配列の index（整数）\n"
        "- reason_ja   : 教員・開発者向けの、日本語での選定理由。\n"
        "                可能であれば以下を含めてください:\n"
        "                - 選んだ問題の domain_ja（分野名）\n"
        "                - tier と difficulty_tag（例:「ファイル操作のティア2で、やや難しめ」）\n"
        "                - 推定正答確率や最近の正答率との関係（「今の力より少し上」など）\n"
        "                - 注意: P(L)_current は習得度であり、正答確率は P(correct_now) を使う\n"
        "- motivation_ja:\n"
        "    学習者向けの短い日本語コメント(1〜2文)。\n"
        "    次のルールに従ってください。\n"
        "    - 必ず内容に触れる（例: 「cp コマンド」や「パーミッション」などのキーワードを入れる）\n"
        "    - mood_state に応じてトーンを変える:\n"
        "        * FRUSTRATED: 優しくねぎらい、失敗を責めず、「一歩ずつ進んでいる」ことを伝える\n"
        "        * NORMAL    : 落ち着いて前向きに、「この問題を解くと何ができるようになるか」を伝える\n"
        "        * CONFIDENT : 少しチャレンジングな言い方で、「次のステップに進んでいる」感を出す\n"
        "    - 「次回も頑張りましょう」「落ち込まなくて大丈夫です」など、\n"
        "      同じような定型フレーズを繰り返さないように表現を変えてください。\n"
        "    - 40文字前後を目安に、読みやすい長さにしてください。\n\n"
        "[assistant]\n"
        "了解しました。JSON だけで応答します。\n\n"
        "[user]\n"
        "次の JSON を読み、最適と思う choice_index と\n"
        "選定理由(reason_ja)、学習者のモチベーションが上がる短いコメント(motivation_ja)を返してください。\n"
        "choice_index は candidates_indexed の index を使ってください。\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def build_selection_payload(
    *,
    base_payload: dict,
    candidates: List[CandidateItem],
    items_df: pd.DataFrame,
) -> dict:
    tier_map = {}
    if "tier" in items_df.columns:
        tier_map = items_df.set_index("item_id")["tier"].to_dict()
    indexed = []
    for idx, cand in enumerate(candidates):
        tier_val = tier_map.get(str(cand.item_id))
        indexed.append(
            {
                "index": idx,
                "item_id": cand.item_id,
                "domain": cand.domain,
                "tier": int(tier_val) if pd.notna(tier_val) else None,
                "p_final": round(float(cand.p_final), 4),
                "question_text": cand.question_text,
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
    mood_state: str,
    tier: str | None,
    recent_correct_rate: float | None,
    choices: List[str],
    user_choice_text: str | None,
    correct_choice_text: str | None,
) -> str:
    outcome = "正解" if is_correct else "不正解"
    rate_text = f"{recent_correct_rate:.3f}" if recent_correct_rate is not None else "N/A"
    tier_text = tier if tier is not None else "N/A"
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
        "[system]\n"
        "あなたは学習者にフィードバックを返すチューターです。\n\n"
        "必ず JSON オブジェクト 1つだけを出力してください。\n"
        "形式:\n"
        '{"feedback_ja": "...", "extra_ja": "..."}\n\n'
        "- feedback_ja:\n"
        "    学習者の感情に寄り添う1〜2文の短いコメント。\n"
        "    ルール:\n"
        "    - is_correct が true のとき:\n"
        "        * 正解を素直にほめる\n"
        "        * どこが良かったかを1か所具体的に述べる\n"
        "        * mood_state が CONFIDENT のときは\n"
        "          「次はもう一段難しい問題に挑戦しよう」などチャレンジを促す表現も可\n"
        "    - is_correct が false のとき:\n"
        "        * 責めない。まず「ここを押さえれば大丈夫」という安心感を伝える\n"
        "        * 可能であれば、学習者の回答のどこが惜しかったのかを1点だけ触れる\n"
        "    - 定型的な励まし表現のみにはしない。\n"
        "      毎回、問題の内容（ドメインやコマンド名）に少なくとも1回は触れること。\n\n"
        "- extra_ja:\n"
        "    1〜3文程度の簡潔な補足説明。\n"
        "    - 「なぜその答えが正解／不正解なのか」\n"
        "    - 「Linux の実務でどんな場面で使うか」\n"
        "    - 「次に確認してほしいポイント」\n"
        "    のうち 1〜2 個を説明してください。\n\n"
        "[user]\n"
        "次の情報を参考に、JSON 形式でフィードバックを返してください。\n"
        f"質問: {question_text}\n"
        f"{choice_lines}"
        f"{user_choice_line}"
        f"{correct_choice_line}"
        f"判定: {outcome}\n"
        f"解説: {explanation}\n"
        f"mood_state: {mood_state}\n"
        f"tier: {tier_text}\n"
        f"recent_correct_rate: {rate_text}\n"
        f"is_correct: {str(is_correct).lower()}\n"
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
    thresholds = MoodThresholds(
        window=args.mood_window,
        thr_low=args.mood_thr_low,
        thr_high=args.mood_thr_high,
        max_consec_wrong=args.mood_consec_wrong,
    )

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
        if args.use_llm and llm_client and preferred_item_id is None and step < (args.max_questions - 1):
            history = load_history(log_path, args.user_id, thresholds.window)
            mood = get_mood_state(history, thresholds=thresholds)
            policy = mood_policy(mood)
            remaining_candidates: List[CandidateItem] = []
            for _, row in get_candidate_pool(
                items,
                asked,
                log_path=log_path,
                user_id=args.user_id,
                limit=15,
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
            selected = select_candidates_by_mood(
                [c.to_dict() for c in remaining_candidates],
                mood=mood,
                k=args.llm_top_k,
            )
            if selected:
                selected_items = [
                    CandidateItem(
                        item_id=str(c["item_id"]),
                        domain=str(c["domain"]),
                        p_final=float(c["p_final"]),
                        difficulty=c.get("difficulty"),
                        tags=c.get("tags"),
                        question_text=c.get("question_text"),
                    )
                    for c in selected
                ]
            else:
                selected_items = remaining_candidates[: args.llm_top_k]
            base_payload = build_llm_payload(
                user_id=str(args.user_id),
                mode="online",
                user_state=states,
                candidates=selected_items,
                k=len(selected_items) if selected_items else args.llm_top_k,
                mood_state=mood,
                policy_hint={
                    "target_p_final": policy.target,
                    "range": {"low": policy.low, "high": policy.high},
                    "comment_ja": policy.comment_ja,
                },
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
        # 現在の習得度と、この問題（同ドメイン）の予測正答確率を表示
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

        if args.use_llm and llm_client:
            explanation = str(item.get("explanation", "")).strip()
            history = load_history(log_path, args.user_id, thresholds.window)
            mood = get_mood_state(history, thresholds=thresholds)
            recent_correct_rate = None
            if history:
                recent_correct_rate = sum(int(c) for c, _ in history) / len(history)
            tier_val = item.get("tier")
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
                mood_state=mood,
                tier=str(tier_val) if tier_val is not None else None,
                recent_correct_rate=recent_correct_rate,
                choices=choices,
                user_choice_text=user_choice_text,
                correct_choice_text=correct_choice_text,
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
            history = load_history(log_path, args.user_id, thresholds.window)
            mood = get_mood_state(history, thresholds=thresholds)
            policy = mood_policy(mood)
            remaining_candidates: List[CandidateItem] = []
            for _, row in get_candidate_pool(
                items,
                asked,
                log_path=log_path,
                user_id=args.user_id,
                limit=15,
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
            selected = select_candidates_by_mood(
                [c.to_dict() for c in remaining_candidates],
                mood=mood,
                k=args.llm_top_k,
            )
            # select_candidates_by_mood returns dicts, rebuild CandidateItem for payload
            if selected:
                selected_items = [
                    CandidateItem(
                        item_id=str(c["item_id"]),
                        domain=str(c["domain"]),
                        p_final=float(c["p_final"]),
                        difficulty=c.get("difficulty"),
                        tags=c.get("tags"),
                        question_text=c.get("question_text"),
                    )
                    for c in selected
                ]
            else:
                # fallback: use top of remaining candidates without mood filter
                selected_items = remaining_candidates[: args.llm_top_k]
            base_payload = build_llm_payload(
                user_id=str(args.user_id),
                mode="online",
                user_state=states,
                candidates=selected_items,
                k=len(selected_items) if selected_items else args.llm_top_k,
                mood_state=mood,
                policy_hint={
                    "target_p_final": policy.target,
                    "range": {"low": policy.low, "high": policy.high},
                    "comment_ja": policy.comment_ja,
                },
            )
            payload = build_selection_payload(
                base_payload=base_payload,
                candidates=selected_items,
                items_df=items,
            )
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
