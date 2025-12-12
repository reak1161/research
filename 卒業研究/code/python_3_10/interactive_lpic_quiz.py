#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LPIC用ミニチューター（CLI版）

- items_lpic.csv から問題を読み込む
- answer_type に応じて入力を受け取って正誤判定
- 正誤を 0/1 にしてログ ct_lpic.csv に追記

解答形式:
  - mcq  : 単一選択 (A/B/C... or 1/2/... で回答)
  - text : 自由入力 (ここでは単純一致 or startswith で判定)

ログ形式 (ct_lpic.csv):
  timestamp, user_id, item_id, L1, L2, L3, correct
"""

import argparse
import time
from pathlib import Path
import pandas as pd
import re

# ===== パス設定 =====
ROOT = Path(__file__).resolve().parents[2]  # .../卒業研究/
CSV_DIR = ROOT / "csv"

DEFAULT_ITEMS_CSV = CSV_DIR / "items_lpic.csv"
DEFAULT_LOG_CSV = CSV_DIR / "ct_lpic.csv"


def load_items(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"item_id", "L1", "L2", "L3", "question_text",
                "answer_type", "choices", "correct_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"items CSV の列が足りません: {missing}")
    return df


# ===== 正規化系 =====
def normalize_text(s: str) -> str:
    return str(s).strip()


# ===== 判定ロジック =====
def judge_mcq(row: pd.Series, user_input: str):
    """
    単一選択の判定。
    - user_input: "A"/"a"/"1"/"2" など
    戻り値:
      True/False … 有効な入力で正解/不正解
      None        … 無効な入力（再入力させたい）
    """
    choices = [c.strip() for c in str(row["choices"]).split(",") if c.strip()]
    if not choices:
        return None

    correct_idx = int(row["correct_key"])
    s = user_input.strip()
    if not s:
        return None

    idx = None

    # 数字で答えた場合
    if s.isdigit():
        n = int(s)
        if 1 <= n <= len(choices):
            idx = n - 1  # 1始まり -> 0始まり
        else:
            idx = n  # 多分0始まりで答えたパターン
    else:
        # A/B/C... など
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if s[0].upper() in letters:
            idx = letters.index(s[0].upper())

    if idx is None or not (0 <= idx < len(choices)):
        return None

    return idx == correct_idx


def judge_text(row: pd.Series, user_input: str) -> bool:
    """
    自由入力の判定。
    ここでは最も単純に:
      - 完全一致
      - （オプションで） "<<EOF" などを許したい場合は startswith を使う
    """
    correct_raw = str(row["correct_key"])
    if not correct_raw:
        return False

    u = normalize_text(user_input)
    c = normalize_text(correct_raw)

    # ぴったり一致
    if u == c:
        return True

    # 例えば "<<EOF" なども許容したいなら:
    if c == "<<" and u.startswith("<<"):
        return True

    return False


def log_answer(log_path: Path, user_id: str | int, row: pd.Series, correct: bool):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    rec = {
        "timestamp": ts,
        "user_id": user_id,
        "item_id": int(row["item_id"]),
        "L1": row["L1"],
        "L2": row["L2"],
        "L3": row["L3"],
        "correct": int(correct),
    }
    df = pd.DataFrame([rec])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, mode="w", header=True, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True,
                        help="ユーザーID (任意の文字列/数値)")
    parser.add_argument("--items-csv", default=str(DEFAULT_ITEMS_CSV))
    parser.add_argument("--log-csv", default=str(DEFAULT_LOG_CSV))
    args = parser.parse_args()

    items = load_items(Path(args.items_csv))
    log_path = Path(args.log_csv)

    user_id = args.user_id
    print(f"=== LPIC練習セッション開始: user={user_id} ===")
    print("※ A/B/C/D または 1/2/3/4 で回答できます。 text 問題は直接入力してください。")
    print("q を入力すると途中終了します。\n")

    # とりあえず item_id 昇順に全部出す（ランダムにしたければ sample してもよい）
    for _, row in items.sort_values("item_id").iterrows():
        q = str(row["question_text"])
        answer_type = str(row["answer_type"])

        print("-" * 60)
        print(f"Q{int(row['item_id'])}: {q}")

        if answer_type == "mcq":
            choices = [c.strip() for c in str(row["choices"]).split(",") if c.strip()]
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            for i, choice in enumerate(choices):
                label = letters[i]
                print(f"  {label}. {choice}")

            while True:
                ans = input("あなたの答え (A/B/C... または 1/2/3..., qで終了) > ").strip()
                if ans.lower() == "q":
                    print("セッションを終了します。")
                    return

                result = judge_mcq(row, ans)
                if result is None:
                    print("入力がよくわかりませんでした。A/B/C... または 1/2/3... で回答してください。")
                    continue

                correct = bool(result)
                break

            correct_idx = int(row["correct_key"])
            correct_label = letters[correct_idx]
            correct_text = choices[correct_idx]

            if correct:
                print("→ 正解！")
            else:
                print(f"→ 不正解… 正解は {correct_label}. {correct_text} です。")

        elif answer_type == "text":
            ans = input("あなたの答え（テキスト, qで終了） > ").strip()
            if ans.lower() == "q":
                print("セッションを終了します。")
                return

            correct = judge_text(row, ans)
            if correct:
                print("→ 正解！")
            else:
                print(f"→ 不正解… 正しい答えの一例: {row['correct_key']}")

        else:
            print(f"未対応の answer_type: {answer_type} （スキップします）")
            continue

        # ログに保存
        log_answer(log_path, user_id, row, correct)

    print("\n=== 全問題を出題し終えました。おつかれさまでした！ ===")


if __name__ == "__main__":
    main()
