#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ターミナル上でユーザーに問題を出題し、その場で BKT を更新する簡易チューター。

想定ディレクトリ構成（スクショのやつ）:

卒業研究/
  code/
    python_3_10/
      interactive_quiz.py  ← このファイル
      pybkt_user_domain.py など
  csv/
    items_demo.csv          ← 問題バンク
    bkt_params_domain.csv   ← domainごとのBKTパラメータ (L0,T,S,G)
    ct_online.csv           ← このスクリプトが書き出すオンライン回答ログ

使い方 (code/python_3_10 ディレクトリで):

  (.venv) $ python interactive_quiz.py --user-id 68

"""

import argparse
import random
import time
from pathlib import Path
import pandas as pd

# ===== パスの設定 =====
# このファイル (interactive_quiz.py) から見て:
#   .../python_3_10/interactive_quiz.py
#   → parents[0] = python_3_10
#   → parents[1] = code
#   → parents[2] = 卒業研究 (プロジェクトルート想定)
ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "csv"

DEFAULT_ITEMS_CSV = CSV_DIR / "items_demo.csv"
DEFAULT_BKT_PARAMS_CSV = CSV_DIR / "bkt_params_domain.csv"
LOG_PATH = CSV_DIR / "ct_online.csv"


def load_items(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "item_id", "domain", "question_text",
        "choice1", "choice2", "choice3", "choice4",
        "correct_choice",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"items_csv の列が足りません: {missing}")
    return df


def load_bkt_params(path: Path):
    """
    pybkt_user_domain.py の --eval-out で吐いた CSV を読み込む想定。

    想定列例:
      skill_name (ここでは domain)
      p_L0, p_T, p_S, p_G, ... など

    ここでは:
      domain, L0, T, S, G
    の形に整形して使う。
    """
    df = pd.read_csv(path)

    # 列名のゆらぎ吸収
    rename_map = {
        "skill": "domain",
        "skill_name": "domain",
        "KC": "domain",
        "kcid": "domain",
        "p_L0": "L0",
        "prior": "L0",
        "L0": "L0",
        "p_T": "T",
        "learn": "T",
        "p_S": "S",
        "slip": "S",
        "p_G": "G",
        "guess": "G",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    required = {"domain", "L0", "T", "S", "G"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"bkt_params_csv の列が足りません: {missing}")

    params = {}
    for _, row in df.iterrows():
        domain = str(row["domain"])
        params[domain] = {
            "L0": float(row["L0"]),
            "T": float(row["T"]),
            "S": float(row["S"]),
            "G": float(row["G"]),
        }
    return params


def init_bkt_state(bkt_params: dict):
    """各 domain の P(L) を L0 で初期化"""
    state = {}
    for domain, p in bkt_params.items():
        state[domain] = {"P_L": p["L0"]}
    return state


def bkt_update(p_L: float, S: float, G: float, T: float, correct: bool) -> float:
    """1ステップ分の BKT 更新（観測→学習）"""
    if correct:
        num = p_L * (1.0 - S)
        denom = num + (1.0 - p_L) * G
    else:
        num = p_L * S
        denom = num + (1.0 - p_L) * (1.0 - G)

    if denom <= 0:
        p_posterior = p_L
    else:
        p_posterior = num / denom

    p_next = p_posterior + (1.0 - p_posterior) * T
    # 0,1 に張り付きすぎないように
    p_next = max(1e-6, min(1.0 - 1e-6, p_next))
    return p_next


def choose_domain(bkt_state: dict) -> str:
    """1 - P(L) を重みとして分野をサンプリング（弱い分野を出題しやすく）"""
    domains = list(bkt_state.keys())
    weights = [1.0 - bkt_state[d]["P_L"] for d in domains]
    total = sum(weights)
    if total <= 0:
        return random.choice(domains)
    r = random.random() * total
    acc = 0.0
    for d, w in zip(domains, weights):
        acc += w
        if r <= acc:
            return d
    return domains[-1]


def choose_item(items: pd.DataFrame, domain: str) -> pd.Series:
    candidates = items[items["domain"] == domain]
    if candidates.empty:
        raise ValueError(f"domain='{domain}' の問題が items_csv にありません")
    return candidates.sample(1).iloc[0]


def log_answer(user_id: int, item_row: pd.Series, correct: bool):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    row = {
        "timestamp": ts,
        "user_id": user_id,
        "item_id": item_row["item_id"],
        "domain": item_row["domain"],
        "correct": int(correct),
    }
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, mode="w", header=True, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--items-csv", type=str, default=str(DEFAULT_ITEMS_CSV))
    parser.add_argument("--bkt-params-csv", type=str, default=str(DEFAULT_BKT_PARAMS_CSV))
    args = parser.parse_args()

    items = load_items(Path(args.items_csv))
    bkt_params = load_bkt_params(Path(args.bkt_params_csv))
    bkt_state = init_bkt_state(bkt_params)

    user_id = args.user_id
    print(f"=== ユーザー {user_id} の対話セッションを開始します ===")
    print("q を入力すると終了します。\n")

    while True:
        domain = choose_domain(bkt_state)
        p = bkt_params[domain]
        state = bkt_state[domain]

        item = choose_item(items, domain)

        # 出題前の推定正答率 (P(correct_now))
        p_L = state["P_L"]
        p_now = p_L * (1.0 - p["S"]) + (1.0 - p_L) * p["G"]

        print(f"\n--- 分野: {domain} ---")
        print(f"現在の推定正答率 (P(correct_now)): {p_now:.2f}")
        print(f"Q{item['item_id']}: {item['question_text']}")
        print(f"  1) {item['choice1']}")
        print(f"  2) {item['choice2']}")
        print(f"  3) {item['choice3']}")
        print(f"  4) {item['choice4']}")
        ans = input("あなたの答え (1-4, qで終了) > ").strip()

        if ans.lower() == "q":
            print("セッションを終了します。おつかれさまでした。")
            break

        if ans not in {"1", "2", "3", "4"}:
            print("1〜4 か q を入力してください（今回はスキップ）。")
            continue

        ans_int = int(ans)
        correct = (ans_int == int(item["correct_choice"]))
        print("結果:", "正解！" if correct else f"不正解… 正解は {item['correct_choice']} でした。")

        # ログ保存
        log_answer(user_id, item, correct)

        # BKT 更新
        new_p_L = bkt_update(p_L, p["S"], p["G"], p["T"], correct)
        bkt_state[domain]["P_L"] = new_p_L
        new_p_next = new_p_L * (1.0 - p["S"]) + (1.0 - new_p_L) * p["G"]

        print(f"この分野の更新後 P(L): {new_p_L:.3f}, 次の推定正答率 P(next): {new_p_next:.2f}")


if __name__ == "__main__":
    main()
