#!/usr/bin/env python
"""Generate synthetic learning logs from an items CSV.

機能:
    - 問題マスタ（items CSV）を読み込み、分野ごとのユーザー能力をランダム生成
    - ロジスティック関数で正答確率を求め、擬似的な user×item シーケンスを作成
    - order_id/timestamp 付きの long 形式 CSV を出力し、BKT 学習テスト用データに利用可能

使い方例:
    python -m code.simulate_user_logs \
        --items-csv csv/items_sample_lpic_tier.csv \
        --items-domain-col L2 \
        --out-csv csv/sim_logs.csv \
        --n-users 50 \
        --interactions-per-user 40 \
        --seed 42
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from code.data_io import ensure_directory  # type: ignore
else:
    from .data_io import ensure_directory


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simulate user logs by randomly answering items.")
    ap.add_argument("--items-csv", required=True, help="問題マスタ CSV")
    ap.add_argument("--items-item-col", default="item_id")
    ap.add_argument("--items-domain-col", default="domain", help="BKT/IRT で使う分野列（例: L2）")
    ap.add_argument("--out-csv", required=True, help="生成するログ CSV")
    ap.add_argument("--n-users", type=int, default=50)
    ap.add_argument("--interactions-per-user", type=int, default=40)
    ap.add_argument("--ability-std", type=float, default=1.0, help="ユーザー能力の標準偏差")
    ap.add_argument("--difficulty-std", type=float, default=1.0, help="問題難易度の標準偏差")
    ap.add_argument("--base-timestamp", default=None, help="ISO 形式。未指定なら現在時刻を使用")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    items = pd.read_csv(args.items_csv)
    if args.items_item_col not in items.columns or args.items_domain_col not in items.columns:
        raise ValueError("items CSV に必須列がありません。列名を --items-* で指定してください。")
    meta_keep = {col: items[col].copy() for col in ("L1", "L2", "L3", "tier") if col in items.columns}
    items = items.rename(
        columns={
            args.items_item_col: "item_id",
            args.items_domain_col: "domain",
        }
    )
    for col, series in meta_keep.items():
        items[col] = series
    if items.empty:
        raise ValueError("items CSV に行がありません。")

    # アイテム難易度をランダムに割り当て
    items["difficulty"] = rng.normal(loc=0.0, scale=args.difficulty_std, size=len(items))

    user_ids = [f"u{idx+1:03d}" for idx in range(args.n_users)]
    base_time = datetime.now(timezone.utc) if args.base_timestamp is None else datetime.fromisoformat(args.base_timestamp)

    rows = []
    order_counter = 0
    for user in user_ids:
        # 分野ごとにユーザー能力θをサンプリング
        domains = items["domain"].unique()
        abilities = {
            domain: rng.normal(loc=0.0, scale=args.ability_std)
            for domain in domains
        }
        for step in range(args.interactions_per_user):
            order_counter += 1
            # 現在の分野は学習度が低そうなものを少し優先
            domain = rng.choice(domains)
            pool = items[items["domain"] == domain]
            if pool.empty:
                continue
            item = pool.sample(n=1, random_state=int(rng.integers(0, 1 << 32))).iloc[0]
            theta = abilities[domain]
            diff = item["difficulty"]
            p_correct = float(sigmoid(theta - diff))
            correct = int(rng.random() < p_correct)
            jitter = int(rng.integers(0, 10))
            timestamp = base_time + timedelta(seconds=int(order_counter * 30 + jitter))

            row = {
                "order_id": order_counter,
                "timestamp": timestamp.isoformat(),
                "user_id": user,
                "domain": domain,
                "item_id": item["item_id"],
                "correct": correct,
                "p_correct_sim": p_correct,
            }
            # オリジナル列を可能な範囲で付与
            for col in ("L1", "L2", "L3", "tier", "question_text"):
                if col in item:
                    row[col] = item[col]
            if "L2" in item and "tier" in item:
                row["L2_tier"] = f"{item['L2']}:t{item['tier']}"
            rows.append(row)

    if not rows:
        raise RuntimeError("シミュレーション結果が空です。パラメータや items CSV を確認してください。")

    df_out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    ensure_directory(out_path)
    df_out.to_csv(out_path, index=False)
    print(f"[info] Simulated {df_out.shape[0]} interactions -> {out_path}")


if __name__ == "__main__":
    main()
