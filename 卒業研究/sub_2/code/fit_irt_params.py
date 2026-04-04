#!/usr/bin/env python
"""Lightweight IRT(3PL) parameter fitter using stochastic gradient descent.

機能:
    - user×item×正誤ログから 3PL パラメータ (a, b, c, theta) を推定
    - domain（例: L2）ごとに独立して推定し、CSV で出力
    - c（当て推量）は学習しつつ、クリッピングで安定化（デフォルト 0.05〜0.35）

使い方例:
    python -m code.fit_irt_params \
        --log-csv csv/sim_logs.csv \
        --user-col user_id \
        --item-col item_id \
        --domain-col L2 \
        --correct-col correct \
        --out-items csv/irt_items_estimated.csv \
        --out-theta csv/irt_theta_estimated.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_io import ensure_directory


@dataclass
class SGDConfig:
    lr_theta: float = 0.05
    lr_item: float = 0.01
    lr_a: float = 0.005
    n_epochs: int = 30
    l2_theta: float = 1e-3
    l2_item: float = 1e-3
    l2_a: float = 1e-4
    min_a: float = 0.1
    max_a: float = 3.0
    lr_c: float = 0.002
    l2_c: float = 0.0
    min_c: float = 0.05
    max_c: float = 0.35
    theta_clip: float = 4.0  # avoid exploding abilities
    b_clip: float = 4.0      # avoid exploding difficulties


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_domain_irt(df: pd.DataFrame, cfg: SGDConfig, init_c: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {q: i for i, q in enumerate(items)}

    theta = np.zeros(len(users))
    a = np.ones(len(items))
    b = np.zeros(len(items))
    c = np.full(len(items), init_c)

    responses = df[["user_id", "item_id", "correct"]].to_numpy()

    for epoch in range(cfg.n_epochs):
        np.random.shuffle(responses)
        for user, item, correct in responses:
            idx_u = user_to_idx[user]
            idx_i = item_to_idx[item]
            x = a[idx_i] * (theta[idx_u] - b[idx_i])
            s = sigmoid(x)
            p = c[idx_i] + (1.0 - c[idx_i]) * s
            err = correct - p
            # Gradients (approximate) for log-likelihood with respect to parameters
            dp_dtheta = (1.0 - c[idx_i]) * s * (1.0 - s) * a[idx_i]
            dp_db = -(1.0 - c[idx_i]) * s * (1.0 - s) * a[idx_i]
            dp_da = (1.0 - c[idx_i]) * (theta[idx_u] - b[idx_i]) * s * (1.0 - s)
            dp_dc = 1.0 - s

            theta[idx_u] += cfg.lr_theta * (err * dp_dtheta - cfg.l2_theta * theta[idx_u])
            b[idx_i] -= cfg.lr_item * (err * dp_db + cfg.l2_item * b[idx_i])
            a[idx_i] += cfg.lr_a * (err * dp_da - cfg.l2_a * (a[idx_i] - 1.0))
            c[idx_i] += cfg.lr_c * (err * dp_dc - cfg.l2_c * (c[idx_i] - init_c))

            if cfg.theta_clip > 0:
                theta[idx_u] = np.clip(theta[idx_u], -cfg.theta_clip, cfg.theta_clip)
            if cfg.b_clip > 0:
                b[idx_i] = np.clip(b[idx_i], -cfg.b_clip, cfg.b_clip)
            a[idx_i] = np.clip(a[idx_i], cfg.min_a, cfg.max_a)
            c[idx_i] = np.clip(c[idx_i], cfg.min_c, cfg.max_c)

    item_rows = []
    for item, idx in item_to_idx.items():
        item_rows.append({"item_id": item, "a": float(a[idx]), "b": float(b[idx]), "c": float(c[idx])})
    user_rows = []
    for user, idx in user_to_idx.items():
        user_rows.append({"user_id": user, "theta": float(theta[idx])})
    return pd.DataFrame(item_rows), pd.DataFrame(user_rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Estimate simple 2PL IRT parameters via SGD.")
    ap.add_argument("--log-csv", required=True)
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--item-col", default="item_id")
    ap.add_argument("--domain-col", default=None, help="ドメイン列（例: L2）。未指定なら全体で推定。")
    ap.add_argument("--correct-col", default="correct")
    ap.add_argument("--out-items", required=True, help="項目パラメータ出力 CSV")
    ap.add_argument("--out-theta", required=True, help="ユーザー能力出力 CSV")
    ap.add_argument("--init-c", type=float, default=0.2, help="当て推量 c の初期値（学習しつつクリップ）")
    ap.add_argument("--min-c", type=float, default=0.05)
    ap.add_argument("--max-c", type=float, default=0.35)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--theta-clip", type=float, default=4.0, help="θ のクリップ範囲（±theta-clip、0 で無効）")
    ap.add_argument("--b-clip", type=float, default=4.0, help="b のクリップ範囲（±b-clip、0 で無効）")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.log_csv)
    for col in (args.user_col, args.item_col, args.correct_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.log_csv}")

    df = df.rename(
        columns={
            args.user_col: "user_id",
            args.item_col: "item_id",
            args.correct_col: "correct",
        }
    )
    df["correct"] = df["correct"].astype(float)
    cfg = SGDConfig(
        n_epochs=args.epochs,
        min_c=args.min_c,
        max_c=args.max_c,
        theta_clip=args.theta_clip,
        b_clip=args.b_clip,
    )

    item_frames: List[pd.DataFrame] = []
    theta_frames: List[pd.DataFrame] = []

    if args.domain_col and args.domain_col in df.columns:
        for domain, group in df.groupby(args.domain_col):
            items, thetas = fit_domain_irt(group.copy(), cfg, init_c=args.init_c)
            items.insert(1, "domain", domain)
            thetas.insert(1, "domain", domain)
            item_frames.append(items)
            theta_frames.append(thetas)
    else:
        items, thetas = fit_domain_irt(df.copy(), cfg, init_c=args.init_c)
        items["domain"] = ""
        theta_frames.append(thetas.assign(domain=""))
        item_frames.append(items)

    ensure_directory(Path(args.out_items))
    pd.concat(item_frames, ignore_index=True).to_csv(args.out_items, index=False)
    ensure_directory(Path(args.out_theta))
    pd.concat(theta_frames, ignore_index=True).to_csv(args.out_theta, index=False)
    print(f"[info] wrote item params -> {args.out_items}")
    print(f"[info] wrote theta params -> {args.out_theta}")


if __name__ == "__main__":
    main()
