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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_io import ensure_directory


@dataclass
class SGDConfig:
    # 学習率は「ユーザー能力(theta)」「項目難易度(b)」「識別力(a)」「当て推量(c)」で分離。
    # パラメータごとにスケールが異なるため、同一 learning rate を使わない設計。
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


def _binary_logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_domain_irt(
    df: pd.DataFrame,
    cfg: SGDConfig,
    init_c: float,
    *,
    domain_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {q: i for i, q in enumerate(items)}

    theta = np.zeros(len(users))
    a = np.ones(len(items))
    b = np.zeros(len(items))
    c = np.full(len(items), init_c)
    # 初期値:
    # - theta=0, b=0 は「平均的受験者・平均的難易度」
    # - a=1 はロジスティックの標準スケール
    # - c は指定初期値から学習しつつクリップで安定化

    responses = df[["user_id", "item_id", "correct"]].to_numpy()
    u_idx_all = np.array([user_to_idx[str(u)] for u in responses[:, 0]], dtype=np.int64)
    i_idx_all = np.array([item_to_idx[str(i)] for i in responses[:, 1]], dtype=np.int64)
    y_all = responses[:, 2].astype(float)
    order = np.arange(len(y_all))
    history_rows: list[dict] = []

    for epoch in range(cfg.n_epochs):
        # SGD: 各 epoch で観測順をシャッフルして局所的な順序バイアスを軽減。
        np.random.shuffle(order)
        for ridx in order:
            idx_u = u_idx_all[ridx]
            idx_i = i_idx_all[ridx]
            correct = y_all[ridx]
            x = a[idx_i] * (theta[idx_u] - b[idx_i])
            s = sigmoid(x)
            p = c[idx_i] + (1.0 - c[idx_i]) * s
            err = correct - p
            # 3PL の確率 p に対する各パラメータの偏導関数（近似）を使って更新。
            # 更新方向は「観測(correct) と予測(p) の差 err」を縮める方向。
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

        x_all = a[i_idx_all] * (theta[u_idx_all] - b[i_idx_all])
        s_all = sigmoid(x_all)
        p_all = c[i_idx_all] + (1.0 - c[i_idx_all]) * s_all
        # 収束の可視化用に、epoch ごとの損失と平均パラメータを保存。
        history_rows.append(
            {
                "domain": domain_name,
                "epoch": int(epoch + 1),
                "logloss": _binary_logloss(y_all, p_all),
                "mse": float(np.mean(np.square(p_all - y_all))),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "mean_c": float(np.mean(c)),
                "mean_theta": float(np.mean(theta)),
                "std_theta": float(np.std(theta)),
            }
        )

    item_rows = []
    for item, idx in item_to_idx.items():
        item_rows.append({"item_id": item, "a": float(a[idx]), "b": float(b[idx]), "c": float(c[idx])})
    user_rows = []
    for user, idx in user_to_idx.items():
        user_rows.append({"user_id": user, "theta": float(theta[idx])})
    return pd.DataFrame(item_rows), pd.DataFrame(user_rows), pd.DataFrame(history_rows)


def plot_irt_history(history: pd.DataFrame, out_dir: Path) -> None:
    ensure_directory(out_dir / "dummy.txt")
    for domain, grp in history.groupby("domain"):
        grp = grp.sort_values("epoch")
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax0, ax1 = axes

        ax0.plot(grp["epoch"], grp["logloss"], label="logloss", color="#d62728", linewidth=2)
        ax0.plot(grp["epoch"], grp["mse"], label="mse", color="#9467bd", linewidth=2)
        # 上段: 損失推移（収束確認）
        ax0.set_ylabel("Loss")
        ax0.set_title(f"IRT fit history / domain={domain}")
        ax0.legend(loc="best")

        ax1.plot(grp["epoch"], grp["mean_a"], label="mean_a", color="#1f77b4", linewidth=2)
        ax1.plot(grp["epoch"], grp["mean_b"], label="mean_b", color="#ff7f0e", linewidth=2)
        ax1.plot(grp["epoch"], grp["mean_c"], label="mean_c", color="#2ca02c", linewidth=2)
        ax1.plot(grp["epoch"], grp["mean_theta"], label="mean_theta", color="#8c564b", linestyle="--", linewidth=2)
        # 下段: パラメータ平均の推移（暴走/過収縮の検知）
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Param mean")
        ax1.legend(loc="best")

        fig.tight_layout()
        base = out_dir / f"irt_fit_history_domain_{domain}"
        fig.savefig(base.with_suffix(".png"))
        fig.savefig(base.with_suffix(".pdf"))
        plt.close(fig)


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
    ap.add_argument("--out-history", help="Optional epoch-wise training history CSV")
    ap.add_argument("--plot-history-dir", help="Optional output directory for history plots (PNG/PDF)")
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
    hist_frames: List[pd.DataFrame] = []

    if args.domain_col and args.domain_col in df.columns:
        for domain, group in df.groupby(args.domain_col):
            items, thetas, hist = fit_domain_irt(group.copy(), cfg, init_c=args.init_c, domain_name=str(domain))
            items.insert(1, "domain", domain)
            thetas.insert(1, "domain", domain)
            item_frames.append(items)
            theta_frames.append(thetas)
            hist_frames.append(hist)
    else:
        items, thetas, hist = fit_domain_irt(df.copy(), cfg, init_c=args.init_c, domain_name="")
        items["domain"] = ""
        theta_frames.append(thetas.assign(domain=""))
        item_frames.append(items)
        hist_frames.append(hist)

    ensure_directory(Path(args.out_items))
    pd.concat(item_frames, ignore_index=True).to_csv(args.out_items, index=False)
    ensure_directory(Path(args.out_theta))
    pd.concat(theta_frames, ignore_index=True).to_csv(args.out_theta, index=False)
    print(f"[info] wrote item params -> {args.out_items}")
    print(f"[info] wrote theta params -> {args.out_theta}")

    if hist_frames:
        history = pd.concat(hist_frames, ignore_index=True)
        if args.out_history:
            out_hist = Path(args.out_history)
            ensure_directory(out_hist)
            history.to_csv(out_hist, index=False)
            print(f"[info] wrote fit history -> {out_hist}")
        if args.plot_history_dir:
            out_plot = Path(args.plot_history_dir)
            plot_irt_history(history, out_plot)
            print(f"[info] wrote fit history plots -> {out_plot}")


if __name__ == "__main__":
    main()
