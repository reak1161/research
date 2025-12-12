#!/usr/bin/env python
"""Visualize BKT learning trajectories for a user."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT))
    from code.main import bkt_core  # type: ignore
    from code.main.data_io import ensure_directory, read_long_format_csv  # type: ignore
else:
    from . import bkt_core
    from .data_io import ensure_directory, read_long_format_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot BKT learning curves per user & skill.")
    ap.add_argument("--log-csv", required=True)
    ap.add_argument("--params-csv", required=True)
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--skills", nargs="*", help="Specific skills to plot. Defaults to all.")
    ap.add_argument("--skill-col", default="skill")
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--order-col", default="timestamp")
    ap.add_argument("--correct-col", default="correct")
    ap.add_argument("--out-dir", required=True, help="Directory for PNG and CSV outputs")
    return ap.parse_args()


def normalize_param_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "skill" in df.columns and "skill_name" not in df.columns:
        df = df.rename(columns={"skill": "skill_name"})
    expected = {"skill_name", "L0", "T", "S", "G"}
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Parameter CSV missing columns {missing}")
    return df


def plot_skill_history(user: str, skill: str, seq_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_directory(out_dir / "dummy.txt")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(seq_df["step"], seq_df["p_L_prior"], label="P(L) prior", color="#1f77b4")
    ax.plot(seq_df["step"], seq_df["p_next"], label="P(next)", color="#ff7f0e")
    ax.scatter(seq_df["step"], seq_df["observation"], label="Observed correct", color="#2ca02c", marker="o")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_title(f"user={user}  skill={skill}")
    ax.legend(loc="lower right")
    png_path = out_dir / f"user_{user}_skill_{skill}.png"
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, cols = read_long_format_csv(
        Path(args.log_csv),
        order_col=args.order_col,
        user_col=args.user_col,
        skill_col=args.skill_col,
        correct_col=args.correct_col,
    )
    params = normalize_param_frame(pd.read_csv(args.params_csv))
    df_user = df[df[cols["user_id"]] == str(args.user_id)]
    if df_user.empty:
        raise ValueError(f"user {args.user_id} not found in log.")
    target_skills: List[str]
    if args.skills:
        target_skills = args.skills
    else:
        target_skills = sorted(df_user[cols["skill_name"]].unique().tolist())
    for skill in target_skills:
        subset = df_user[df_user[cols["skill_name"]] == skill]
        if subset.empty:
            continue
        if skill not in params["skill_name"].values:
            continue
        param_row = params[params["skill_name"] == skill].iloc[0]
        seq = bkt_core.rollout_bkt_sequence(subset[cols["correct"]].tolist(), bkt_core.BKTParams.from_row(param_row))
        seq = seq.assign(
            order_value=subset[cols["order_id"]].to_numpy(),
            item_id=subset.get("item_id", pd.Series([""] * subset.shape[0])).to_numpy(),
        )
        csv_path = out_dir / f"user_{args.user_id}_skill_{skill}.csv"
        seq.to_csv(csv_path, index=False)
        plot_skill_history(str(args.user_id), skill, seq, out_dir)


if __name__ == "__main__":
    main()
