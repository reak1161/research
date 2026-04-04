#!/usr/bin/env python
"""Visualize BKT learning trajectories for a user.

機能:
    - BKT パラメータとログから指定ユーザー×スキルの P(L), P(next) を逐次計算
    - 各ステップの数値を CSV に保存し、確率推移のグラフ(PNG)を出力
    - スキル指定や列名差し替えなど柔軟な可視化オプションを提供

使い方例:
    python -m code.plot_user_history \
        --log-csv csv/sim_logs.csv \
        --params-csv csv/bkt_params_multi.csv \
        --user-id user000 \
        --skills 基本的なファイル管理 \
        --skill-col L2 \
        --order-col timestamp \
        --correct-col correct \
        --out-dir runs/history_plots/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

# 日本語フォント設定（Noto Sans CJK JP があれば採用）
def _configure_fonts() -> None:
    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Noto Serif CJK JP",
        "Noto Serif JP",
    ]
    candidate_files = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"),
    ]
    picked = None
    for fpath in candidate_files:
        if fpath.exists():
            font_manager.fontManager.addfont(str(fpath))
            props = font_manager.FontProperties(fname=str(fpath))
            picked = props.get_name()
            break
    if picked is None:
        available = {f.name.lower(): f.fname for f in font_manager.fontManager.ttflist}
        for name in candidates:
            lower = name.lower()
            if lower in available:
                font_manager.fontManager.addfont(available[lower])
                picked = font_manager.FontProperties(fname=available[lower]).get_name()
                break
    if picked:
        plt.rcParams["font.family"] = [picked, "DejaVu Sans"]
    else:
        print(
            "[warn] Noto 系フォントが見つかりません。デフォルトフォントを使用します。",
            file=sys.stderr,
        )
    plt.rcParams["axes.unicode_minus"] = False


_configure_fonts()

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from code import bkt_core  # type: ignore
    from code.data_io import ensure_directory, read_long_format_csv  # type: ignore
else:
    from . import bkt_core
    from .data_io import ensure_directory, read_long_format_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot BKT learning curves per user & skill.")
    ap.add_argument("--log-csv", required=True)
    ap.add_argument("--params-csv", required=True)
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--skills", nargs="*", help="Specific skills to plot. Defaults to all.")
    ap.add_argument("--include-overall", action="store_true", help="Also output overall (cross-skill) trajectory.")
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
    ax.plot(seq_df["step"], seq_df["p_L_prior"], label="P(L) prior", color="#1f77b4", linewidth=2)
    ax.plot(
        seq_df["step"],
        seq_df["p_correct_now"],
        label="P(correct)",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
    )
    correct_mask = seq_df["observation"] == 1
    incorrect_mask = ~correct_mask
    ax.scatter(
        seq_df.loc[correct_mask, "step"],
        seq_df.loc[correct_mask, "observation"],
        label="correct",
        color="#1f77b4",
        marker="o",
        s=30,
    )
    ax.scatter(
        seq_df.loc[incorrect_mask, "step"],
        seq_df.loc[incorrect_mask, "observation"],
        label="incorrect",
        color="#ff7f0e",
        marker="x",
        s=30,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_title(f"user={user}  skill={skill}")
    ax.legend(loc="lower right")
    png_path = out_dir / f"user_{user}_skill_{skill}.png"
    pdf_path = out_dir / f"user_{user}_skill_{skill}.pdf"
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_overall_sequence(
    df_user: pd.DataFrame,
    cols: dict[str, str],
    params: pd.DataFrame,
) -> pd.DataFrame:
    param_map = {
        str(row["skill_name"]): bkt_core.BKTParams.from_row(row)
        for _, row in params.iterrows()
    }
    states = {skill: prm.L0 for skill, prm in param_map.items()}
    rows: list[dict] = []
    step = 0
    for _, row in df_user.iterrows():
        skill = str(row[cols["skill_name"]])
        if skill not in param_map:
            continue
        prm = param_map[skill]
        p_l_domain_prior = states.get(skill, prm.L0)
        p_correct_now = p_l_domain_prior * (1.0 - prm.S) + (1.0 - p_l_domain_prior) * prm.G
        p_l_overall_prior = float(sum(states.values()) / len(states)) if states else prm.L0
        obs = int(row[cols["correct"]])
        step += 1
        rows.append(
            {
                "step": step,
                "skill_name": skill,
                "observation": obs,
                "p_L_overall_prior": p_l_overall_prior,
                "p_L_domain_prior": p_l_domain_prior,
                "p_correct_now": p_correct_now,
            }
        )
        state_info = bkt_core.update_state(p_l_domain_prior, prm, obs)
        states[skill] = state_info["p_L_after"]
    return pd.DataFrame(rows)


def plot_overall_history(user: str, seq_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_directory(out_dir / "dummy.txt")
    fig, ax = plt.subplots(figsize=(8, 4))
    seq_df = seq_df.copy()
    seq_df["acc_cum"] = seq_df["observation"].expanding().mean()
    ax.plot(seq_df["step"], seq_df["p_correct_now"], label="P(correct, asked skill)", color="#ff7f0e", linewidth=2)
    ax.plot(seq_df["step"], seq_df["acc_cum"], label="Cumulative accuracy", color="#2ca02c", linewidth=2)
    ax.plot(seq_df["step"], seq_df["p_L_overall_prior"], label="Mean P(L) across skills", color="#1f77b4", linestyle="--", linewidth=2)
    correct_mask = seq_df["observation"] == 1
    incorrect_mask = ~correct_mask
    ax.scatter(seq_df.loc[correct_mask, "step"], seq_df.loc[correct_mask, "observation"], label="correct", color="#1f77b4", marker="o", s=30)
    ax.scatter(seq_df.loc[incorrect_mask, "step"], seq_df.loc[incorrect_mask, "observation"], label="incorrect", color="#ff7f0e", marker="x", s=30)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_title(f"user={user}  overall")
    ax.legend(loc="lower right")
    png_path = out_dir / f"user_{user}_overall.png"
    pdf_path = out_dir / f"user_{user}_overall.pdf"
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
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
    requested_skills = list(args.skills or [])
    request_overall = args.include_overall or any(str(s).lower() == "overall" for s in requested_skills)
    requested_skills = [s for s in requested_skills if str(s).lower() != "overall"]

    target_skills: List[str]
    if requested_skills:
        target_skills = requested_skills
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

    if request_overall:
        seq_overall = build_overall_sequence(df_user, cols, params)
        if not seq_overall.empty:
            csv_path = out_dir / f"user_{args.user_id}_overall.csv"
            seq_overall.to_csv(csv_path, index=False)
            plot_overall_history(str(args.user_id), seq_overall, out_dir)


if __name__ == "__main__":
    main()
