#!/usr/bin/env python
"""Offline BKT + IRT probability estimator.

機能:
    - BKT のユーザー×分野状態を集計し、IRT 項目パラメータと結合
    - θ があれば利用、無い場合は BKT から擬似 θ を生成
    - P_bkt と P_irt を重み付きで合成し、ユーザー×問題の P_final を CSV 出力

使い方例:
    python -m code.offline_predict_scores \
        --log-csv csv/sim_logs.csv \
        --params-csv csv/bkt_params_multi.csv \
        --params-main-field L2 \
        --params-tier-field L2_tier \
        --irt-items-csv csv/irt_items_estimated.csv \
        --items-csv csv/items_sample_lpic_tier.csv \
        --items-domain-col L2 \
        --skill-col L2 \
        --tier-skill-col L2_tier \
        --mode hybrid_mean \
        --w-bkt 0.6 \
        --w-tier 0.5 \
        --out runs/sim_user_item_scores.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from code import bkt_core  # type: ignore
    from code.data_io import ensure_directory, read_long_format_csv  # type: ignore
    from code.irt_core import (  # type: ignore
        bkt_prob_to_theta,
        load_irt_items,
        load_theta,
        three_pl,
    )
else:
    from . import bkt_core
    from .data_io import ensure_directory, read_long_format_csv
    from .irt_core import bkt_prob_to_theta, load_irt_items, load_theta, three_pl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Combine BKT mastery with IRT item params.")
    ap.add_argument("--log-csv", required=True)
    ap.add_argument("--params-csv", required=True, help="BKT parameter CSV（L2/L2_tier の両方を含む）")
    ap.add_argument("--params-main-field", default="L2", help="params-csv の L2 フィルタ値（skill_field）")
    ap.add_argument("--params-tier-csv", help="L2_tier のBKTパラメータ CSV（未指定時は --params-csv を使用）")
    ap.add_argument("--params-tier-field", default="L2_tier", help="params-tier の skill_field フィルタ値")
    ap.add_argument("--irt-items-csv", required=True)
    ap.add_argument("--items-csv", required=True, help="Problem master CSV (metadata optional)")
    ap.add_argument("--items-item-col", default="item_id")
    ap.add_argument("--items-domain-col", default="domain")
    ap.add_argument("--irt-theta-csv", help="Optional user×domain theta CSV")
    ap.add_argument("--order-col", default="timestamp")
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--skill-col", default="domain")
    ap.add_argument("--tier-skill-col", default="L2_tier", help="L2×tier のスキル列")
    ap.add_argument("--correct-col", default="correct")
    ap.add_argument("--cut-timestamp", help="Only use observations <= this timestamp (ISO)")
    ap.add_argument("--mode", choices=["hybrid_mean", "bkt_to_theta"], default="hybrid_mean")
    ap.add_argument("--w-bkt", type=float, default=0.5, help="Weight for BKT score in hybrid mode")
    ap.add_argument("--w-tier", type=float, default=0.5, help="Weight for L2_tier in BKT blend")
    ap.add_argument("--out", required=True, help="Output CSV path")
    return ap.parse_args()


def normalize_params(path: Path, skill_field_filter: str | None = None) -> pd.DataFrame:
    params = pd.read_csv(path)
    if skill_field_filter and "skill_field" in params.columns:
        params = params[params["skill_field"].astype(str) == str(skill_field_filter)]
        if params.empty:
            raise ValueError(f"{path} has no rows with skill_field={skill_field_filter}")
    if "skill" in params.columns and "skill_name" not in params.columns:
        params = params.rename(columns={"skill": "skill_name"})
    missing = [c for c in ("skill_name", "L0", "T", "S", "G") if c not in params.columns]
    if missing:
        raise ValueError(f"Parameter CSV missing {missing}")
    return params


def maybe_cut(df: pd.DataFrame, column: str, cutoff: str) -> pd.DataFrame:
    if cutoff is None:
        return df
    ts = pd.to_datetime(df[column], errors="coerce")
    mask = ts <= pd.to_datetime(cutoff)
    return df[mask].copy()




def main() -> None:
    args = parse_args()
    df_logs, cols = read_long_format_csv(
        Path(args.log_csv),
        order_col=args.order_col,
        user_col=args.user_col,
        skill_col=args.skill_col,
        correct_col=args.correct_col,
    )
    df_logs = maybe_cut(df_logs, args.order_col, args.cut_timestamp)
    params_main = normalize_params(Path(args.params_csv), args.params_main_field)

    if args.tier_skill_col not in df_logs.columns and "L2" in df_logs.columns and "tier" in df_logs.columns:
        df_logs[args.tier_skill_col] = df_logs["L2"].astype(str) + ":t" + df_logs["tier"].astype(str)

    def summarize_level(skill_col: str, params_df: pd.DataFrame, label: str) -> pd.DataFrame:
        s = bkt_core.summarize_user_skill_states(
            df_logs,
            params_df,
            user_col=cols["user_id"],
            skill_col=skill_col,
            correct_col=cols["correct"],
        )
        if s.empty:
            return s
        return s.rename(
            columns={
                "skill_name": label,
                "p_next": f"P_bkt_{label}",
                "n_obs": f"N_{label}",
            }
        )

    # メイン階層（L2 相当）
    user_skill_main = summarize_level(cols["skill_name"], params_main, "L2")
    if user_skill_main.empty:
        raise RuntimeError("No user×skill combinations available after filtering.")
    user_skill_main = user_skill_main.rename(columns={"L2": "domain"})

    # L2_tier のパラメータ
    params_tier = None
    if args.params_tier_csv:
        params_tier = normalize_params(Path(args.params_tier_csv), args.params_tier_field)
    else:
        try:
            params_tier = normalize_params(Path(args.params_csv), args.params_tier_field)
        except Exception:
            params_tier = None

    user_skill_tier = (
        summarize_level(args.tier_skill_col, params_tier, "L2_tier")
        if params_tier is not None and args.tier_skill_col in df_logs.columns
        else pd.DataFrame()
    )

    theta_df = None
    if args.irt_theta_csv:
        theta_df = load_theta(args.irt_theta_csv)
    irt_items = load_irt_items(args.irt_items_csv, item_col="item_id", domain_col="domain")
    items_meta_raw = pd.read_csv(args.items_csv)
    items_meta = items_meta_raw.rename(
        columns={
            args.items_item_col: "item_id",
            args.items_domain_col: "domain",
        }
    )
    # 元の L1/L2/L3/tier 列を保持（domain にリネームされた分もコピーしておく）
    for level_col in ("L1", "L2", "L3", "tier"):
        if level_col in items_meta_raw.columns and level_col not in items_meta.columns:
            items_meta[level_col] = items_meta_raw[level_col]
    if args.items_domain_col in items_meta_raw.columns:
        items_meta["L2"] = items_meta_raw[args.items_domain_col]
    if "L2" in items_meta.columns and "tier" in items_meta.columns:
        items_meta["L2_tier"] = items_meta["L2"].astype(str) + ":t" + items_meta["tier"].astype(str)

    merged = user_skill_main.merge(irt_items, on="domain", how="inner")
    merged = merged.merge(items_meta, on=["domain", "item_id"], how="left", suffixes=("", "_meta"))
    if theta_df is not None:
        merged = merged.merge(theta_df, on=["user_id", "domain"], how="left")
    if "theta" not in merged.columns:
        merged["theta"] = merged["p_L_after"].apply(bkt_prob_to_theta)
    else:
        missing_theta = merged["theta"].isna()
        merged.loc[missing_theta, "theta"] = merged.loc[missing_theta, "p_L_after"].apply(bkt_prob_to_theta)

    mask = merged["theta"].notna()
    merged["P_irt"] = np.nan
    merged.loc[mask, "P_irt"] = three_pl(
        theta=merged.loc[mask, "theta"].to_numpy(),
        a=merged.loc[mask, "a"].to_numpy(),
        b=merged.loc[mask, "b"].to_numpy(),
        c=merged.loc[mask, "c"].to_numpy(),
    )
    if not user_skill_tier.empty and "L2_tier" in merged.columns:
        merged = merged.merge(
            user_skill_tier[["user_id", "L2_tier", "P_bkt_L2_tier", "N_L2_tier"]],
            on=["user_id", "L2_tier"],
            how="left",
        )
        merged["P_bkt"] = (
            (1.0 - args.w_tier) * merged["P_bkt_L2"].fillna(0.5)
            + args.w_tier * merged["P_bkt_L2_tier"].fillna(0.5)
        )
    else:
        merged["P_bkt"] = merged["P_bkt_L2"]

    if args.mode == "hybrid_mean":
        merged["P_final"] = args.w_bkt * merged["P_bkt"] + (1.0 - args.w_bkt) * merged["P_irt"].fillna(merged["P_bkt"])
    else:
        merged["P_final"] = merged["P_irt"]
    merged["P_final"] = merged["P_final"].fillna(merged["P_bkt"])

    correctness = (
        df_logs[[cols["user_id"], "item_id", cols["correct"]]]
        .rename(columns={cols["user_id"]: "user_id", cols["correct"]: "correct"})
        .drop_duplicates(subset=["user_id", "item_id"])
    )
    merged = merged.merge(correctness, on=["user_id", "item_id"], how="left", suffixes=("", "_log"))

    for col in ("L2_tier", "P_bkt_L2", "P_bkt_L2_tier"):
        if col not in merged.columns:
            merged[col] = np.nan

    out_cols = [
        "user_id",
        "item_id",
        "domain",
        "L2_tier",
        "correct",
        "P_bkt_L2",
        "P_bkt_L2_tier",
        "P_bkt",
        "P_irt",
        "P_final",
        "theta",
        "a",
        "b",
        "c",
    ]
    extra_cols = [c for c in ("question_text", "difficulty", "tags") if c in merged.columns]
    df_out = merged[out_cols + extra_cols]
    out_path = Path(args.out)
    ensure_directory(out_path)
    if out_path.exists():
        out_path.unlink()
    df_out.to_csv(out_path, index=False)
    print(f"[info] Wrote user×item scores -> {out_path}")


if __name__ == "__main__":
    main()
