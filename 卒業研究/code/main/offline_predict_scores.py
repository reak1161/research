#!/usr/bin/env python
"""Offline BKT + IRT probability estimator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT))
    from code.main import bkt_core  # type: ignore
    from code.main.data_io import ensure_directory, read_long_format_csv  # type: ignore
    from code.main.irt_core import (  # type: ignore
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
    ap.add_argument("--params-csv", required=True, help="BKT parameter CSV")
    ap.add_argument("--irt-items-csv", required=True)
    ap.add_argument("--items-csv", required=True, help="Problem master CSV (metadata optional)")
    ap.add_argument("--items-item-col", default="item_id")
    ap.add_argument("--items-domain-col", default="domain")
    ap.add_argument("--irt-theta-csv", help="Optional user×domain theta CSV")
    ap.add_argument("--order-col", default="timestamp")
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--skill-col", default="domain")
    ap.add_argument("--correct-col", default="correct")
    ap.add_argument("--cut-timestamp", help="Only use observations <= this timestamp (ISO)")
    ap.add_argument("--mode", choices=["hybrid_mean", "bkt_to_theta"], default="hybrid_mean")
    ap.add_argument("--w-bkt", type=float, default=0.5, help="Weight for BKT score in hybrid mode")
    ap.add_argument("--out", required=True, help="Output CSV path")
    return ap.parse_args()


def normalize_params(path: Path) -> pd.DataFrame:
    params = pd.read_csv(path)
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
    params = normalize_params(Path(args.params_csv))
    user_skill = bkt_core.summarize_user_skill_states(
        df_logs,
        params,
        user_col=cols["user_id"],
        skill_col=cols["skill_name"],
        correct_col=cols["correct"],
    )
    if user_skill.empty:
        raise RuntimeError("No user×skill combinations available after filtering.")
    user_skill = user_skill.rename(columns={"skill_name": "domain"})

    theta_df = None
    if args.irt_theta_csv:
        theta_df = load_theta(args.irt_theta_csv)
    irt_items = load_irt_items(args.irt_items_csv, item_col="item_id", domain_col="domain")
    items_meta = pd.read_csv(args.items_csv).rename(
        columns={
            args.items_item_col: "item_id",
            args.items_domain_col: "domain",
        }
    )

    merged = user_skill.merge(irt_items, on="domain", how="inner")
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
    merged["P_bkt"] = merged["p_next"]
    if args.mode == "hybrid_mean":
        merged["P_final"] = args.w_bkt * merged["P_bkt"] + (1.0 - args.w_bkt) * merged["P_irt"].fillna(merged["P_bkt"])
    else:
        merged["P_final"] = merged["P_irt"]
    merged["P_final"] = merged["P_final"].fillna(merged["P_bkt"])

    out_cols = ["user_id", "item_id", "domain", "P_bkt", "P_irt", "P_final", "theta", "a", "b", "c"]
    extra_cols = [c for c in ("question_text", "difficulty", "tags") if c in merged.columns]
    df_out = merged[out_cols + extra_cols]
    out_path = Path(args.out)
    ensure_directory(out_path)
    df_out.to_csv(out_path, index=False)
    print(f"[info] Wrote user×item scores -> {out_path}")


if __name__ == "__main__":
    main()
