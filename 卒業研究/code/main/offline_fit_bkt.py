#!/usr/bin/env python
"""Offline BKT parameter fitting CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    ap = argparse.ArgumentParser(description="Fit BKT parameters per skill.")
    ap.add_argument("--csv", required=True, help="Input long-format log CSV")
    ap.add_argument("--order-col", default="timestamp")
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--skill-col", default="skill")
    ap.add_argument("--correct-col", default="correct")
    ap.add_argument("--out-params", required=True, help="Output CSV for parameters")
    ap.add_argument("--out-report", help="Optional evaluation CSV (per skill metrics)")
    ap.add_argument("--forgets", action="store_true", help="Enable forgets=True in pyBKT")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-fits", type=int, default=1)
    return ap.parse_args()


def detect_score_column(df: pd.DataFrame) -> str:
    for cand in ("correct_predictions", "predictions", "pred"):
        if cand in df.columns:
            return cand
    raise ValueError("Unable to locate prediction column in pyBKT output.")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    params_path = Path(args.out_params)

    df, cols = read_long_format_csv(
        csv_path,
        order_col=args.order_col,
        user_col=args.user_col,
        skill_col=args.skill_col,
        correct_col=args.correct_col,
    )
    predictions, param_df = bkt_core.fit_bkt(
        df,
        cols,
        forgets=args.forgets,
        seed=args.seed,
        num_fits=args.num_fits,
    )
    ensure_directory(params_path)
    param_df.to_csv(params_path, index=False)
    print(f"[info] Wrote parameters -> {params_path}")

    if args.out_report:
        score_col = detect_score_column(predictions)
        metrics = bkt_core.compute_skill_metrics(
            predictions.rename(columns={cols["skill_name"]: "skill"}),
            skill_col="skill",
            correct_col=cols["correct"],
            score_col=score_col,
        )
        report_path = Path(args.out_report)
        ensure_directory(report_path)
        metrics.to_csv(report_path, index=False)
        print(f"[info] Wrote metrics -> {report_path}")


if __name__ == "__main__":
    main()
