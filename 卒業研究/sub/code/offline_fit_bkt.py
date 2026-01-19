#!/usr/bin/env python
"""Offline BKT parameter fitting CLI.

機能:
    - 縦持ちログ CSV を読み込み、スキルごとに pyBKT モデルを学習
    - 推定した L0/T/S/G（必要に応じて forgets）を CSV として保存
    - オプションでスキル別 AUC・Logloss を算出し評価レポートを出力

使い方例:
    python -m code.offline_fit_bkt \
        --csv csv/sim_logs.csv \
        --skill-col L2 L2_tier \
        --order-col order_id \
        --correct-col correct \
        --out-params csv/bkt_params_multi.csv \
        --out-report runs/bkt_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from code import bkt_core  # type: ignore
    from code.data_io import ensure_directory, read_long_format_csv  # type: ignore
else:
    from . import bkt_core
    from .data_io import ensure_directory, read_long_format_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fit BKT parameters per skill.")
    ap.add_argument("--csv", required=True, help="Input long-format log CSV")
    ap.add_argument("--order-col", default="timestamp")
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--skill-col", nargs="+", default=["L2", "L2_tier"], help="スキル列を複数指定可能（例: --skill-col L2 L2_tier）")
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

    param_frames = []
    metric_frames = []
    for skill_col in args.skill_col:
        df, cols = read_long_format_csv(
            csv_path,
            order_col=args.order_col,
            user_col=args.user_col,
            skill_col=skill_col,
            correct_col=args.correct_col,
        )
        predictions, param_df = bkt_core.fit_bkt(
            df,
            cols,
            forgets=args.forgets,
            seed=args.seed,
            num_fits=args.num_fits,
        )
        param_df.insert(0, "skill_field", skill_col)
        param_frames.append(param_df)

        if args.out_report:
            score_col = detect_score_column(predictions)
            metrics_df = predictions.copy()
            skill_name_col = cols["skill_name"]
            if skill_name_col not in metrics_df.columns:
                if "skill_name" in metrics_df.columns:
                    skill_name_col = "skill_name"
                elif "skill" in metrics_df.columns:
                    skill_name_col = "skill"
            if skill_name_col not in metrics_df.columns:
                raise KeyError(f"Missing skill column in predictions: {skill_name_col}")
            metrics = bkt_core.compute_skill_metrics(
                metrics_df,
                skill_col=skill_name_col,
                correct_col=cols["correct"],
                score_col=score_col,
            )
            metrics.insert(0, "skill_field", skill_col)
            metric_frames.append(metrics)

    if not param_frames:
        raise RuntimeError("No skill columns processed.")

    params_out = pd.concat(param_frames, ignore_index=True)
    ensure_directory(params_path)
    params_out.to_csv(params_path, index=False)
    print(f"[info] Wrote parameters -> {params_path}")

    if args.out_report and metric_frames:
        report_path = Path(args.out_report)
        ensure_directory(report_path)
        pd.concat(metric_frames, ignore_index=True).to_csv(report_path, index=False)
        print(f"[info] Wrote metrics -> {report_path}")


if __name__ == "__main__":
    main()
