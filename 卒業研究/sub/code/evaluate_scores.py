#!/usr/bin/env python
"""Evaluate user×item score files via AUC/Logloss."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from code.data_io import ensure_directory  # type: ignore
else:
    from .data_io import ensure_directory


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate predictions in user_item_scores.csv.")
    ap.add_argument("--scores-csv", required=True)
    ap.add_argument("--by", choices=["overall", "domain", "user"], default="overall")
    ap.add_argument("--out", help="Optional CSV for metrics")
    ap.add_argument("--label-col", default="correct", help="Ground truth column (0/1)")
    ap.add_argument("--pred-col", default="P_final", help="Prediction probability column")
    return ap.parse_args()


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    clipped = np.clip(y_score, eps, 1.0 - eps)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        try:
            auc = float(roc_auc_score(y_true, clipped))
        except ValueError:
            auc = float("nan")
    try:
        ll = float(log_loss(y_true, clipped))
    except ValueError:
        ll = float("nan")
    return auc, ll


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.scores_csv)
    required = {args.label_col, args.pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scores CSV missing {missing}")

    df = df.dropna(subset=[args.label_col]).copy()
    if df.empty:
        raise ValueError("No rows with valid ground-truth labels were found in the scores CSV.")

    rows = []
    if args.by == "overall":
        auc, ll = compute_metrics(df[args.label_col].to_numpy(), df[args.pred_col].to_numpy())
        rows.append({"key": "overall", "auc": auc, "logloss": ll, "n": int(df.shape[0])})
    else:
        group_col = "domain" if args.by == "domain" else "user_id"
        if group_col not in df.columns:
            raise ValueError(f"Scores CSV missing column '{group_col}' for grouping.")
        for key, group in df.groupby(group_col):
            auc, ll = compute_metrics(group[args.label_col].to_numpy(), group[args.pred_col].to_numpy())
            rows.append({"key": key, "auc": auc, "logloss": ll, "n": int(group.shape[0])})

    metrics = pd.DataFrame(rows)
    if args.out:
        out_path = Path(args.out)
        ensure_directory(out_path)
        metrics.to_csv(out_path, index=False)
        print(f"[info] wrote metrics -> {out_path}")
    else:
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
