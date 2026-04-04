"""Utility helpers for loading CSV files and preparing long-format logs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def ensure_directory(path: Path) -> None:
    """Create parent directories for the given path if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_long_format_csv(
    csv_path: Path,
    *,
    order_col: str = "timestamp",
    user_col: str = "user_id",
    skill_col: str = "skill",
    correct_col: str = "correct",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load a long-format response log and return normalized columns."""
    df = pd.read_csv(csv_path)
    missing = [c for c in (order_col, user_col, skill_col, correct_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path}")

    df = df.copy()
    df[skill_col] = df[skill_col].astype(str)
    df[user_col] = df[user_col].astype(str)

    # Normalize order column (support datetime strings)
    order_series = df[order_col]
    if not pd.api.types.is_integer_dtype(order_series):
        if not pd.api.types.is_datetime64_any_dtype(order_series):
            df["_order_tmp"] = pd.to_datetime(order_series, errors="coerce")
            sort_keys = [user_col, "_order_tmp", order_col]
        else:
            sort_keys = [user_col, order_col]
        sorted_df = df.sort_values(sort_keys)
        df = sorted_df.assign(_order_int=sorted_df.groupby(user_col).cumcount())
        df = df.drop(columns=[c for c in ["_order_tmp"] if c in df.columns])
        norm_order_col = "_order_int"
    else:
        df = df.sort_values([user_col, order_col])
        norm_order_col = order_col

        if "order_id" in df.columns:
            df = df.drop(columns=["order_id"])
        df = df.reset_index(drop=True)
    cols = {
        "order_id": norm_order_col,
        "user_id": user_col,
        "skill_name": skill_col,
        "correct": correct_col,
    }
    return df, cols
