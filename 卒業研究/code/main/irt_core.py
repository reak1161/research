"""Helpers for working with IRT-style probability estimates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class IRTItem:
    item_id: str
    domain: str
    a: float
    b: float
    c: float


def three_pl(theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorized 3PL."""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def logit(p: float, eps: float = 1e-6) -> float:
    """Safe logit transform."""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def bkt_prob_to_theta(p: float) -> float:
    """Convert a BKT mastery probability into a pseudo-IRT theta."""
    return logit(p)


def load_irt_items(path: str, item_col: str = "item_id", domain_col: str = "domain") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (item_col, domain_col, "a", "b", "c") if c not in df.columns]
    if missing:
        raise ValueError(f"IRT item file missing columns {missing}")
    df = df.rename(
        columns={
            item_col: "item_id",
            domain_col: "domain",
        }
    )
    return df[["item_id", "domain", "a", "b", "c"]].copy()


def load_theta(path: str, user_col: str = "user_id", domain_col: str = "domain", value_col: str = "theta") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (user_col, domain_col, value_col) if c not in df.columns]
    if missing:
        raise ValueError(f"IRT theta file missing {missing}")
    return df.rename(columns={user_col: "user_id", domain_col: "domain", value_col: "theta"})
