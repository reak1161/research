"""Core utilities for fitting and simulating Bayesian Knowledge Tracing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from pyBKT.models import Model
except ImportError as exc:  # pragma: no cover - pyBKT should be installed but guard anyway
        Model = None  # type: ignore


class MissingPyBKTError(RuntimeError):
    """Raised when pyBKT is not available in the current environment."""


@dataclass
class BKTParams:
    """Container for the canonical 4-parameter BKT model."""

    L0: float
    T: float
    S: float
    G: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "BKTParams":
        return cls(
            L0=float(row.get("L0", row.get("p_L0", 0.1))),
            T=float(row.get("T", row.get("p_T", 0.15))),
            S=float(row.get("S", row.get("p_S", 0.1))),
            G=float(row.get("G", row.get("p_G", 0.2))),
        )


def _require_model() -> None:
    if Model is None:
        raise MissingPyBKTError("pyBKT is not installed. Please `pip install pyBKT==1.4.1`.")


def fit_bkt(
    df: pd.DataFrame,
    cols: Dict[str, str],
    *,
    forgets: bool = False,
    seed: int = 42,
    num_fits: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit pyBKT on the provided log DataFrame and return predictions + params."""
    _require_model()
    model = Model(seed=seed, num_fits=num_fits)
    data = df.rename(
        columns={
            cols["order_id"]: "order_id",
            cols["user_id"]: "user_id",
            cols["skill_name"]: "skill_name",
            cols["correct"]: "correct",
        }
    )
    fit_kwargs = {"data": data, "defaults": dict(components=df.columns)}
    try:
        model.fit(data=data, defaults=cols, forgets=forgets)
    except TypeError:
        if forgets:
            print("[warn] pyBKT version does not support forgets=True. Falling back.", flush=True)
        model.fit(data=data, defaults=cols)

    try:
        pred = model.predict(data=data, defaults=cols)
    except TypeError:
        pred = model.predict(data=data)

    params = extract_skill_params(model)
    return pred, params


def extract_skill_params(model: "Model") -> pd.DataFrame:
    """Convert pyBKT model parameters into a tidy DataFrame."""
    P = getattr(model, "params", None)
    if callable(P):
        P = P()
    if P is None:
        raise RuntimeError("model.params is not available. Upgrade pyBKT.")
    df = pd.DataFrame(P).copy()
    rename_map = {
        "learns": "T",
        "learn": "T",
        "slips": "S",
        "slip": "S",
        "guesses": "G",
        "guess": "G",
        "prior": "L0",
        "L0": "L0",
    }
    df = df.rename(columns=rename_map)
    if "skill_name" not in df.columns:
        df.insert(0, "skill_name", getattr(model, "skills", range(len(df))))
    keep_cols = ["skill_name", "L0", "T", "S", "G"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep_cols]
    return df


def compute_skill_metrics(
    predictions: pd.DataFrame,
    *,
    skill_col: str,
    correct_col: str,
    score_col: str,
) -> pd.DataFrame:
    """Aggregate ROC-AUC and logloss per skill."""
    rows: List[Dict[str, float]] = []
    for skill, group in predictions.groupby(skill_col):
        y_true = group[correct_col].to_numpy()
        y_score = group[score_col].to_numpy()
        rows.append(
            {
                "skill": skill,
                "n": int(group.shape[0]),
                "auc": _roc_auc_safe(y_true, y_score),
                "logloss": _logloss_safe(y_true, y_score),
            }
        )
    return pd.DataFrame(rows)


def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _logloss_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import log_loss

    eps = 1e-6
    clipped = np.clip(y_score, eps, 1 - eps)
    try:
        return float(log_loss(y_true, clipped))
    except ValueError:
        return float("nan")


def rollout_bkt_sequence(
    correct_seq: Iterable[int],
    params: BKTParams,
) -> pd.DataFrame:
    """Simulate BKT posterior for a single (user, skill) sequence."""
    rows: List[Dict[str, float]] = []
    p_L = float(params.L0)
    eps = 1e-9
    for idx, obs in enumerate(correct_seq):
        obs_bin = 1 if int(obs) == 1 else 0
        p_correct_now = p_L * (1.0 - params.S) + (1.0 - p_L) * params.G
        denom = p_correct_now if obs_bin == 1 else (1.0 - p_correct_now)
        denom = max(denom, eps)
        if obs_bin == 1:
            p_posterior = (p_L * (1.0 - params.S)) / denom
        else:
            p_posterior = (p_L * params.S) / denom
        p_after = p_posterior + (1.0 - p_posterior) * params.T
        p_next = p_after * (1.0 - params.S) + (1.0 - p_after) * params.G
        rows.append(
            {
                "step": idx,
                "p_L_prior": p_L,
                "p_correct_now": p_correct_now,
                "p_L_post": p_posterior,
                "p_L_after": p_after,
                "p_next": p_next,
                "observation": obs_bin,
            }
        )
        p_L = p_after
    return pd.DataFrame(rows)


def update_state(p_L: float, params: BKTParams, observation: int) -> Dict[str, float]:
    """Single-step BKT update returning intermediate probabilities."""
    obs = 1 if int(observation) == 1 else 0
    eps = 1e-9
    p_correct_now = p_L * (1.0 - params.S) + (1.0 - p_L) * params.G
    denom = p_correct_now if obs == 1 else (1.0 - p_correct_now)
    denom = max(denom, eps)
    if obs == 1:
        p_posterior = (p_L * (1.0 - params.S)) / denom
    else:
        p_posterior = (p_L * params.S) / denom
    p_after = p_posterior + (1.0 - p_posterior) * params.T
    p_next = p_after * (1.0 - params.S) + (1.0 - p_after) * params.G
    return {
        "p_L_prior": p_L,
        "p_correct_now": p_correct_now,
        "p_L_post": p_posterior,
        "p_L_after": p_after,
        "p_next": p_next,
    }


def summarize_user_skill_states(
    df: pd.DataFrame,
    params_df: pd.DataFrame,
    *,
    user_col: str,
    skill_col: str,
    correct_col: str,
) -> pd.DataFrame:
    """Roll out BKT for every user×skill pair using the supplied parameters."""
    params_df = params_df.set_index("skill_name")
    summaries: List[pd.Series] = []
    for (user, skill), group in df.groupby([user_col, skill_col]):
        if skill not in params_df.index:
            continue
        params = BKTParams.from_row(params_df.loc[skill])
        seq = rollout_bkt_sequence(group[correct_col].tolist(), params)
        if seq.empty:
            continue
        last = seq.iloc[-1]
        summaries.append(
            pd.Series(
                {
                    "user_id": user,
                    "skill_name": skill,
                    "p_L_after": last["p_L_after"],
                    "p_next": last["p_next"],
                    "n_obs": group.shape[0],
                }
            )
        )
    if not summaries:
        return pd.DataFrame(columns=["user_id", "skill_name", "p_L_after", "p_next", "n_obs"])
    return pd.DataFrame(summaries)
