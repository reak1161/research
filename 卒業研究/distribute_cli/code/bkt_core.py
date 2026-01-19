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
    # Guard against duplicate column names after rename
    data = data.loc[:, ~data.columns.duplicated()]
    if "order_id" in data.columns:
        data["order_id"] = pd.to_numeric(data["order_id"], errors="coerce")
    defaults_data = {
        "order_id": "order_id",
        "user_id": "user_id",
        "skill_name": "skill_name",
        "correct": "correct",
    }
    fit_kwargs = {"data": data, "defaults": dict(components=df.columns)}
    try:
        model.fit(data=data, defaults=defaults_data, forgets=forgets)
    except TypeError:
        if forgets:
            print("[warn] pyBKT version does not support forgets=True. Falling back.", flush=True)
        model.fit(data=data, defaults=defaults_data)

    try:
        pred = model.predict(data=data, defaults=defaults_data)
    except TypeError:
        pred = model.predict(data=data)

    params = extract_skill_params(model)
    return pred, params


def extract_skill_params(model: "Model") -> pd.DataFrame:
    """Convert pyBKT model parameters into a tidy DataFrame (multi-format compatible)."""
    P = getattr(model, "params", None)
    if P is None:
        raise RuntimeError("model.params is not available. Upgrade pyBKT.")
    if callable(P):
        P = P()

    skill_names = None
    for key in ("skills", "unique_skills", "KC_list", "kcs"):
        v = getattr(model, key, None)
        if v is not None:
            try:
                skill_names = [str(x) for x in list(v)]
            except Exception:
                pass
            break

    if isinstance(P, pd.DataFrame):
        df = P.copy()
        rename_map = {
            "learns": "p_T",
            "learn": "p_T",
            "slips": "p_S",
            "slip": "p_S",
            "guesses": "p_G",
            "guess": "p_G",
            "prior": "p_L0",
            "L0": "p_L0",
            "forgets": "p_F",
            "forget": "p_F",
        }
        df = df.rename(columns=rename_map)

        if isinstance(df.index, pd.MultiIndex):
            level_names = list(df.index.names)
            pick_name = None
            for cand in ("skill", "skill_name", "KC", "kc", "kcid", "KC_id", "skills"):
                if cand in level_names:
                    pick_name = cand
                    break
            if pick_name is not None:
                names = df.index.get_level_values(pick_name).astype(str)
            else:
                names = df.index.get_level_values(0).astype(str)
            df = df.reset_index()
            other_levels = [n for n in level_names if n != pick_name]
            param_level = other_levels[0] if other_levels else None
            if param_level and param_level in df.columns:
                df = df.rename(columns={pick_name: "skill_name", param_level: "param"})
            else:
                df = df.rename(columns={pick_name: "skill_name", df.columns[1]: "param"})
        else:
            if df.index.name and df.index.name not in df.columns:
                df = df.reset_index().rename(columns={df.index.name: "skill_name"})
            if "skill_name" not in df.columns:
                if skill_names is not None and len(skill_names) == len(df):
                    df.insert(0, "skill_name", skill_names)
                else:
                    df = df.reset_index()
                    df = df.rename(columns={df.columns[0]: "skill_name"})

        keep = ["skill_name", "p_T", "p_S", "p_G", "p_L0"]
        if not any(k in df.columns for k in ["p_T", "p_S", "p_G", "p_L0"]):
            cand_param = None
            for c in df.columns:
                if str(c).lower() in ("param", "parameter", "name", "variable", "par"):
                    cand_param = c
                    break
            cand_value = None
            for c in df.columns:
                if c in ("skill_name", cand_param):
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    cand_value = c
                    break
            if cand_param is not None and cand_value is not None:
                df[cand_param] = df[cand_param].astype(str).str.lower()
                piv = (
                    df.pivot_table(
                        index="skill_name",
                        columns=cand_param,
                        values=cand_value,
                        aggfunc="first",
                    )
                    .reset_index()
                )
                piv.columns = [str(x) for x in piv.columns]
                rename_map2 = {
                    "learns": "p_T",
                    "learn": "p_T",
                    "t": "p_T",
                    "slips": "p_S",
                    "slip": "p_S",
                    "s": "p_S",
                    "guesses": "p_G",
                    "guess": "p_G",
                    "g": "p_G",
                    "prior": "p_L0",
                    "l0": "p_L0",
                    "prior_l0": "p_L0",
                }
                piv = piv.rename(columns=rename_map2)
                for k in keep:
                    if k not in piv.columns:
                        piv[k] = np.nan
                piv["skill_name"] = piv["skill_name"].astype(str)
                df = piv[keep]
            else:
                return pd.DataFrame({k: [] for k in keep})
        else:
            for k in keep:
                if k not in df.columns:
                    df[k] = np.nan
            if "skill_name" not in df.columns:
                if skill_names is not None and len(skill_names) == len(df):
                    df.insert(0, "skill_name", skill_names)
                else:
                    df.insert(0, "skill_name", df.index.astype(str))
            df["skill_name"] = df["skill_name"].astype(str)
            df = df[keep]

        df = (
            df.groupby("skill_name", as_index=False)
            .agg({"p_T": "mean", "p_S": "mean", "p_G": "mean", "p_L0": "mean"})
            .rename(columns={"p_T": "T", "p_S": "S", "p_G": "G", "p_L0": "L0"})
        )
        return df

    if isinstance(P, dict):
        src = P
        for key in ("by_skill", "skills", "kcs"):
            if key in src and isinstance(src[key], dict):
                src = src[key]
                break

        def to_series(val):
            if isinstance(val, dict):
                s = pd.Series(val)
                s.index = s.index.astype(str)
                return s
            arr = list(val) if hasattr(val, "__len__") else []
            return pd.Series(arr, index=[str(i) for i in range(len(arr))])

        entries = []
        for param_name in ("p_L0", "p_T", "p_S", "p_G", "L0", "T", "S", "G"):
            if param_name in src:
                entries.append((param_name, to_series(src[param_name])))
        if not entries:
            return pd.DataFrame(columns=["skill_name", "L0", "T", "S", "G"])
        frames = []
        for param_name, series in entries:
            df_param = series.reset_index()
            df_param.columns = ["skill_name", param_name]
            frames.append(df_param)
        df = frames[0]
        for add in frames[1:]:
            df = df.merge(add, on="skill_name", how="outer")
        df = df.rename(
            columns={
                "p_L0": "L0",
                "p_T": "T",
                "p_S": "S",
                "p_G": "G",
            }
        )
        keep = ["skill_name", "L0", "T", "S", "G"]
        for k in keep:
            if k not in df.columns:
                df[k] = np.nan
        return df[keep]

    return pd.DataFrame(columns=["skill_name", "L0", "T", "S", "G"])


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
