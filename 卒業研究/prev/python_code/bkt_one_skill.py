# -*- coding: utf-8 -*-
"""
BKT (weighted MAP-EM) + Multi-skill prediction + Recommendation
- A) Online user-state update (per item outcome)
- B) Cooldown penalty & MMR diversification
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datetime as dt
import numpy as np
import pandas as pd


# =========================
# 0) 設定
# =========================
CSV_PATH = "../csv/200_50_500_250000.csv"

# A案（explode＋1/m重み）で学習するか？
USE_WEIGHTED_EM = False

# 学習対象とするスキルの「有効サンプル数」(重み合計)の下限
MIN_EFFECTIVE_SAMPLES = 200

# EMの反復回数
EM_MAX_ITERS = 30

# 期待値は slip/guess/prior ≈ 0.2、learn ≈ 0.03 あたりに寄せる（MAP-EMの先験）
BETA_PRIOR = {
    "L0": (2.0, 8.0),   # mean 0.20
    "T":  (3.0, 97.0),  # mean 0.03 （1問あたり学習）
    "S":  (2.0, 8.0),   # mean 0.20
    "G":  (2.0, 8.0),   # mean 0.20
}

# 物理クランプ（安全網）
CLAMP = {
    "L0": (0.01, 0.90),
    "T":  (0.001, 0.30),
    "S":  (0.01, 0.40),
    "G":  (0.01, 0.40),
}

# マルチスタートの初期値候補
INIT_GRID = [
    (0.2, 0.02, 0.10, 0.20),
    (0.3, 0.05, 0.15, 0.15),
    (0.1, 0.03, 0.05, 0.25),
]

# デバッグ用：ラベル反転（False推奨）
FLIP_LABELS = False


# =========================
# 1) 前処理
# =========================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    if FLIP_LABELS:
        df["result"] = 1 - df["result"].astype(int)
    return df


def single_skill_subset(df_all: pd.DataFrame) -> pd.DataFrame:
    df_single = df_all[df_all["skill"].astype(str).str.count(",") == 0].copy()
    df_single["skill_id"] = df_single["skill"].astype(int)
    df_single = df_single.sort_values(["user_id", "skill_id", "date"])
    return df_single


def prepare_exploded(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    skill='a,b,c' を行分解し、重み=1/m を付与
    """
    skl = df_all["skill"].astype(str).str.split(",")
    m = skl.str.len()
    tmp = df_all.copy()
    tmp["skills_list"] = skl
    tmp["m"] = m
    ex = tmp.explode("skills_list", ignore_index=True)
    ex["skill_id"] = ex["skills_list"].astype(int)
    ex["weight"] = 1.0 / ex["m"].astype(float)
    ex = ex[["user_id", "item_id", "skill_id", "result", "date", "weight"]]
    ex = ex.sort_values(["user_id", "skill_id", "date"])
    return ex


# =========================
# 2) BKT（忘却なし） — EM学習
# =========================
def bkt_em_standard(sequences: List[List[int]],
                    max_iter: int = 30,
                    tol: float = 1e-4,
                    init: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
    """
    標準BKT（忘却なし）のEM（単一スキル、重みなし）
    sequences: ユーザごとの {0/1} 観測系列のリスト
    """
    if init is None:
        p_L0, p_T, p_S, p_G = 0.2, 0.1, 0.1, 0.2
    else:
        p_L0, p_T, p_S, p_G = init

    def clip(x: float) -> float:
        return float(np.clip(x, 1e-5, 1 - 1e-5))

    last_ll = None
    for it in range(max_iter):
        A = np.array([[1 - p_T, p_T], [0.0, 1.0]])  # U->Lのみ、L->Uなし
        sum_gamma_L1 = sum_xi_U2L = sum_gamma_U = 0.0
        sum_emit_L = sum_emit_L_incorrect = sum_emit_U = sum_emit_U_correct = 0.0
        total_ll = 0.0

        for obs in sequences:
            T = len(obs)
            if T == 0:
                continue
            pi = np.array([1 - p_L0, p_L0])
            # emission
            def b(o):  # [U, L]
                return np.array([p_G if o == 1 else (1 - p_G),
                                 (1 - p_S) if o == 1 else p_S])

            # forward-backward（スケーリングあり）
            alpha = np.zeros((T, 2))
            scale = np.zeros(T)
            alpha[0] = pi * b(obs[0])
            scale[0] = alpha[0].sum() + 1e-300
            alpha[0] /= scale[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1].dot(A)) * b(obs[t])
                scale[t] = alpha[t].sum() + 1e-300
                alpha[t] /= scale[t]
            beta = np.zeros((T, 2))
            beta[-1] = 1.0 / scale[-1]
            for t in range(T - 2, -1, -1):
                bt1 = b(obs[t + 1])
                beta[t] = (A * (bt1 * beta[t + 1])).sum(axis=1) / scale[t]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            # U->L 期待回数
            for t in range(T - 1):
                bt1 = b(obs[t + 1])
                num = alpha[t, 0] * p_T * bt1[1] * beta[t + 1, 1]
                den = (alpha[t, 0] * (1 - p_T) * bt1[0] * beta[t + 1, 0]
                       + alpha[t, 0] * p_T * bt1[1] * beta[t + 1, 1]
                       + alpha[t, 1] * 1.0 * bt1[1] * beta[t + 1, 1])
                sum_xi_U2L += 0.0 if den <= 0 else num / den
                sum_gamma_U += gamma[t, 0]

            # 放出カウント
            obs_arr = np.array(obs, dtype=float)
            sum_emit_L += gamma[:, 1].sum()
            sum_emit_L_incorrect += ((1 - obs_arr) * gamma[:, 1]).sum()
            sum_emit_U += gamma[:, 0].sum()
            sum_emit_U_correct += (obs_arr * gamma[:, 0]).sum()
            sum_gamma_L1 += gamma[0, 1]

            total_ll += np.sum(np.log(scale + 1e-300))

        # M-step
        nseq = max(len(sequences), 1)
        p_L0 = clip(sum_gamma_L1 / nseq)
        p_T = clip(sum_xi_U2L / max(sum_gamma_U, 1e-12))
        p_S = clip(sum_emit_L_incorrect / max(sum_emit_L, 1e-12))
        p_G = clip(sum_emit_U_correct / max(sum_emit_U, 1e-12))

        if last_ll is not None and abs(total_ll - last_ll) < tol:
            break
        last_ll = total_ll

    return {"p_L0": p_L0, "p_T": p_T, "p_S": p_S, "p_G": p_G, "loglik": last_ll, "iters": it + 1}


def fit_per_skill_standard(df_single: pd.DataFrame, max_iter: int = 30) -> Dict[int, Dict[str, float]]:
    params: Dict[int, Dict[str, float]] = {}
    for sid, sdf in df_single.groupby("skill_id"):
        seqs = [udf["result"].astype(int).tolist() for _, udf in sdf.groupby("user_id")]
        params[sid] = bkt_em_standard(seqs, max_iter=max_iter)
    return params


def fit_bkt_weighted_for_skill(df_skill: pd.DataFrame,
                               max_iter: int = 30,
                               tol: float = 1e-4,
                               init: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
    """
    重み付きEM（忘却なし）＋ MAP正則化（Beta事前）＋ クランプ ＋ マルチスタート
    df_skill: 単一 skill_id のデータ（user_idで系列化済み）
    """
    # ユーザ別系列（obs, wts）
    seqs: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, udf in df_skill.groupby("user_id", sort=False):
        obs = udf["result"].astype(int).to_numpy()
        wts = udf["weight"].astype(float).to_numpy()
        if len(obs) > 0:
            seqs.append((obs, wts))

    def run_em_once(init_tuple: Tuple[float, float, float, float]) -> Dict[str, Any]:
        p_L0, p_T, p_S, p_G = init_tuple
        last_ll = None

        for it in range(max_iter):
            A = np.array([[1 - p_T, p_T], [0.0, 1.0]], dtype=float)
            B1 = np.array([p_G, 1 - p_S], dtype=float)   # o=1
            B0 = np.array([1 - p_G, p_S], dtype=float)   # o=0

            sum_gamma_L1_w = 0.0
            sum_first_w = 0.0
            sum_gamma_U_w = 0.0
            sum_xi_U2L_w = 0.0
            sum_emit_L_w = 0.0
            sum_emit_L_incorrect_w = 0.0
            sum_emit_U_w = 0.0
            sum_emit_U_correct_w = 0.0
            total_ll = 0.0

            for obs, wts in seqs:
                T = len(obs)
                if T == 0:
                    continue
                pi = np.array([1 - p_L0, p_L0], dtype=float)

                # Forward
                alpha = np.zeros((T, 2), dtype=float)
                scale = np.zeros(T, dtype=float)
                alpha[0] = pi * (B1 if obs[0] == 1 else B0)
                s = alpha[0].sum() + 1e-300
                scale[0] = s
                alpha[0] /= s
                for t in range(1, T):
                    b = B1 if obs[t] == 1 else B0
                    alpha[t] = (alpha[t - 1].dot(A)) * b
                    s = alpha[t].sum() + 1e-300
                    scale[t] = s
                    alpha[t] /= s

                # Backward
                beta = np.zeros((T, 2), dtype=float)
                beta[-1] = 1.0 / scale[-1]
                for t in range(T - 2, -1, -1):
                    b = B1 if obs[t + 1] == 1 else B0
                    tmp0 = (1 - p_T) * b[0] * beta[t + 1, 0] + p_T * b[1] * beta[t + 1, 1]
                    tmp1 = 1.0 * b[1] * beta[t + 1, 1]
                    beta[t] = np.array([tmp0, tmp1]) / scale[t]

                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)

                # 期待カウント（重み付き）
                for t in range(T - 1):
                    b = B1 if obs[t + 1] == 1 else B0
                    num = alpha[t, 0] * p_T * b[1] * beta[t + 1, 1]
                    den = (alpha[t, 0] * (1 - p_T) * b[0] * beta[t + 1, 0]
                           + alpha[t, 0] * p_T * b[1] * beta[t + 1, 1]
                           + alpha[t, 1] * 1.0 * b[1] * beta[t + 1, 1])
                    if den > 0:
                        sum_xi_U2L_w += (num / den) * wts[t]
                    sum_gamma_U_w += gamma[t, 0] * wts[t]

                o_arr = obs.astype(float)
                w_arr = wts
                sum_emit_L_w += (gamma[:, 1] * w_arr).sum()
                sum_emit_L_incorrect_w += ((1 - o_arr) * gamma[:, 1] * w_arr).sum()
                sum_emit_U_w += (gamma[:, 0] * w_arr).sum()
                sum_emit_U_correct_w += (o_arr * gamma[:, 0] * w_arr).sum()

                sum_gamma_L1_w += gamma[0, 1] * wts[0]
                sum_first_w += wts[0]
                total_ll += np.log(scale + 1e-300).sum()

            # M-step（MAP + クランプ）
            a_L0, b_L0 = BETA_PRIOR["L0"]
            a_T, b_T = BETA_PRIOR["T"]
            a_S, b_S = BETA_PRIOR["S"]
            a_G, b_G = BETA_PRIOR["G"]

            p_L0 = (sum_gamma_L1_w + (a_L0 - 1)) / (max(sum_first_w, 1e-12) + (a_L0 + b_L0 - 2))
            p_T = (sum_xi_U2L_w + (a_T - 1)) / (max(sum_gamma_U_w, 1e-12) + (a_T + b_T - 2))
            p_S = (sum_emit_L_incorrect_w + (a_S - 1)) / (max(sum_emit_L_w, 1e-12) + (a_S + b_S - 2))
            p_G = (sum_emit_U_correct_w + (a_G - 1)) / (max(sum_emit_U_w, 1e-12) + (a_G + b_G - 2))

            p_L0 = float(np.clip(p_L0, *CLAMP["L0"]))
            p_T = float(np.clip(p_T, *CLAMP["T"]))
            p_S = float(np.clip(p_S, *CLAMP["S"]))
            p_G = float(np.clip(p_G, *CLAMP["G"]))

            if last_ll is not None and abs(total_ll - last_ll) < tol:
                return {"p_L0": p_L0, "p_T": p_T, "p_S": p_S, "p_G": p_G,
                        "loglik": total_ll, "iters": it + 1}
            last_ll = total_ll

        return {"p_L0": p_L0, "p_T": p_T, "p_S": p_S, "p_G": p_G,
                "loglik": last_ll, "iters": it + 1}

    # ==== マルチスタート ====
    inits = [init] if init is not None else INIT_GRID
    best: Optional[Dict[str, Any]] = None
    for ini in inits:
        cand = run_em_once(ini)
        if (best is None) or (cand["loglik"] is not None and cand["loglik"] > best["loglik"]):
            best = cand
    assert best is not None
    return best


def fit_per_skill_weighted(df_exploded: pd.DataFrame,
                           max_iter: int = 30,
                           min_eff_samples: int = 200) -> Dict[int, Dict[str, float]]:
    """重み合計が閾値以上のスキルだけ学習"""
    eff = df_exploded.groupby("skill_id")["weight"].sum()
    params: Dict[int, Dict[str, float]] = {}
    for sid in eff.index:
        if eff.loc[sid] < min_eff_samples:
            continue
        sdf = df_exploded[df_exploded["skill_id"] == sid].sort_values(["user_id", "date"])
        params[sid] = fit_bkt_weighted_for_skill(sdf, max_iter=max_iter)
    return params


# =========================
# 3) 推論（単一スキル）
# =========================
def bkt_predict_next(obs: List[int], p: Dict[str, float]) -> Tuple[float, float, float]:
    pL = p["p_L0"]
    for o in obs:
        pL = pL + (1 - pL) * p["p_T"]  # 事前学習
        if o == 1:
            num = (1 - p["p_S"]) * pL
            den = num + p["p_G"] * (1 - pL)
        else:
            num = p["p_S"] * pL
            den = num + (1 - p["p_G"]) * (1 - pL)
        pL = pL if den <= 0 else num / den
    pL_next = pL + (1 - pL) * p["p_T"]
    p_correct = (1 - p["p_S"]) * pL_next + p["p_G"] * (1 - pL_next)
    return pL, pL_next, p_correct


def predict_user_single(df_single: pd.DataFrame,
                        params_by_skill: Dict[int, Dict[str, float]],
                        user_id: int) -> pd.DataFrame:
    rows = []
    for sid, sdf in df_single[df_single["user_id"] == user_id].groupby("skill_id"):
        obs = sdf.sort_values("date")["result"].astype(int).tolist()
        p = params_by_skill.get(sid)
        if not p:
            continue
        pL, pL_next, p_corr = bkt_predict_next(obs, p)
        rows.append({"user_id": user_id, "skill_id": sid, "n_obs": len(obs),
                     "P(L)_posterior": pL, "P(L)_after_learn": pL_next, "P(next correct)": p_corr,
                     "learn": p["p_T"], "slip": p["p_S"], "guess": p["p_G"]})
    out = pd.DataFrame(rows).sort_values("P(next correct)", ascending=False)
    return out


# =========================
# 4) 多スキルの結合・期待上昇Δ
# =========================
def combine_Tks(Tks: List[float], mode: str = "and", alpha: float = 0.5) -> Optional[float]:
    """
    mode:
      - "and"       : 積（Noisy-AND）
      - "geom"      : 幾何平均
      - "logit_avg" : ロジット平均
      - "soft_and"  : (∏ T_k) ** α   （α∈(0,1]でANDの潰れを緩和）
    """
    if not Tks:
        return None
    Tks = [float(np.clip(t, 1e-9, 1 - 1e-9)) for t in Tks]
    if mode == "and":
        p = 1.0
        for t in Tks:
            p *= t
        return p
    if mode == "geom":
        prod = 1.0
        for t in Tks:
            prod *= t
        return prod ** (1.0 / len(Tks))
    if mode == "logit_avg":
        logits = [math.log(t / (1 - t)) for t in Tks]
        avg = sum(logits) / len(logits)
        return 1.0 / (1.0 + math.exp(-avg))
    if mode == "soft_and":
        prod = 1.0
        for t in Tks:
            prod *= t
        return prod ** alpha
    raise ValueError("unknown combine mode")


def predict_item_multiskill(user_id: int, item_id: int,
                            df_all: pd.DataFrame,
                            df_history_exploded: pd.DataFrame,
                            params_by_skill: Dict[int, Dict[str, float]],
                            combine_mode: str = "and",
                            use_noisy_and: Optional[bool] = None) -> Dict[str, Any]:
    """
    指定ユーザに item_id を出題した場合の正答確率を推定
    combine_mode: "and" / "geom" / "logit_avg" / "soft_and"
    use_noisy_and: 後方互換（True→"and", False→"geom"）
    """
    if use_noisy_and is not None:
        combine_mode = "and" if use_noisy_and else "geom"

    skills_str = df_all.loc[df_all["item_id"] == item_id, "skill"].astype(str).iloc[0]
    skill_ids = [int(s) for s in skills_str.split(",")]

    per_skill = []
    for sid in skill_ids:
        if sid not in params_by_skill:
            continue
        p = params_by_skill[sid]
        sdf = df_history_exploded[(df_history_exploded["user_id"] == user_id) &
                                  (df_history_exploded["skill_id"] == sid)].sort_values("date")
        obs = sdf["result"].astype(int).tolist()
        pL, pL_next, _ = bkt_predict_next(obs, p)
        T_k = (1 - p["p_S"]) * pL_next + p["p_G"] * (1 - pL_next)
        per_skill.append({"skill_id": sid, "P(L)": pL, "P(L^+)": pL_next, "T_k": T_k,
                          "learn": p["p_T"], "slip": p["p_S"], "guess": p["p_G"]})

    if len(per_skill) == 0:
        return {"user_id": user_id, "item_id": item_id, "skills": skill_ids,
                "p_correct": None, "per_skill": []}

    p_correct = combine_Tks([r["T_k"] for r in per_skill], mode=combine_mode)
    return {"user_id": user_id, "item_id": item_id, "skills": skill_ids,
            "p_correct": p_correct, "per_skill": per_skill}


def expected_gain_for_item(user_id: int, item_id: int,
                           df_all: pd.DataFrame,
                           df_history_exploded: pd.DataFrame,
                           params_by_skill: Dict[int, Dict[str, float]],
                           combine_mode: str = "geom",
                           alpha: float = 0.5) -> Optional[Dict[str, Any]]:
    """
    期待習熟上昇（平均Δ）と p_correct を同時に返す
    """
    skills_str = df_all.loc[df_all["item_id"] == item_id, "skill"].astype(str).iloc[0]
    K = [int(s) for s in skills_str.split(",")]

    per_skill = []
    for sid in K:
        if sid not in params_by_skill:
            continue
        p = params_by_skill[sid]
        sdf = df_history_exploded[(df_history_exploded["user_id"] == user_id) &
                                  (df_history_exploded["skill_id"] == sid)].sort_values("date")
        obs = sdf["result"].astype(int).tolist()
        pL, pL_next, _ = bkt_predict_next(obs, p)
        T_k = (1 - p["p_S"]) * pL_next + p["p_G"] * (1 - pL_next)
        per_skill.append({"skill_id": sid, "P(L)": pL, "P(L^+)": pL_next, "T_k": T_k,
                          "learn": p["p_T"], "slip": p["p_S"], "guess": p["p_G"]})
    if not per_skill:
        return None

    Tks = [r["T_k"] for r in per_skill]
    p_corr = combine_Tks(Tks, mode=combine_mode, alpha=alpha)
    if p_corr is None:
        return None

    # AND前提の厳密式で誤答時の事後を計算（他モードでも近似として利用）
    Q = 1.0
    for t in Tks:
        Q *= t

    gains = []
    for r in per_skill:
        Tk = r["T_k"]
        p = params_by_skill[r["skill_id"]]
        pL_next = r["P(L^+)"]

        # 正答時
        post1 = ((1 - p["p_S"]) * pL_next) / (((1 - p["p_S"]) * pL_next) + (p["p_G"] * (1 - pL_next)))
        # 誤答時（Noisy-ANDの式）
        Q_minus = Q / max(Tk, 1e-12)
        num0 = (1 - (1 - p["p_S"]) * Q_minus) * pL_next
        den0 = num0 + (1 - p["p_G"] * Q_minus) * (1 - pL_next)
        post0 = num0 / max(den0, 1e-12)

        after1 = post1 + (1 - post1) * p["p_T"]
        after0 = post0 + (1 - post0) * p["p_T"]
        E_after = p_corr * after1 + (1 - p_corr) * after0
        gains.append(E_after - r["P(L)"])

    delta = float(sum(gains) / len(gains))
    return {"user_id": user_id, "item_id": item_id, "skills": K,
            "p_correct": p_corr, "delta": delta,
            "per_skill": per_skill, "per_skill_gain": gains}


# =========================
# 5) 推薦（スコアリング・多様化・クールダウン）
# =========================
def score_multi_objective(delta: float, p: float,
                          target_band: Tuple[float, float] = (0.4, 0.7),
                          lam: float = 0.3) -> float:
    """
    lam: 0〜1。0ならΔだけ、1なら帯への近さだけ。
    """
    lo, hi = target_band
    if p < lo:
        dist = (lo - p)
    elif p > hi:
        dist = (p - hi)
    else:
        dist = 0.0
    closeness = 1.0 / (1.0 + 10.0 * dist)  # 係数は適宜
    return (1 - lam) * delta + lam * closeness


def auto_target_band_from_preds(p_list: List[float],
                                low_q: float = 0.60,
                                high_q: float = 0.80) -> Tuple[float, float]:
    p = np.array(p_list, dtype=float)
    lo = float(np.quantile(p, low_q))
    hi = float(np.quantile(p, high_q))
    if hi - lo < 0.05:
        mid = (lo + hi) / 2
        lo, hi = max(0.01, mid - 0.03), min(0.99, mid + 0.03)
    return (lo, hi)


def recommend_topK(user_id: int, K: int,
                   df_all: pd.DataFrame,
                   df_exploded: pd.DataFrame,
                   params_by_skill: Dict[int, Dict[str, float]],
                   combine_mode: str = "soft_and",
                   alpha: float = 0.5,
                   target_band: str | Tuple[float, float] = "auto",
                   lam: float = 0.3,
                   max_candidates: int = 2000) -> pd.DataFrame:
    seen = set(df_all[df_all["user_id"] == user_id]["item_id"].unique())
    cand_all = df_all["item_id"].unique().tolist()
    cand = [iid for iid in cand_all if iid not in seen][:max_candidates]

    # --- 1st pass: p, Δ の粗計算（帯の自動化用） ---
    scratch: List[Tuple[int, float, float, Tuple[int, ...]]] = []
    for iid in cand:
        eg = expected_gain_for_item(user_id, iid, df_all, df_exploded,
                                    params_by_skill, combine_mode=combine_mode, alpha=alpha)
        if eg is None:
            continue
        scratch.append((iid, eg["p_correct"], eg["delta"], tuple(sorted(eg["skills"]))))

    if not scratch:
        return pd.DataFrame(columns=["item_id", "p_correct", "delta", "score", "skills"])

    if target_band == "auto":
        band = auto_target_band_from_preds([p for _, p, _, _ in scratch], low_q=0.50, high_q=0.75)
    else:
        band = target_band  # type: ignore[assignment]

    # --- 2nd pass: 多目的スコア ---
    rows = []
    for iid, p, dlt, skills in scratch:
        s = score_multi_objective(dlt, p, target_band=band, lam=lam)  # type: ignore[arg-type]
        rows.append({"item_id": iid, "p_correct": p, "delta": dlt, "score": s, "skills": skills})
    rec = pd.DataFrame(rows).sort_values(["score", "delta", "p_correct"], ascending=[False, False, False])
    return rec


def jaccard(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    A, B = set(a), set(b)
    return 0.0 if not A or not B else len(A & B) / len(A | B)


def diversify_by_skills(recs_df: pd.DataFrame, topN: int = 10, mmr_lambda: float = 0.7) -> pd.DataFrame:
    """
    recs_df: columns ['item_id','score','skills', ...]
    mmr_lambda: 1に近いほど元スコア重視、0に近いほど多様性重視
    """
    if recs_df.empty:
        return recs_df.copy()
    selected: List[Dict[str, Any]] = []
    pool = recs_df.to_dict("records")
    while pool and len(selected) < topN:
        if not selected:
            selected.append(pool.pop(0))
            continue
        best = None
        best_val = -1e9
        for r in pool:
            sim = 0.0
            for s in selected:
                sim = max(sim, jaccard(tuple(r["skills"]), tuple(s["skills"])))
            val = mmr_lambda * r["score"] - (1 - mmr_lambda) * sim
            if val > best_val:
                best_val, best = val, r
        selected.append(best)  # type: ignore[arg-type]
        pool.remove(best)      # type: ignore[arg-type]
    return pd.DataFrame(selected)


# --- B) クールダウン ---
from collections import defaultdict
skill_last_seen = defaultdict(lambda: -10**9)  # スキルの最終出題ステップ
current_step = 0


def add_cooldown_penalty(recs_df: pd.DataFrame,
                         skill_last_seen: Dict[int, int],
                         step: int,
                         cooldown: int = 5,
                         max_penalty: float = 0.2) -> pd.DataFrame:
    """
    cooldown: 直近 'cooldown' ステップ以内に出したスキルを含むアイテムを減点
    max_penalty: 最大でスコアを何割落とすか（0.2=20%）
    """
    if recs_df.empty:
        return recs_df.copy()

    def penalize_row(row: pd.Series) -> float:
        skills = set(row["skills"])
        if not skills:
            return row["score"]
        gaps = [step - skill_last_seen.get(s, -10**9) for s in skills]
        if min(gaps) < cooldown:
            k = (cooldown - min(gaps)) / max(cooldown, 1)
            return row["score"] * (1 - max_penalty * k)
        return row["score"]

    out = recs_df.copy()
    out["score"] = out.apply(penalize_row, axis=1)
    return out.sort_values(["score", "delta", "p_correct"], ascending=[False, False, False])


def update_skill_last_seen(skill_last_seen: Dict[int, int], item_id: int, step: int, df_all: pd.DataFrame) -> None:
    skills_str = str(df_all.loc[df_all["item_id"] == item_id, "skill"].iloc[0])
    for s in [int(x) for x in skills_str.split(",")]:
        skill_last_seen[s] = step


# =========================
# 6) A) オンライン更新（ユーザ状態）
# =========================
def init_user_state(user_id: int,
                    df_exploded: pd.DataFrame,
                    params_by_skill: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    """
    履歴から user_state[sid] = P(L) を構築
    """
    state: Dict[int, float] = {}
    for sid, p in params_by_skill.items():
        sdf = df_exploded[(df_exploded["user_id"] == user_id) &
                          (df_exploded["skill_id"] == sid)].sort_values("date")
        obs = sdf["result"].astype(int).tolist()
        pL, _, _ = bkt_predict_next(obs, p)
        state[sid] = pL
    return state


def bkt_update_state_for_item(user_state: Dict[int, float],
                              item_id: int,
                              outcome: int,
                              params_by_skill: Dict[int, Dict[str, float]],
                              df_all: pd.DataFrame) -> Dict[int, float]:
    """
    出題 item_id の観測 outcome ∈ {0,1} を受けて、関係スキルだけ更新
    """
    skills_str = str(df_all.loc[df_all["item_id"] == item_id, "skill"].iloc[0])
    skills = [int(s) for s in skills_str.split(",")]
    for sid in skills:
        if sid not in params_by_skill:
            continue
        p = params_by_skill[sid]
        pL = user_state.get(sid, p["p_L0"])
        pL_prior = pL + (1 - pL) * p["p_T"]
        if outcome == 1:
            num = (1 - p["p_S"]) * pL_prior
            den = num + p["p_G"] * (1 - pL_prior)
        else:
            num = p["p_S"] * pL_prior
            den = num + (1 - p["p_G"]) * (1 - pL_prior)
        pL_post = pL_prior if den <= 0 else num / den
        user_state[sid] = float(np.clip(pL_post, 1e-9, 1 - 1e-9))
    return user_state


# =========================
# 7) I/O ユーティリティ（保存・読み込み）
# =========================
def build_llm_payload(user_id: int, recs: pd.DataFrame) -> Dict[str, Any]:
    payload = {"user_id": user_id,
               "policy": "maximize_expected_mastery_gain + target_band(auto)",
               "recommendations": []}
    for _, r in recs.iterrows():
        payload["recommendations"].append({
            "item_id": int(r["item_id"]),
            "p_next_correct": float(r["p_correct"]),
            "expected_mastery_gain": float(r["delta"]),
            "score": float(r["score"]),
            "reason": f"skills={list(r['skills'])}, p≈{r['p_correct']:.2f}, Δ≈{r['delta']:.3f}"
        })
    return payload


def save_payload_json(payload: Dict[str, Any],
                      user_id: int,
                      out_dir: str = "../json",
                      prefix: str = "recommendations") -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = out_path / f"{prefix}_user{user_id}_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved JSON] {fname.resolve()}")
    return str(fname)


def load_latest_recs(out_dir: str = "../json",
                     user_id: Optional[int] = None,
                     prefix: str = "recommendations") -> Tuple[Path, Dict[str, Any], pd.DataFrame]:
    out = Path(out_dir)
    pattern = f"{prefix}_user{user_id}_*.json" if user_id is not None else f"{prefix}_user*_*.json"
    files = sorted(out.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {out} with pattern {pattern}")
    path = files[-1]
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    recs = pd.DataFrame(payload["recommendations"])
    recs["rank"] = range(1, len(recs) + 1)
    return path, payload, recs


def show_top(recs: pd.DataFrame, topN: int = 10) -> None:
    cols = ["rank", "item_id", "p_next_correct", "expected_mastery_gain", "score"]
    print(recs[cols].head(topN).to_string(index=False))


def save_recs_csv(recs: pd.DataFrame,
                  user_id: int,
                  out_dir: str = "../json",
                  prefix: str = "recs_latest") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}_user{user_id}.csv"
    recs.to_csv(path, index=False, encoding="utf-8")
    print(f"[Saved CSV] {path.resolve()}")


# =========================
# 8) メイン
# =========================
if __name__ == "__main__":
    df_all = load_data(CSV_PATH)
    df_single = single_skill_subset(df_all)

    if USE_WEIGHTED_EM:
        # ---- A案：explode＋1/m重み ----
        df_ex = prepare_exploded(df_all)
        params_by_skill = fit_per_skill_weighted(df_ex, max_iter=EM_MAX_ITERS,
                                                 min_eff_samples=MIN_EFFECTIVE_SAMPLES)

        # 最も解答数が多いユーザを例に
        sample_user = int(df_ex["user_id"].value_counts().idxmax())

        # ユーザ別スキルレポート
        rep_rows = []
        for sid in sorted(params_by_skill.keys()):
            p = params_by_skill[sid]
            sdf = df_ex[(df_ex["user_id"] == sample_user) & (df_ex["skill_id"] == sid)].sort_values("date")
            obs = sdf["result"].astype(int).tolist()
            pL, pL_next, p_corr = bkt_predict_next(obs, p)
            rep_rows.append({"user_id": sample_user, "skill_id": sid, "n_obs": len(obs),
                             "P(L)_posterior": pL, "P(L)_after_learn": pL_next,
                             "P(next correct)": p_corr, "learn": p["p_T"], "slip": p["p_S"], "guess": p["p_G"]})
        report = pd.DataFrame(rep_rows).sort_values("P(next correct)", ascending=False)
        print(f"\n[Weighted-EM] User {sample_user} skill report (top 10):")
        print(report.head(10).to_string(index=False))

        # 多スキルアイテムの正答確率例
        recent_item = int(df_all[df_all["user_id"] == sample_user].sort_values("date")["item_id"].iloc[-1])
        pred = predict_item_multiskill(sample_user, recent_item, df_all, df_ex, params_by_skill,
                                       combine_mode="and")
        print("\n[Multi-skill prediction] user={}, item={}, skills={}, p_correct={:.6f}"
              .format(pred["user_id"], pred["item_id"], pred["skills"],
                      pred["p_correct"] if pred["p_correct"] is not None else float('nan')))

        # データの概況
        print("Global correctness =", df_all["result"].mean())
        print("avg_skills_per_item =", df_all["skill"].astype(str).str.count(",").add(1).mean())
        print("Top-10 skills by effective samples:")
        print(df_ex.groupby("skill_id")["weight"].sum().sort_values(ascending=False).head(10))

        # 推薦（raw → 多様化 → クールダウン）
        raw = recommend_topK(sample_user, K=100, df_all=df_all, df_exploded=df_ex,
                             params_by_skill=params_by_skill,
                             combine_mode="soft_and", alpha=0.5,
                             target_band="auto", lam=0.3)
        top_div = diversify_by_skills(raw, topN=50, mmr_lambda=0.7)
        top_div_cd = add_cooldown_penalty(top_div, skill_last_seen, current_step,
                                          cooldown=5, max_penalty=0.2)
        topK = top_div_cd.head(10)

        print("\n[Top-K diversified]")
        print(topK[["item_id", "p_correct", "delta", "score"]].to_string(index=False))

        # 保存（多様化後の結果を保存）
        llm_payload = build_llm_payload(sample_user, topK)
        _ = save_payload_json(llm_payload, user_id=sample_user, out_dir="../json", prefix="recommendations")

        # ロード → 表示 → CSV保存 → 次の1問
        path, payload, recs = load_latest_recs("../json", user_id=sample_user)
        print(f"[Loaded] {path}")
        show_top(recs, topN=10)
        save_recs_csv(recs, user_id=payload["user_id"], out_dir="../json")

        next_item = int(recs.sort_values(["score", "expected_mastery_gain", "p_next_correct"],
                                         ascending=[False, False, False]).iloc[0]["item_id"])
        print("Next item to serve:", next_item)

        # B) クールダウン用“最終出題ステップ”更新
        update_skill_last_seen(skill_last_seen, next_item, current_step, df_all)
        current_step += 1

        # A) オンライン更新の初期化＆（例）更新呼び出し
        user_state = init_user_state(sample_user, df_ex, params_by_skill)
        # 例：採点結果 outcome=1/0 を得たら、即座に更新
        # user_state = bkt_update_state_for_item(user_state, item_id=next_item, outcome=1,
        #                                        params_by_skill=params_by_skill, df_all=df_all)

    else:
        # ---- 既存の単一スキルBKT ----
        params_by_skill = fit_per_skill_standard(df_single, max_iter=EM_MAX_ITERS)
        sample_user = 68
        report = predict_user_single(df_single, params_by_skill, user_id=sample_user)
        print(report.head(10).to_string(index=False))
