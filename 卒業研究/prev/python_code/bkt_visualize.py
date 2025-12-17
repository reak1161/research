# bkt_visualize.py
# ------------------------------------------------------------
# 単一スキルBKTの推定遷移を、学習者×スキルごとに可視化する後付けツール
# 既存コードの大改造不要：生の正誤ログ + スキル別BKTパラメータから再計算して描画
# 必須: pandas, matplotlib
# ------------------------------------------------------------

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ====== BKT 基本式 ======

def bkt_predict_correct(p_mastery: float, slip: float, guess: float) -> float:
    """P(X=1 | L_t) = L_t*(1-S) + (1-L_t)*G"""
    return p_mastery * (1.0 - slip) + (1.0 - p_mastery) * guess


def bkt_posterior(p_mastery: float, correct: int, slip: float, guess: float) -> float:
    """観測C∈{0,1}後の事後 P(L_t | C)"""
    if correct not in (0, 1):
        raise ValueError("correct must be 0 or 1")

    if correct == 1:
        num = p_mastery * (1.0 - slip)
        den = p_mastery * (1.0 - slip) + (1.0 - p_mastery) * guess
    else:
        num = p_mastery * slip
        den = p_mastery * slip + (1.0 - p_mastery) * (1.0 - guess)

    if den == 0.0:
        # 数値安定対策（極端パラメータ時）
        return 0.0
    return num / den


def bkt_learn_step(p_posterior: float, learn: float) -> float:
    """学習遷移 P(L_{t+1}) = P(L_t|C) + (1 - P(L_t|C))*T"""
    return p_posterior + (1.0 - p_posterior) * learn


# ====== データ仕様 ======
# interactions: 必須列 -> user_id, skill_id, correct(0/1)
#               任意列 -> t（整数の時系列），または timestamp/datetime
# params:       必須列 -> skill_id, p_init, learn, slip, guess


@dataclass(frozen=True)
class BKTParams:
    p_init: float
    learn: float
    slip: float
    guess: float


def _normalize_time(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    """時系列カラムが無ければ、user_id×skill_idごとに入力順で0,1,2...を付与"""
    df = df.copy()
    if time_col and time_col in df.columns:
        # 既存の時系列でソート
        df = df.sort_values(by=["user_id", "skill_id", time_col, df.index.name or df.index])
    else:
        # 自動付与
        df = df.sort_values(by=["user_id", "skill_id", df.index.name or df.index])
        df["__auto_t__"] = (
            df.groupby(["user_id", "skill_id"]).cumcount()
        )
        time_col = "__auto_t__"
    df["__t__"] = df[time_col].values
    return df


def compute_bkt_log(
    interactions: pd.DataFrame,
    skill_params: pd.DataFrame,
    time_col: Optional[str] = None,
    correct_col: str = "correct",
) -> pd.DataFrame:
    """
    生の正誤ログとスキル別パラメータから、推定ごとの指標を再計算して返す。

    戻り値の列:
      user_id, skill_id, idx(=t), correct,
      p_mastery_prior,   # 事前 P(L_t)
      p_posterior,       # 観測後 P(L_t | C)
      p_mastery_after,   # 学習遷移後 P(L_{t+1})
      p_next_correct     # 観測前の P(X=1)（=次回正答確率の事前予測）
    """
    req_int_cols = {"user_id", "skill_id", correct_col}
    missing = req_int_cols - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions に不足列: {missing}")

    req_param_cols = {"skill_id", "p_init", "learn", "slip", "guess"}
    missing_p = req_param_cols - set(skill_params.columns)
    if missing_p:
        raise ValueError(f"skill_params に不足列: {missing_p}")

    df = _normalize_time(interactions, time_col=time_col)
    params_map: dict[int, BKTParams] = {
        int(r.skill_id): BKTParams(float(r.p_init), float(r.learn), float(r.slip), float(r.guess))
        for _, r in skill_params.iterrows()
    }

    logs = []
    for (uid, sid), g in df.groupby(["user_id", "skill_id"], sort=True):
        sid_int = int(sid)
        if sid_int not in params_map:
            raise KeyError(f"skill_id={sid_int} のパラメータが見つかりません")

        p = params_map[sid_int]
        p_mastery = float(p.p_init)

        g_sorted = g.sort_values(by="__t__")
        for _, row in g_sorted.iterrows():
            c = int(row[correct_col])

            p_next = bkt_predict_correct(p_mastery, p.slip, p.guess)      # 観測前の正答確率
            p_post = bkt_posterior(p_mastery, c, p.slip, p.guess)         # 観測後の事後
            p_after = bkt_learn_step(p_post, p.learn)                     # 学習後（次の事前）

            logs.append({
                "user_id": int(uid),
                "skill_id": sid_int,
                "idx": int(row["__t__"]),
                "correct": c,
                "p_mastery_prior": p_mastery,
                "p_posterior": p_post,
                "p_mastery_after": p_after,
                "p_next_correct": p_next,
            })

            p_mastery = p_after  # 次ステップの事前へ更新

    out = pd.DataFrame(logs).sort_values(by=["user_id", "skill_id", "idx"]).reset_index(drop=True)
    return out


def _nice_title(uid: int, sid: int) -> str:
    return f"User {uid} × Skill {sid} — BKT推定の遷移"


def plot_bkt_progress(
    log_df: pd.DataFrame,
    user_ids: Optional[Sequence[int]] = None,
    skill_ids: Optional[Sequence[int]] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> List[Tuple[int, int]]:
    """
    指定した学習者×スキルごとに、以下の2系列を同一軸(0〜1)で重ね描き:
      - p_mastery_after（学習後の「最新の習得度」）
      - p_next_correct（観測前の「次回正答確率」）
    さらに、正誤(0/1)を下端にヒゲで並べ視覚補助。

    戻り値: 描画した (user_id, skill_id) のリスト
    """
    req_cols = {"user_id", "skill_id", "idx", "correct", "p_mastery_after", "p_next_correct"}
    if not req_cols.issubset(set(log_df.columns)):
        raise ValueError(f"log_df に必要な列が足りません: {req_cols - set(log_df.columns)}")

    # フィルタ
    _df = log_df.copy()
    if user_ids:
        _df = _df[_df["user_id"].isin([int(u) for u in user_ids])]
    if skill_ids:
        _df = _df[_df["skill_id"].isin([int(s) for s in skill_ids])]

    plotted: List[Tuple[int, int]] = []
    for (uid, sid), g in _df.groupby(["user_id", "skill_id"], sort=True):
        g = g.sort_values(by="idx")
        x = g["idx"].values.astype(int)
        y_mastery = g["p_mastery_after"].values.astype(float)
        y_next = g["p_next_correct"].values.astype(float)
        corr = g["correct"].values.astype(int)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y_mastery, label="習得度 P(L)（学習後）", linewidth=2)
        ax.plot(x, y_next, label="次回正答確率 P(X=1)", linewidth=2, linestyle="--")

        # 正誤のヒゲ（0→短い/1→やや長い）を下端に
        y0 = np.zeros_like(x, dtype=float)
        y1 = np.where(corr == 1, 0.07, 0.03)
        for xi, yi in zip(x, y1):
            ax.vlines(xi, 0.0, yi, linewidth=2, alpha=0.7)

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_xlabel("Attempt index (t)")
        ax.set_ylabel("Probability")
        ax.grid(True, linestyle=":")
        ax.legend(loc="lower right")
        ax.set_title(_nice_title(int(uid), int(sid)))

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            fn = os.path.join(save_dir, f"user{uid}_skill{sid}.png")
            plt.savefig(fn, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        plotted.append((int(uid), int(sid)))

    if not plotted:
        print("（該当データがありません。user_ids / skill_ids の指定を見直してください）")
    return plotted


# ====== CLI ======

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BKT可視化：学習者×スキルの推定遷移をプロット")
    p.add_argument("--data", type=str, required=True, help="正誤ログCSV（user_id,skill_id,correct,…）")
    p.add_argument("--params", type=str, required=True, help="BKTパラメータCSV（skill_id,p_init,learn,slip,guess）")
    p.add_argument("--users", type=int, nargs="*", default=None, help="表示する user_id（スペース区切りで複数）")
    p.add_argument("--skills", type=int, nargs="*", default=None, help="表示する skill_id（スペース区切りで複数）")
    p.add_argument("--time-col", type=str, default=None, help="時系列列名（例: t, timestamp）無指定なら自動採番")
    p.add_argument("--correct-col", type=str, default="correct", help="正誤列名（0/1）")
    p.add_argument("--save-dir", type=str, default=None, help="画像の保存先ディレクトリ")
    p.add_argument("--no-show", action="store_true", help="画面表示しない（保存のみ）")
    return p.parse_args()


def _main_cli():
    args = _parse_args()
    df = pd.read_csv(args.data)
    params = pd.read_csv(args.params)
    log_df = compute_bkt_log(
        df, params, time_col=args.time_col, correct_col=args.correct_col
    )
    plot_bkt_progress(
        log_df,
        user_ids=args.users,
        skill_ids=args.skills,
        save_dir=args.save_dir,
        show=not args.no_show
    )


if __name__ == "__main__":
    _main_cli()
