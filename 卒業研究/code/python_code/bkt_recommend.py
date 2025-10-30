#!/usr/bin/env python
# coding: utf-8
"""
BKT→IRT→(任意)LLM 再ランクまで一体の最小実装
- 前提: 各レコードは単一スキル（skill_id/skill）に紐づく
- 出力: これまでのBKTレポート形式を維持（コンソール表示）
- 追加: IRT(3PL)で候補問題を作成し、(任意)LLMで再ランク（JSON返し想定）
- 依存: pandas, numpy（LLMを使う場合のみ openai などを任意で導入）

使い方（例）:
  python bkt_recommend.py --csv ../csv/one_200_50_500_50000.csv --user_id 68 --nrows 50000 --k 5 --use_llm 0
"""

from __future__ import annotations
import argparse, json, math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ========== ユーティリティ ==========
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def logit(p: float, eps: float=1e-9) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def clip01(x: float) -> float:
    return float(np.clip(float(x), 1e-5, 1-1e-5))

# ========== データ読み込み（単一スキル前提） ==========
def load_df(path: str, nrows: Optional[int]=None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)

    # skill_id
    if 'skill_id' in df.columns:
        sk = df['skill_id']
    elif 'skill' in df.columns:
        sk = df['skill']
    else:
        raise ValueError("Missing 'skill' or 'skill_id' column.")
    df['skill_id'] = pd.to_numeric(sk, errors='coerce')

    # user_id
    if 'user_id' not in df.columns:
        raise ValueError("Missing 'user_id' column.")
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')

    # item_id（無ければ連番にする）
    if 'item_id' in df.columns:
        it = pd.to_numeric(df['item_id'], errors='coerce')
        df['item_id'] = it
    else:
        df['item_id'] = np.arange(len(df), dtype=np.int64)

    # result/correct/is_correct/outcome/y
    cand = [c for c in ['result','correct','is_correct','outcome','y'] if c in df.columns]
    if not cand:
        raise ValueError("Missing result column (result/correct/is_correct/outcome/y).")
    df['result'] = pd.to_numeric(df[cand[0]], errors='coerce').fillna(0).astype(int)
    df['result'] = (df['result'] > 0).astype(int)

    # date/timestamp/time/ts
    date_col = next((c for c in ['date','timestamp','time','ts'] if c in df.columns), None)
    if date_col is not None:
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        if df['date'].isna().any():
            df.loc[df['date'].isna(), 'date'] = pd.to_datetime(np.arange(df['date'].isna().sum()), unit='s')
    else:
        df['date'] = pd.to_datetime(np.arange(len(df)), unit='s')

    # a,b,c（無ければデフォルトを与える）
    if 'a' not in df.columns: df['a'] = 1.0
    if 'b' not in df.columns: df['b'] = 0.0
    if 'c' not in df.columns: df['c'] = 0.0

    # 整形
    df = df.dropna(subset=['user_id','skill_id']).copy()
    df['user_id']  = df['user_id'].astype(int)
    df['skill_id'] = df['skill_id'].astype(int)
    df = df.sort_values(['user_id','skill_id','date','item_id']).reset_index(drop=True)
    return df

# ========== BKT（忘却なし）EM ==========
def bkt_em(sequences: List[List[int]], max_iter: int = 30, tol: float = 1e-4,
           init: Optional[Tuple[float,float,float,float]] = None) -> Dict[str, float]:
    if init is None:
        p_L0, p_T, p_S, p_G = 0.2, 0.1, 0.1, 0.2
    else:
        p_L0, p_T, p_S, p_G = init

    last_ll = None
    it_used = 0
    for it in range(1, max_iter+1):
        A = np.array([[1-p_T, p_T],
                      [0.0,   1.0]])  # U->L

        sum_gamma_L1 = 0.0
        sum_xi_U2L   = 0.0
        sum_gamma_U  = 0.0
        sum_emit_L = sum_emit_L_incorrect = 0.0
        sum_emit_U = sum_emit_U_correct   = 0.0
        total_ll = 0.0

        for obs in sequences:
            T = len(obs)
            if T == 0:
                continue

            pi = np.array([1-p_L0, p_L0])

            def b(o: int) -> np.ndarray:
                return np.array([ p_G if o==1 else (1-p_G),
                                  (1-p_S) if o==1 else p_S ])

            # forward (scaled)
            alpha = np.zeros((T,2))
            scale = np.zeros(T)
            alpha[0] = pi * b(obs[0])
            scale[0] = alpha[0].sum() + 1e-300
            alpha[0] /= scale[0]
            for t in range(1, T):
                alpha[t] = (alpha[t-1].dot(A)) * b(obs[t])
                scale[t] = alpha[t].sum() + 1e-300
                alpha[t] /= scale[t]

            # backward (scaled)
            beta = np.zeros((T,2))
            beta[-1] = 1.0 / scale[-1]
            for t in range(T-2, -1, -1):
                bt1 = b(obs[t+1])
                beta[t] = (A * (bt1 * beta[t+1])).sum(axis=1)
                beta[t] /= scale[t]

            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)

            for t in range(T-1):
                bt1 = b(obs[t+1])
                num = alpha[t,0] * p_T * bt1[1] * beta[t+1,1]
                den = (alpha[t,0]*(1-p_T)*bt1[0]*beta[t+1,0] +
                       alpha[t,0]*p_T     *bt1[1]*beta[t+1,1] +
                       alpha[t,1]*1.0     *bt1[1]*beta[t+1,1])
                if den > 0:
                    sum_xi_U2L += num / den
                sum_gamma_U += gamma[t,0]

            sum_emit_L += gamma[:,1].sum()
            sum_emit_L_incorrect += ((1-np.array(obs)) * gamma[:,1]).sum()
            sum_emit_U += gamma[:,0].sum()
            sum_emit_U_correct   += ( np.array(obs)   * gamma[:,0]).sum()

            sum_gamma_L1 += gamma[0,1]
            total_ll += float(np.sum(np.log(scale + 1e-300)))

        nseq = max(len(sequences), 1)
        p_L0 = clip01(sum_gamma_L1 / nseq)
        p_T  = clip01(sum_xi_U2L   / max(sum_gamma_U, 1e-12))
        p_S  = clip01(sum_emit_L_incorrect / max(sum_emit_L, 1e-12))
        p_G  = clip01(sum_emit_U_correct   / max(sum_emit_U, 1e-12))

        it_used = it
        if last_ll is not None and abs(total_ll - last_ll) < tol:
            break
        last_ll = total_ll

    return {"p_L0": p_L0, "p_T": p_T, "p_S": p_S, "p_G": p_G,
            "loglik": last_ll if last_ll is not None else float('nan'),
            "iters": it_used}

def fit_per_skill(df: pd.DataFrame, max_iter: int=30) -> Dict[int, Dict[str, float]]:
    params: Dict[int, Dict[str, float]] = {}
    for sid, sdf in df.groupby("skill_id"):
        seqs = [udf["result"].astype(int).tolist() for _, udf in sdf.groupby("user_id")]
        if not seqs:
            continue
        params[int(sid)] = bkt_em(seqs, max_iter=max_iter)
    return params

def bkt_predict_next(obs: List[int], p: Dict[str, float]) -> Tuple[float,float,float]:
    pL = p["p_L0"]
    for o in obs:
        pL = pL + (1 - pL) * p["p_T"]   # pre-learn
        num = ((1 - p["p_S"]) if o == 1 else p["p_S"]) * pL
        den = num + ((p["p_G"]) if o == 1 else (1 - p["p_G"])) * (1 - pL)
        if den > 0:
            pL = num / den
    pL_next   = pL + (1 - pL) * p["p_T"]
    p_correct = (1 - p["p_S"]) * pL_next + p["p_G"] * (1 - pL_next)
    return pL, pL_next, p_correct

def predict_user(df: pd.DataFrame, params_by_skill: Dict[int, Dict[str, float]], user_id: int) -> pd.DataFrame:
    rows = []
    user_df = df[df["user_id"] == user_id]
    for sid, sdf in user_df.groupby("skill_id"):
        obs = sdf.sort_values("date")["result"].astype(int).tolist()
        p = params_by_skill.get(int(sid))
        if not p:
            continue
        pL, pL_next, p_corr = bkt_predict_next(obs, p)
        rows.append({
            "user_id": int(user_id),
            "skill_id": int(sid),
            "n_obs": len(obs),
            "P(L)_posterior": pL,
            "P(L)_after_learn": pL_next,
            "P(next correct)": p_corr,
            "learn":  p["p_T"],
            "slip":   p["p_S"],
            "guess":  p["p_G"],
        })
    out = pd.DataFrame(rows).sort_values("P(next correct)", ascending=False)
    return out

# ========== IRT(3PL)：θ推定（ユーザ×スキル） ==========
def three_pl_prob(theta: float, a: float, b: float, c: float) -> float:
    return c + (1 - c) * sigmoid(a * (theta - b))

def three_pl_dPdtheta(theta: float, a: float, b: float, c: float) -> float:
    z = a * (theta - b)
    s = sigmoid(z)
    return (1 - c) * a * s * (1 - s)

def estimate_theta_3pl(item_params: pd.DataFrame, y: np.ndarray,
                       max_iter: int=25, tol: float=1e-4) -> float:
    """
    item_params: DataFrame with columns a,b,c (each row is an item answered by the user in a skill)
    y: 0/1 outcomes aligned with item_params
    Newton-Raphson with Fisher情報量近似: θ_{t+1} = θ_t + grad / I
    """
    theta = 0.0
    for _ in range(max_iter):
        P = three_pl_prob(theta, item_params['a'].values, item_params['b'].values, item_params['c'].values)
        d = three_pl_dPdtheta(theta, item_params['a'].values, item_params['b'].values, item_params['c'].values)

        # 端に寄らないようクリップ
        P = np.clip(P, 1e-6, 1 - 1e-6)

        # 勾配とFisher情報量
        grad = np.sum((y - P) * d / (P * (1 - P)))
        I    = np.sum((d ** 2) / (P * (1 - P))) + 1e-9

        step = grad / I
        theta_new = theta + step
        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    return float(theta)

def estimate_theta_by_skill(df: pd.DataFrame, user_id: int) -> Dict[int, float]:
    thetas: Dict[int, float] = {}
    u = df[df['user_id']==user_id]
    for sid, sdf in u.groupby('skill_id'):
        # ユーザが既に解いた項目に基づいて θ を推定
        params = sdf[['a','b','c']].copy()
        y = sdf['result'].astype(int).values
        if len(sdf) >= 3:  # 最低3観測程度
            thetas[int(sid)] = estimate_theta_3pl(params, y)
        else:
            thetas[int(sid)] = 0.0  # データ不足なら0で初期化
    return thetas

# ========== 候補生成（IRTでフィルタ＆スコア） ==========
def item_info_approx(a: float, b: float, c: float, theta: float, P: Optional[float]=None) -> float:
    if P is None:
        P = three_pl_prob(theta, a, b, c)
    # 3PL 近似の情報量（2PLの a^2 P(1-P) を拡張した簡易形）
    return (a*a) * ((P - c)**2) / (((1 - c)**2) * P * (1 - P) + 1e-12)

def build_candidates(df: pd.DataFrame, user_id: int, theta_by_skill: Dict[int, float],
                     pL_by_skill: Dict[int, float],
                     p_range=(0.55, 0.75), recent_window: int=20, topK_pool: int=50) -> List[Dict]:
    """
    - ユーザが未出題の item を優先。未出題が無ければ最近出題以外を使用
    - P(next correct) を 0.55-0.75 周辺に寄せる
    - 情報量で降順ソートして topK_pool 件返す
    """
    u = df[df['user_id']==user_id].sort_values('date')
    seen_items = set(u['item_id'].tolist())
    recent_items = set(u.tail(recent_window)['item_id'].tolist())

    # アイテム全集合（同一スキル内の全 item_id）
    item_bank = df[['item_id','skill_id','a','b','c']].drop_duplicates()

    cands: List[Dict] = []
    for sid, theta in theta_by_skill.items():
        # そのスキルの全アイテム
        sitems = item_bank[item_bank['skill_id']==sid]
        for _, row in sitems.iterrows():
            iid, a, b, c = int(row['item_id']), float(row['a']), float(row['b']), float(row['c'])
            P = three_pl_prob(theta, a, b, c)
            info = item_info_approx(a, b, c, theta, P=P)

            prereq_ok = True  # 本最小実装では常にTrue
            if p_range[0] <= P <= p_range[1] and prereq_ok:
                unseen = (iid not in seen_items)
                seen_recently = (iid in recent_items)
                # 未出題を強く優先。未出題が無ければrecent以外
                score_bias = (2.0 if unseen else (1.0 if not seen_recently else 0.5))
                cands.append({
                    "item_id": iid,
                    "skill_id": sid,
                    "a": a, "b": b, "c": c,
                    "p_next_correct": float(P),
                    "info": float(info * score_bias),
                    "seen_recently": bool(seen_recently),
                    "prereq_ok": prereq_ok,
                    "unseen": unseen
                })

    # もしゼロならP帯域を広げて再試行
    if not cands:
        lo, hi = 0.45, 0.85
        for sid, theta in theta_by_skill.items():
            sitems = item_bank[item_bank['skill_id']==sid]
            for _, row in sitems.iterrows():
                iid, a, b, c = int(row['item_id']), float(row['a']), float(row['b']), float(row['c'])
                P = three_pl_prob(theta, a, b, c)
                if lo <= P <= hi:
                    info = item_info_approx(a, b, c, theta, P=P)
                    prereq_ok = True
                    unseen = (iid not in seen_items)
                    seen_recently = (iid in recent_items)
                    score_bias = (2.0 if unseen else (1.0 if not seen_recently else 0.5))
                    cands.append({
                        "item_id": iid, "skill_id": sid, "a": a, "b": b, "c": c,
                        "p_next_correct": float(P),
                        "info": float(info * score_bias),
                        "seen_recently": bool(seen_recently),
                        "prereq_ok": prereq_ok,
                        "unseen": unseen
                    })

    # 情報量（バイアス込み）で降順、未出題を優先
    cands.sort(key=lambda x: (not x["unseen"], -x["info"]))  # unseen(True)を先に
    return cands[:topK_pool]

# ========== LLM再ランク（任意） ==========
def llm_rerank(candidates: List[Dict], user_id: int, theta_by_skill: Dict[int,float],
               pL_by_skill: Dict[int,float], top_k: int=5, use_llm: bool=False) -> List[Dict]:
    """
    - use_llm=False: パススルー（info降順&未出題優先の先頭からTop-k）
    - use_llm=True : プロンプトを構成してLLMにJSONのみを要求（環境に合わせて実装）
      ※ この最小実装では外部呼び出しは行わず、同順位付けで返す
    """
    slate = candidates[:]
    # （ここに本番のLLM呼び出しを実装する）
    # 例：OpenAIを使う場合は prompt/candidates をJSONで渡し、JSONだけ返すよう指示
    return [{"item_id": c["item_id"],
             "skill_id": c["skill_id"],
             "reason": f"skill {c['skill_id']} / P≈{c['p_next_correct']:.2f}, info高め, {'未出題' if c['unseen'] else '再出題'}"}
            for c in slate[:top_k]]

# ========== メイン ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="入力CSV（単一スキル前提）")
    ap.add_argument("--user_id", type=int, required=True)
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--nrows", type=int, default=None, help="先頭から読む行数を制限（例: 50000）")
    ap.add_argument("--k", type=int, default=5, help="最終推薦件数")
    ap.add_argument("--use_llm", type=int, default=0, help="1でLLM再ランク（この最小実装では同順位）")
    args = ap.parse_args()

    # 1) 読み込み
    df = load_df(args.csv, nrows=args.nrows)
    # 2) BKT学習
    params_by_skill = fit_per_skill(df, max_iter=args.max_iter)

    # 3) BKTユーザレポート（元の出力形式）
    report = predict_user(df, params_by_skill, user_id=args.user_id)
    print(f"[Weighted-EM] User {args.user_id} skill report (top 10):")
    if not report.empty:
        # 列順を元に寄せる
        cols = ["user_id","skill_id","n_obs","P(L)_posterior","P(L)_after_learn","P(next correct)",
                "learn","slip","guess"]
        cols = [c for c in cols if c in report.columns]
        print(report[cols].head(10).to_string(index=False))
    else:
        print("(no records)")

    # 4) θ（IRT, 3PL）推定 → 候補生成
    theta_by_skill = estimate_theta_by_skill(df, user_id=args.user_id)
    # pL_by_skill は将来の拡張用（ここでは未使用だがインターフェイスに残す）
    pL_by_skill = {int(r.skill_id): float(r["P(L)_posterior"]) for _, r in report.iterrows()}

    candidates = build_candidates(df, args.user_id, theta_by_skill, pL_by_skill,
                                  p_range=(0.55,0.75), recent_window=20, topK_pool=50)

    # 5) 候補の上位を表示（人間確認用）
    if candidates:
        print("\n[Candidate items by IRT (top 10)]:")
        df_c = pd.DataFrame(candidates).sort_values(["unseen","info"], ascending=[False, False]).head(10)
        cols = ["item_id","skill_id","p_next_correct","info","unseen","seen_recently"]
        print(df_c[cols].to_string(index=False))
    else:
        print("\n[Candidate items by IRT]: (no candidates)")

    # 6) （任意）LLM再ランク → 最終推薦
    final = llm_rerank(candidates, args.user_id, theta_by_skill, pL_by_skill,
                       top_k=args.k, use_llm=bool(args.use_llm))
    if final:
        print(f"\n[LLM Top-{args.k}]:")
        for i, rec in enumerate(final, 1):
            print(f"{i}. item_id={rec['item_id']} (skill {rec['skill_id']}): {rec['reason']}")
    else:
        print("\n[LLM Top-k]: (no selection)")

if __name__ == "__main__":
    main()
