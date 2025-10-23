import pandas as pd, numpy as np
from typing import List, Dict, Any, Tuple

# =========================
# 0) 設定
# =========================
CSV_PATH = "../csv/200_50_500_100000.csv"

# A案（explode＋1/m重み）で学習するか？
USE_WEIGHTED_EM = True

# 学習対象とするスキルの「有効サンプル数」(重み合計)の下限
MIN_EFFECTIVE_SAMPLES = 200

# EMの反復回数
EM_MAX_ITERS = 30

# 多スキル問題の推論で Noisy-AND を使う（平均ではなく積）
USE_NOISY_AND_FOR_PRED = True  # True: Noisy-AND, False: per-skill平均


# ===== MAP-EM（Beta事前） =====
# 期待値は slip/guess/prior ≈ 0.2、learn ≈ 0.03 あたりに寄せる
BETA_PRIOR = {
    "L0": (2.0, 8.0),   # mean 0.20
    "T":  (3.0, 97.0),  # mean 0.03 （1問あたり学習）
    "S":  (2.0, 8.0),   # mean 0.20
    "G":  (2.0, 8.0),   # mean 0.20
}

# ===== 物理クランプ（安全網） =====
CLAMP = {
    "L0": (0.01, 0.90),
    "T":  (0.001, 0.30),
    "S":  (0.01, 0.40),
    "G":  (0.01, 0.40),
}

# ===== マルチスタートの初期値候補 =====
INIT_GRID = [
    (0.2, 0.02, 0.10, 0.20),
    (0.3, 0.05, 0.15, 0.15),
    (0.1, 0.03, 0.05, 0.25),
]



# =========================
# 1) データ読み込み
# =========================
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])

# ----- 単一スキル subset（既存の流れを保持）
df_single = df[df["skill"].astype(str).str.count(",")==0].copy()
df_single["skill_id"] = df_single["skill"].astype(int)
df_single = df_single.sort_values(["user_id","skill_id","date"])

# データ読み込み直後に追加（診断用）
FLIP_LABELS = False  # Trueにして再学習して比較
if FLIP_LABELS:
    df["result"] = 1 - df["result"].astype(int)


# =========================
# 2) 既存：BKT（忘却なし）EM（単一スキル）
# =========================
def bkt_em(sequences: List[List[int]], max_iter=30, tol=1e-4, init=None) -> Dict[str, Any]:
    # sequences: ユーザごとの {0/1} 観測系列のリスト（同一スキル）
    if init is None:
        p_L0, p_T, p_S, p_G = 0.2, 0.1, 0.1, 0.2
    else:
        p_L0, p_T, p_S, p_G = init
    def clip(x): return float(np.clip(x, 1e-5, 1-1e-5))
    last_ll = None
    for it in range(max_iter):
        A = np.array([[1-p_T, p_T],[0.0, 1.0]])  # U->Lのみ、L->Uなし
        # 期待値カウント
        sum_gamma_L1=sum_xi_U2L=sum_gamma_U=0.0
        sum_emit_L=sum_emit_L_incorrect=sum_emit_U=sum_emit_U_correct=0.0
        total_ll = 0.0
        for obs in sequences:
            T=len(obs)
            if T==0: 
                continue
            pi=np.array([1-p_L0, p_L0])
            # Forward-Backward（スケーリング付き）
            alpha=np.zeros((T,2)); scale=np.zeros(T)
            b=lambda o: np.array([ p_G if o==1 else (1-p_G),
                                   (1-p_S) if o==1 else p_S ])
            alpha[0]=pi*b(obs[0]); scale[0]=alpha[0].sum()+1e-300; alpha[0]/=scale[0]
            for t in range(1,T):
                alpha[t]=(alpha[t-1].dot(A))*b(obs[t])
                scale[t]=alpha[t].sum()+1e-300; alpha[t]/=scale[t]
            beta=np.zeros((T,2)); beta[-1]=1.0/scale[-1]
            for t in range(T-2,-1,-1):
                bt1=b(obs[t+1]); beta[t]=(A*(bt1*beta[t+1])).sum(axis=1); beta[t]/=scale[t]
            gamma=alpha*beta; gamma/=gamma.sum(axis=1, keepdims=True)
            # 遷移U->Lの期待回数
            for t in range(T-1):
                bt1=b(obs[t+1])
                num = alpha[t,0]*p_T*bt1[1]*beta[t+1,1]
                den = (alpha[t,0]*(1-p_T)*bt1[0]*beta[t+1,0] +
                       alpha[t,0]*p_T*bt1[1]*beta[t+1,1] +
                       alpha[t,1]*1.0*bt1[1]*beta[t+1,1])
                sum_xi_U2L += 0.0 if den<=0 else num/den
                sum_gamma_U += gamma[t,0]
            # 尤度用カウント
            sum_emit_L += gamma[:,1].sum()
            sum_emit_L_incorrect += ((1-np.array(obs))*gamma[:,1]).sum()
            sum_emit_U += gamma[:,0].sum()
            sum_emit_U_correct += (np.array(obs)*gamma[:,0]).sum()
            sum_gamma_L1 += gamma[0,1]
            total_ll += np.sum(np.log(scale+1e-300))
        # M-step
        nseq = max(len(sequences),1)
        p_L0 = clip(sum_gamma_L1 / nseq)
        p_T  = clip(sum_xi_U2L / max(sum_gamma_U,1e-12))
        p_S  = clip(sum_emit_L_incorrect / max(sum_emit_L,1e-12))
        p_G  = clip(sum_emit_U_correct / max(sum_emit_U,1e-12))
        if last_ll is not None and abs(total_ll-last_ll)<tol: break
        last_ll = total_ll
    return {"p_L0":p_L0,"p_T":p_T,"p_S":p_S,"p_G":p_G,"loglik":last_ll,"iters":it+1}


def fit_per_skill(df_single: pd.DataFrame, max_iter=30) -> Dict[int, Dict[str,float]]:
    params={}
    for sid, sdf in df_single.groupby("skill_id"):
        seqs=[udf["result"].astype(int).tolist() for _,udf in sdf.groupby("user_id")]
        params[sid]=bkt_em(seqs, max_iter=max_iter)
    return params


# =========================
# 3) A案：explode＋1/m重み での学習（新規）
# =========================
def prepare_exploded(df: pd.DataFrame) -> pd.DataFrame:
    """skill='a,b,c' を行分解し、重み=1/m を付与"""
    skl = df["skill"].astype(str).str.split(",")
    m = skl.str.len()
    tmp = df.copy()
    tmp["skills_list"] = skl
    tmp["m"] = m
    ex = tmp.explode("skills_list", ignore_index=True)
    ex["skill_id"] = ex["skills_list"].astype(int)
    ex["weight"] = 1.0 / ex["m"].astype(float)
    ex = ex[["user_id","item_id","skill_id","result","date","weight"]].sort_values(["user_id","skill_id","date"])
    return ex

def fit_bkt_weighted_for_skill(df_skill: pd.DataFrame, max_iter=30, tol=1e-4, init=None) -> Dict[str,Any]:
    """
    重み付きEM（忘却なし）＋ MAP正則化（Beta事前）＋ クランプ ＋ マルチスタート
    """
    def clip01(x): return float(np.clip(x, 1e-5, 1-1e-5))

    # ユーザ別系列（obs, wts）
    seqs: List[Tuple[np.ndarray,np.ndarray]] = []
    for _, udf in df_skill.groupby("user_id", sort=False):
        obs = udf["result"].astype(int).to_numpy()
        wts = udf["weight"].astype(float).to_numpy()
        if len(obs)>0:
            seqs.append((obs, wts))

    def run_em_once(init_tuple):
        p_L0, p_T, p_S, p_G = init_tuple
        last_ll = None
        for it in range(max_iter):
            A = np.array([[1-p_T, p_T],[0.0, 1.0]], dtype=float)

            sum_gamma_L1_w = 0.0
            sum_first_w    = 0.0
            sum_gamma_U_w  = 0.0
            sum_xi_U2L_w   = 0.0
            sum_emit_L_w   = 0.0
            sum_emit_L_incorrect_w = 0.0
            sum_emit_U_w   = 0.0
            sum_emit_U_correct_w   = 0.0
            total_ll = 0.0

            B1 = np.array([p_G, 1-p_S], dtype=float)   # o=1
            B0 = np.array([1-p_G, p_S], dtype=float)   # o=0

            for obs, wts in seqs:
                T = len(obs)
                if T==0: 
                    continue
                pi = np.array([1-p_L0, p_L0], dtype=float)

                # Forward
                alpha = np.zeros((T,2), dtype=float)
                scale = np.zeros(T, dtype=float)
                alpha[0] = pi * (B1 if obs[0]==1 else B0)
                s = alpha[0].sum() + 1e-300
                scale[0] = s; alpha[0] /= s
                for t in range(1, T):
                    b = B1 if obs[t]==1 else B0
                    alpha[t] = (alpha[t-1].dot(A)) * b
                    s = alpha[t].sum() + 1e-300
                    scale[t] = s; alpha[t] /= s

                # Backward
                beta = np.zeros((T,2), dtype=float)
                beta[-1] = 1.0/scale[-1]
                for t in range(T-2, -1, -1):
                    b = B1 if obs[t+1]==1 else B0
                    tmp0 = (1-p_T)*b[0]*beta[t+1,0] + p_T*b[1]*beta[t+1,1]
                    tmp1 = 1.0*b[1]*beta[t+1,1]
                    beta[t] = np.array([tmp0, tmp1]) / scale[t]

                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)

                # 期待カウント（重み付き）
                for t in range(T-1):
                    b = B1 if obs[t+1]==1 else B0
                    num = alpha[t,0]*p_T*b[1]*beta[t+1,1]
                    den = (alpha[t,0]*(1-p_T)*b[0]*beta[t+1,0] +
                           alpha[t,0]*p_T*b[1]*beta[t+1,1] +
                           alpha[t,1]*1.0*b[1]*beta[t+1,1])
                    if den>0:
                        sum_xi_U2L_w += (num/den) * wts[t]
                    sum_gamma_U_w  += gamma[t,0] * wts[t]

                w_arr = wts; o_arr = obs.astype(float)
                sum_emit_L_w += (gamma[:,1] * w_arr).sum()
                sum_emit_L_incorrect_w += ((1 - o_arr) * gamma[:,1] * w_arr).sum()
                sum_emit_U_w += (gamma[:,0] * w_arr).sum()
                sum_emit_U_correct_w += (o_arr * gamma[:,0] * w_arr).sum()

                sum_gamma_L1_w += gamma[0,1] * wts[0]
                sum_first_w    += wts[0]

                total_ll += np.log(scale + 1e-300).sum()

            # ===== M-step: MAP（Beta事前）＋クランプ =====
            a_L0,b_L0 = BETA_PRIOR["L0"]
            a_T, b_T  = BETA_PRIOR["T"]
            a_S, b_S  = BETA_PRIOR["S"]
            a_G, b_G  = BETA_PRIOR["G"]

            p_L0 = (sum_gamma_L1_w + (a_L0 - 1)) / (max(sum_first_w,1e-12) + (a_L0 + b_L0 - 2))
            p_T  = (sum_xi_U2L_w  + (a_T  - 1)) / (max(sum_gamma_U_w,  1e-12) + (a_T  + b_T  - 2))
            p_S  = (sum_emit_L_incorrect_w + (a_S - 1)) / (max(sum_emit_L_w, 1e-12) + (a_S + b_S - 2))
            p_G  = (sum_emit_U_correct_w   + (a_G - 1)) / (max(sum_emit_U_w, 1e-12) + (a_G + b_G - 2))

            # 物理クランプ
            p_L0 = float(np.clip(p_L0, *CLAMP["L0"]))
            p_T  = float(np.clip(p_T,  *CLAMP["T"]))
            p_S  = float(np.clip(p_S,  *CLAMP["S"]))
            p_G  = float(np.clip(p_G,  *CLAMP["G"]))

            if last_ll is not None and abs(total_ll - last_ll) < tol:
                return {"p_L0": p_L0, "p_T": p_T, "p_S": p_S, "p_G": p_G,
                        "loglik": total_ll, "iters": it+1}
            last_ll = total_ll

        return {"p_L0": p_L0, "p_T": p_T, "p_S": p_S, "p_G": p_G,
                "loglik": last_ll, "iters": it+1}

    # ==== マルチスタート ====
    inits = [init] if init is not None else INIT_GRID
    best = None
    for ini in inits:
        cand = run_em_once(ini)
        if (best is None) or (cand["loglik"] is not None and cand["loglik"] > best["loglik"]):
            best = cand
    return best


def fit_per_skill_weighted(df_exploded: pd.DataFrame, max_iter=30, min_eff_samples=200) -> Dict[int, Dict[str,float]]:
    """重み合計が閾値以上のスキルだけ学習"""
    eff = df_exploded.groupby("skill_id")["weight"].sum()
    params={}
    for sid in eff.index:
        if eff.loc[sid] < min_eff_samples:
            continue
        sdf = df_exploded[df_exploded["skill_id"]==sid].sort_values(["user_id","date"])
        params[sid] = fit_bkt_weighted_for_skill(sdf, max_iter=max_iter)
    return params


# =========================
# 4) 推論：P(L) と 次の正答確率（単一スキル）
# =========================
def bkt_predict_next(obs: List[int], p: Dict[str,float]) -> Tuple[float,float,float]:
    pL=p["p_L0"]
    for o in obs:
        pL = pL + (1-pL)*p["p_T"]  # 事前学習
        num = ((1-p["p_S"]) if o==1 else p["p_S"]) * pL
        den = num + ((p["p_G"]) if o==1 else (1-p["p_G"]))*(1-pL)
        pL = pL if den<=0 else num/den
    pL_next = pL + (1-pL)*p["p_T"]
    p_correct = (1-p["p_S"])*pL_next + p["p_G"]*(1-pL_next)
    return pL, pL_next, p_correct


def predict_user_single(df_single: pd.DataFrame, params_by_skill: Dict[int,Dict[str,float]], user_id: int) -> pd.DataFrame:
    rows=[]
    for sid, sdf in df_single[df_single["user_id"]==user_id].groupby("skill_id"):
        obs=sdf.sort_values("date")["result"].astype(int).tolist()
        p=params_by_skill.get(sid)
        if not p: 
            continue
        pL, pL_next, p_corr = bkt_predict_next(obs, p)
        rows.append({"user_id":user_id,"skill_id":sid,"n_obs":len(obs),
                     "P(L)_posterior":pL,"P(L)_after_learn":pL_next,"P(next correct)":p_corr,
                     "learn":p["p_T"],"slip":p["p_S"],"guess":p["p_G"]})
    out=pd.DataFrame(rows).sort_values("P(next correct)", ascending=False)
    return out


# =========================
# 5) 多スキル問題の推論（Noisy-AND/平均）
# =========================

# --- 多スキル T_k の結合 ---
def combine_Tks(Tks, mode="and", alpha=1.0):
    """
    mode: "and" (=積), "geom" (=幾何平均), "logit_avg" (=ロジット平均),
          "soft_and" (=∏ T_k^α の 1/α次根; α∈(0,1]でANDを緩める)
    """
    Tks = [float(max(1e-9, min(1-1e-9, t))) for t in Tks]
    if not Tks: return None
    if mode == "and":
        p = 1.0
        for t in Tks: p *= t
        return p
    elif mode == "geom":
        prod = 1.0
        for t in Tks: prod *= t
        return prod ** (1.0/len(Tks))
    elif mode == "logit_avg":
        import math
        logits = [math.log(t/(1-t)) for t in Tks]
        avg = sum(logits)/len(logits)
        return 1/(1+math.exp(-avg))
    elif mode == "soft_and":
        # ∏ T_k^α の 1/α次根（α<1で積の“潰し”を緩める）
        prod = 1.0
        for t in Tks: prod *= (t**alpha)
        return prod ** (1.0/alpha/len(Tks))  # 平均化も入れておく
    else:
        raise ValueError("unknown combine mode")


def predict_item_multiskill(user_id: int, item_id: int,
                            df_history_exploded: pd.DataFrame,
                            params_by_skill: Dict[int,Dict[str,float]],
                            combine_mode: str = "and",
                            use_noisy_and: bool | None = None):
    """
    combine_mode: "and" / "geom" / "logit_avg"
    use_noisy_and: 後方互換用（True→"and", False→"geom"）
    """
    # 後方互換マッピング
    if use_noisy_and is not None:
        combine_mode = "and" if use_noisy_and else "geom"

    skills_str = df[df["item_id"]==item_id]["skill"].astype(str).iloc[0]
    skill_ids = [int(s) for s in skills_str.split(",")]

    per_skill = []
    for sid in skill_ids:
        if sid not in params_by_skill:
            continue
        p = params_by_skill[sid]
        sdf = df_history_exploded[(df_history_exploded["user_id"]==user_id) &
                                  (df_history_exploded["skill_id"]==sid)].sort_values("date")
        obs = sdf["result"].astype(int).tolist()
        pL, pL_next, _ = bkt_predict_next(obs, p)
        T_k = (1-p["p_S"])*pL_next + p["p_G"]*(1-pL_next)
        per_skill.append({"skill_id": sid, "P(L)": pL, "P(L^+)": pL_next, "T_k": T_k,
                          "learn":p["p_T"], "slip":p["p_S"], "guess":p["p_G"]})

    if len(per_skill)==0:
        return {"user_id": user_id, "item_id": item_id, "skills": skill_ids,
                "p_correct": None, "per_skill": []}

    p_correct = combine_Tks([r["T_k"] for r in per_skill], mode=combine_mode)
    return {"user_id": user_id, "item_id": item_id, "skills": skill_ids,
            "p_correct": p_correct, "per_skill": per_skill}




# 期待習熟上昇
def expected_gain_for_item(user_id: int, item_id: int,
                           df_history_exploded: pd.DataFrame,
                           params_by_skill: Dict[int,Dict[str,float]],
                           combine_mode: str = "geom", alpha: float = 0.5):
    # アイテムのスキル集合
    skills_str = df[df["item_id"]==item_id]["skill"].astype(str).iloc[0]
    K = [int(s) for s in skills_str.split(",")]

    per_skill = []
    for sid in K:
        if sid not in params_by_skill:
            continue
        p = params_by_skill[sid]
        sdf = df_history_exploded[(df_history_exploded["user_id"]==user_id) &
                                  (df_history_exploded["skill_id"]==sid)].sort_values("date")
        obs = sdf["result"].astype(int).tolist()
        pL, pL_next, _ = bkt_predict_next(obs, p)
        T_k = (1-p["p_S"])*pL_next + p["p_G"]*(1-pL_next)
        per_skill.append({
            "skill_id": sid, "P(L)": pL, "P(L^+)": pL_next, "T_k": T_k,
            "learn": p["p_T"], "slip": p["p_S"], "guess": p["p_G"]
        })
    if not per_skill:
        return None

    # 結合確率（幾何平均/soft-AND/AND/ロジット平均など）
    Tks = [r["T_k"] for r in per_skill]
    p_corr = combine_Tks(Tks, mode=combine_mode, alpha=alpha)
    if p_corr is None:
        return None

    # Noisy-AND の誤答式を用いた事後（他モードでも近似として利用）
    Q = 1.0
    for t in Tks: Q *= t
    gains = []
    for r in per_skill:
        Tk = r["T_k"]; p = params_by_skill[r["skill_id"]]
        pL_next = r["P(L^+)"]

        # 正答時
        post1 = ((1-p["p_S"])*pL_next) / (((1-p["p_S"])*pL_next) + (p["p_G"]*(1-pL_next)))
        # 誤答時（AND前提の厳密式）
        Q_minus = Q / max(Tk, 1e-12)
        num0 = (1 - (1-p["p_S"])*Q_minus) * pL_next
        den0 = num0 + (1 - p["p_G"]*Q_minus) * (1 - pL_next)
        post0 = num0 / max(den0, 1e-12)

        after1 = post1 + (1-post1)*p["p_T"]
        after0 = post0 + (1-post0)*p["p_T"]
        E_after = p_corr*after1 + (1-p_corr)*after0
        gains.append(E_after - r["P(L)"])

    delta = float(sum(gains)/len(gains))
    return {
        "user_id": user_id, "item_id": item_id, "skills": K,
        "p_correct": p_corr, "delta": delta,
        "per_skill": per_skill,          # ← これを返すように修正
        "per_skill_gain": gains
    }


def score_multi_objective(delta, p, target_band=(0.4, 0.7), lam=0.3):
    """
    lam: 0〜1。0ならΔだけ、1なら帯への近さだけ。
    """
    lo, hi = target_band
    # 帯からの距離（内側は距離0、外側は線形ペナルティ）
    if p < lo:  dist = (lo - p)
    elif p > hi: dist = (p - hi)
    else:        dist = 0.0
    # 近さスコア（0〜1, 高いほど良い）
    closeness = 1.0 / (1.0 + 10.0*dist)  # 係数は適宜
    return (1-lam)*delta + lam*closeness


def auto_target_band_from_preds(p_list, low_q=0.60, high_q=0.80):
    import numpy as np
    p = np.array(p_list, dtype=float)
    lo = float(np.quantile(p, low_q))
    hi = float(np.quantile(p, high_q))
    # 最低幅を確保
    if hi - lo < 0.05:
        mid = (lo+hi)/2
        lo, hi = max(0.01, mid-0.03), min(0.99, mid+0.03)
    return (lo, hi)


# トップ候補の抽出 & LLM への JSON
def recommend_topK(user_id: int, K: int,
                   df_all: pd.DataFrame, df_exploded: pd.DataFrame,
                   params_by_skill: Dict[int,Dict[str,float]],
                   combine_mode="soft_and", alpha=0.5,
                   target_band="auto", lam=0.3,
                   max_candidates=2000):
    seen = set(df_all[df_all["user_id"]==user_id]["item_id"].unique())
    cand = [iid for iid in df_all["item_id"].unique().tolist() if iid not in seen][:max_candidates]
    # --- 1st pass: p, Δ の粗計算（帯の自動化用） ---
    scratch=[]
    for iid in cand:
        eg = expected_gain_for_item(user_id, iid, df_exploded, params_by_skill,
                                    combine_mode=combine_mode, alpha=alpha)
        if eg is None: 
            continue
        scratch.append((iid, eg["p_correct"], eg["delta"], eg["skills"]))
    if not scratch:
        return pd.DataFrame(columns=["item_id","p_correct","delta","score","skills"])

    if target_band == "auto":
        band = auto_target_band_from_preds([p for _,p,_,_ in scratch], low_q=0.50, high_q=0.75)
    else:
        band = target_band

    # --- 2nd pass: 多目的スコアを付与 ---
    rows=[]
    for iid, p, dlt, skills in scratch:
        s = score_multi_objective(dlt, p, target_band=band, lam=lam)
        rows.append({"item_id": iid, "p_correct": p, "delta": dlt, "score": s, "skills": tuple(sorted(skills))})
    rec = pd.DataFrame(rows).sort_values(["score","delta","p_correct"], ascending=[False,False,False])
    return rec



def build_llm_payload(user_id: int, recs: pd.DataFrame):
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


def explain_recommendation(item_id: int, user_id: int, df_exploded: pd.DataFrame,
                           params_by_skill: Dict[int,Dict[str,float]],
                           combine_mode="soft_and", alpha=0.5):
    # 簡単な“根拠文”を生成（LLMに渡す前のテンプレ）
    skills_str = df[df["item_id"]==item_id]["skill"].astype(str).iloc[0]
    K = [int(s) for s in skills_str.split(",")]
    parts=[]
    for sid in K:
        if sid not in params_by_skill: continue
        p = params_by_skill[sid]
        sdf = df_exploded[(df_exploded["user_id"]==user_id)&(df_exploded["skill_id"]==sid)].sort_values("date")
        obs = sdf["result"].astype(int).tolist()
        pL, pL_next, _ = bkt_predict_next(obs, p)
        parts.append(f"skill {sid}: P(L)={pL:.2f}→{pL_next:.2f}")
    return f"次の問題は {', '.join(parts)} の強化に最適です。"



def jaccard(a: tuple, b: tuple) -> float:
    A, B = set(a), set(b)
    return 0.0 if not A or not B else len(A & B) / len(A | B)

def diversify_by_skills(recs_df: pd.DataFrame, topN=10, mmr_lambda=0.7):
    """
    recs_df: columns ['item_id','score','skills', ...]
    mmr_lambda: 1に近いほど元スコア重視、0に近いほど多様性重視
    """
    selected = []
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
                sim = max(sim, jaccard(r["skills"], s["skills"]))
            val = mmr_lambda * r["score"] - (1-mmr_lambda) * sim
            if val > best_val:
                best_val, best = val, r
        selected.append(best)
        pool.remove(best)
    return pd.DataFrame(selected)




from pathlib import Path
import json
import datetime as dt

def save_payload_json(payload, user_id, out_dir="../json", prefix="recommendations"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # ディレクトリが無ければ作成
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = out_dir / f"{prefix}_user{user_id}_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved JSON] {fname.resolve()}")
    return str(fname)



from pathlib import Path
import json, pandas as pd

def load_latest_recs(out_dir="../json", user_id=None, prefix="recommendations"):
    out = Path(out_dir)
    pattern = f"{prefix}_user{user_id}_*.json" if user_id is not None else f"{prefix}_user*_*.json"
    files = sorted(out.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {out} with pattern {pattern}")
    path = files[-1]
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    recs = pd.DataFrame(payload["recommendations"])
    recs["rank"] = range(1, len(recs)+1)   # 1始まりの順位
    return path, payload, recs

def show_top(recs, topN=10):
    cols = ["rank","item_id","p_next_correct","expected_mastery_gain","score"]
    print(recs[cols].head(topN).to_string(index=False))

def save_recs_csv(recs, user_id, out_dir="../json", prefix="recs_latest"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}_user{user_id}.csv"
    recs.to_csv(path, index=False, encoding="utf-8")
    print(f"[Saved CSV] {path.resolve()}")




# --- 1) 初期化：履歴から user_state[sid] = P(L) を作る ---
def init_user_state(user_id, df_exploded, params_by_skill):
    state = {}
    for sid, p in params_by_skill.items():
        sdf = df_exploded[(df_exploded["user_id"]==user_id) &
                          (df_exploded["skill_id"]==sid)].sort_values("date")
        obs = sdf["result"].astype(int).tolist()
        pL, _, _ = bkt_predict_next(obs, p)  # 既存ロジックと整合
        state[sid] = pL
    return state

# --- 2) 出題の結果 o∈{0,1} を受けて、関係スキルだけ更新 ---
def bkt_update_state_for_item(user_state, item_id, outcome, params_by_skill, df_all):
    skills_str = str(df_all[df_all["item_id"]==item_id]["skill"].iloc[0])
    skills = [int(s) for s in skills_str.split(",")]
    for sid in skills:
        if sid not in params_by_skill: 
            continue
        p = params_by_skill[sid]
        pL = user_state.get(sid, p["p_L0"])
        # 既存実装と同じ順序：学習遷移 → 観測で後退更新
        pL_prior = pL + (1 - pL) * p["p_T"]
        if outcome == 1:
            num = (1 - p["p_S"]) * pL_prior
            den = num + p["p_G"] * (1 - pL_prior)
        else:
            num = p["p_S"] * pL_prior
            den = num + (1 - p["p_G"]) * (1 - pL_prior)
        pL_post = pL_prior if den <= 0 else num / den
        user_state[sid] = float(np.clip(pL_post, 1e-9, 1-1e-9))
    return user_state




# =========================
# 6) メイン：学習と例示
# =========================
if __name__ == "__main__":
    if USE_WEIGHTED_EM:
        # ---- A案：explode＋1/m重み ----
        df_ex = prepare_exploded(df)
        params_by_skill = fit_per_skill_weighted(df_ex, max_iter=EM_MAX_ITERS,
                                                 min_eff_samples=MIN_EFFECTIVE_SAMPLES)
        # ユーザ別のスキルレポート（explodeデータを使う）
        sample_user = int(df_ex["user_id"].value_counts().idxmax())
        rep_rows=[]
        for sid in sorted(params_by_skill.keys()):
            p = params_by_skill[sid]
            sdf = df_ex[(df_ex["user_id"]==sample_user)&(df_ex["skill_id"]==sid)].sort_values("date")
            obs = sdf["result"].astype(int).tolist()
            pL, pL_next, p_corr = bkt_predict_next(obs, p)
            rep_rows.append({"user_id": sample_user, "skill_id": sid, "n_obs": len(obs),
                             "P(L)_posterior": pL, "P(L)_after_learn": pL_next,
                             "P(next correct)": p_corr, "learn":p["p_T"], "slip":p["p_S"], "guess":p["p_G"]})
        report = pd.DataFrame(rep_rows).sort_values("P(next correct)", ascending=False)
        print("\n[Weighted-EM] User {} skill report (top 10):".format(sample_user))
        print(report.head(10).to_string(index=False))

        # 多スキルアイテムの将来正答確率（Noisy-AND）
        # 例として、そのユーザが最近解いた item を一つ取って推定
        recent_item = int(df[df["user_id"]==sample_user].sort_values("date")["item_id"].iloc[-1])
        pred = predict_item_multiskill(sample_user, recent_item, df_ex, params_by_skill,
                                       use_noisy_and=USE_NOISY_AND_FOR_PRED)
        print("\n[Multi-skill prediction] user={}, item={}, skills={}, p_correct={:.6f}"
            .format(pred["user_id"], pred["item_id"], pred["skills"],
                    pred["p_correct"] if pred["p_correct"] is not None else float('nan')))



        # 学習直後に追加（全体の正解率や有効サンプルを確認）
        print("Global correctness =", df["result"].mean())
        df_ex = prepare_exploded(df)
        print("avg_skills_per_item =", df["skill"].astype(str).str.count(",").add(1).mean())
        print("Top-10 skills by effective samples:")
        print(df_ex.groupby("skill_id")["weight"].sum().sort_values(ascending=False).head(10))

        # 学習後の main 内


        # 推薦作成（多様化まで）
        raw = recommend_topK(sample_user, K=100, df_all=df, df_exploded=df_ex,
                            params_by_skill=params_by_skill,
                            combine_mode="soft_and", alpha=0.5,
                            target_band="auto", lam=0.3)
        topK_div = diversify_by_skills(raw, topN=10, mmr_lambda=0.7)

        print("\n[Top-K diversified]")
        print(topK_div[["item_id","p_correct","delta","score"]].to_string(index=False))


        combine_mode = "geom"  # ← 初手は幾何平均が無難
        topK = recommend_topK(sample_user, K=10, df_all=df, df_exploded=df_ex,
                      params_by_skill=params_by_skill,
                      combine_mode="soft_and", alpha=0.5,
                      target_band=(0.4,0.7), lam=0.3)

        print("\n[Top-K by Δ]"); print(topK.to_string(index=False))

        # ここで保存するのを「多様化版」にする
        llm_payload = build_llm_payload(sample_user, topK_div)

        # ファイル保存（既存の関数をそのまま利用）
        _ = save_payload_json(llm_payload, user_id=sample_user, out_dir="../json", prefix="recommendations")

        #import json; print("\n[LLM payload]\n" + json.dumps(llm_payload, ensure_ascii=False, indent=2))


        # 使い方例（user_id=58 の最新JSONを読む）
        path, payload, recs = load_latest_recs("../json", user_id=58)
        print(f"[Loaded] {path}")
        show_top(recs, topN=10)
        save_recs_csv(recs, user_id=payload["user_id"], out_dir="../json")

        # “次の1問”を使うときは、単純に rank=1 を採用
        next_item = int(recs.sort_values(["score","expected_mastery_gain","p_next_correct"],
                                        ascending=[False,False,False]).iloc[0]["item_id"])
        print("Next item to serve:", next_item)






    else:
        # ---- 既存の単一スキルBKT ----
        params_by_skill = fit_per_skill(df_single, max_iter=EM_MAX_ITERS)
        sample_user = 68  # 例
        report = predict_user_single(df_single, params_by_skill, user_id=sample_user)
        print(report.head(10).to_string(index=False))


