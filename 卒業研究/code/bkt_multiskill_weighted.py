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


# =========================
# 1) データ読み込み
# =========================
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])

# ----- 単一スキル subset（既存の流れを保持）
df_single = df[df["skill"].astype(str).str.count(",")==0].copy()
df_single["skill_id"] = df_single["skill"].astype(int)
df_single = df_single.sort_values(["user_id","skill_id","date"])


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
    """重み付きEM（忘却なし）。観測毎に weight を掛けた期待カウントで更新"""
    if init is None:
        p_L0, p_T, p_S, p_G = 0.2, 0.1, 0.1, 0.2
    else:
        p_L0, p_T, p_S, p_G = init
    def clip(x): return float(np.clip(x, 1e-5, 1-1e-5))
    last_ll = None

    # ユーザ別系列（obs, wts）
    seqs: List[Tuple[np.ndarray,np.ndarray]] = []
    for _, udf in df_skill.groupby("user_id", sort=False):
        obs = udf["result"].astype(int).to_numpy()
        wts = udf["weight"].astype(float).to_numpy()
        if len(obs)>0:
            seqs.append((obs, wts))

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

        for obs, wts in seqs:
            T = len(obs)
            if T==0: 
                continue
            pi = np.array([1-p_L0, p_L0], dtype=float)
            B1 = np.array([p_G, 1-p_S], dtype=float)   # o=1
            B0 = np.array([1-p_G, p_S], dtype=float)   # o=0

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

            # 尤度/放出
            w_arr = wts; o_arr = obs.astype(float)
            sum_emit_L_w += (gamma[:,1] * w_arr).sum()
            sum_emit_L_incorrect_w += ((1 - o_arr) * gamma[:,1] * w_arr).sum()
            sum_emit_U_w += (gamma[:,0] * w_arr).sum()
            sum_emit_U_correct_w += (o_arr * gamma[:,0] * w_arr).sum()

            sum_gamma_L1_w += gamma[0,1] * wts[0]
            sum_first_w    += wts[0]

            total_ll += np.log(scale + 1e-300).sum()

        # M-step
        p_L0 = clip(sum_gamma_L1_w / max(sum_first_w, 1e-12))
        p_T  = clip(sum_xi_U2L_w  / max(sum_gamma_U_w,  1e-12))
        p_S  = clip(sum_emit_L_incorrect_w / max(sum_emit_L_w, 1e-12))
        p_G  = clip(sum_emit_U_correct_w   / max(sum_emit_U_w, 1e-12))

        if last_ll is not None and abs(total_ll - last_ll) < tol:
            break
        last_ll = total_ll

    return {"p_L0": p_L0, "p_T": p_T, "p_S": p_S, "p_G": p_G, "loglik": last_ll, "iters": it+1}


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
def predict_item_multiskill(user_id: int, item_id: int,
                            df_history_exploded: pd.DataFrame,
                            params_by_skill: Dict[int,Dict[str,float]],
                            use_noisy_and: bool = True) -> Dict[str,Any]:
    """
    指定ユーザの「今このタイミングで item_id を出題したら」の正答確率を推定。
    - 各スキルの直前習熟 P(L^+) を計算し、Noisy-AND なら積、平均なら平均で結合。
    返り値: {"user_id","item_id","skills":[...], "p_correct", "per_skill": [{skill_id, P(L), P(L^+), T_k}]}
    """
    # そのアイテムが要求するスキル集合
    skills_str = df[df["item_id"]==item_id]["skill"].astype(str).iloc[0]
    skill_ids = [int(s) for s in skills_str.split(",")]

    # ユーザの各スキル履歴（explode済み）から P(L), P(L^+) を計算
    per_skill = []
    for sid in skill_ids:
        if sid not in params_by_skill:
            continue
        p = params_by_skill[sid]
        sdf = df_history_exploded[(df_history_exploded["user_id"]==user_id) &
                                  (df_history_exploded["skill_id"]==sid)].sort_values("date")
        obs = sdf["result"].astype(int).tolist()
        pL, pL_next, _ = bkt_predict_next(obs, p)
        # このスキル単体での「部分正解確率」 T_k
        T_k = (1-p["p_S"])*pL_next + p["p_G"]*(1-pL_next)
        per_skill.append({"skill_id": sid, "P(L)": pL, "P(L^+)": pL_next, "T_k": T_k,
                          "learn":p["p_T"], "slip":p["p_S"], "guess":p["p_G"]})

    if len(per_skill)==0:
        return {"user_id": user_id, "item_id": item_id, "skills": skill_ids, "p_correct": None, "per_skill": []}

    if use_noisy_and:
        p_correct = 1.0
        for r in per_skill:
            p_correct *= r["T_k"]
    else:
        p_correct = float(np.mean([r["T_k"] for r in per_skill]))

    return {"user_id": user_id, "item_id": item_id, "skills": skill_ids,
            "p_correct": p_correct, "per_skill": per_skill}


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
        print("\n[Multi-skill prediction] user={}, item={}, skills={}, p_correct={:.3f}"
              .format(pred["user_id"], pred["item_id"], pred["skills"], pred["p_correct"] if pred["p_correct"] is not None else float('nan')))

    else:
        # ---- 既存の単一スキルBKT ----
        params_by_skill = fit_per_skill(df_single, max_iter=EM_MAX_ITERS)
        sample_user = 68  # 例
        report = predict_user_single(df_single, params_by_skill, user_id=sample_user)
        print(report.head(10).to_string(index=False))
