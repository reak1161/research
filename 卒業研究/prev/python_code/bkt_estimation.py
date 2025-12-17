import pandas as pd, numpy as np

# ===== 1) データ読み込み & 単一スキル抽出 =====
df = pd.read_csv("../csv/200_50_500_100000.csv")
df["date"] = pd.to_datetime(df["date"])

# 単一スキル問題だけに限定（まずはBKTの前提に合わせる）
df_single = df[df["skill"].str.count(",")==0].copy()
df_single["skill_id"] = df_single["skill"].astype(int)
df_single = df_single.sort_values(["user_id","skill_id","date"])

# ===== 2) BKT（忘却なし） EM 推定 =====
def bkt_em(sequences, max_iter=30, tol=1e-4, init=None):
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
            T=len(obs); 
            if T==0: continue
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

def fit_per_skill(df_single):
    params={}
    for sid, sdf in df_single.groupby("skill_id"):
        seqs=[udf["result"].astype(int).tolist() for _,udf in sdf.groupby("user_id")]
        params[sid]=bkt_em(seqs, max_iter=30)
    return params

# 学習（スキル別BKT）
params_by_skill = fit_per_skill(df_single)

# ===== 3) 推論：ユーザ×スキルの P(L) と“次の正答確率” =====
def bkt_predict_next(obs, p):
    pL=p["p_L0"]
    for o in obs:
        pL = pL + (1-pL)*p["p_T"]  # 事前学習
        num = ((1-p["p_S"]) if o==1 else p["p_S"]) * pL
        den = num + ((p["p_G"]) if o==1 else (1-p["p_G"]))*(1-pL)
        pL = pL if den<=0 else num/den
    pL_next = pL + (1-pL)*p["p_T"]
    p_correct = (1-p["p_S"])*pL_next + p["p_G"]*(1-pL_next)
    return pL, pL_next, p_correct

def predict_user(df_single, params_by_skill, user_id):
    rows=[]
    for sid, sdf in df_single[df_single["user_id"]==user_id].groupby("skill_id"):
        obs=sdf.sort_values("date")["result"].astype(int).tolist()
        p=params_by_skill.get(sid)
        if not p: continue
        pL, pL_next, p_corr = bkt_predict_next(obs, p)
        rows.append({"user_id":user_id,"skill_id":sid,"n_obs":len(obs),
                     "P(L)_posterior":pL,"P(next correct)":p_corr, **p})
    out=pd.DataFrame(rows).sort_values("P(next correct)", ascending=False)
    return out

# 例：ユーザ68のレポート
report = predict_user(df_single, params_by_skill, user_id=68)
print(report.head(10).to_string(index=False))

report = predict_user(df_single, params_by_skill, user_id=42)
print(report.head(10).to_string(index=False))

report = predict_user(df_single, params_by_skill, user_id=1)
print(report.head(10).to_string(index=False))