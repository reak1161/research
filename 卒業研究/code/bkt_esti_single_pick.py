# bkt_esti_single_pick.py
import pandas as pd, numpy as np
import argparse, os

# ===== 1) データ読み込み（単一スキル前提） =====
def load_data(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    df["date"] = pd.to_datetime(df["date"])
    df_single = df.copy()
    df_single["skill_id"] = df_single["skill"].astype(int)
    df_single = df_single.sort_values(["user_id","skill_id","date"])
    return df_single

# ===== 2) BKT（忘却なし） EM 推定 =====
def bkt_em(sequences, max_iter=30, tol=1e-4, init=None):
    if init is None:
        p_L0, p_T, p_S, p_G = 0.2, 0.1, 0.1, 0.2
    else:
        p_L0, p_T, p_S, p_G = init

    def clip(x):
        return float(np.clip(x, 1e-5, 1-1e-5))

    last_ll = None
    it_used = 0
    for it in range(1, max_iter+1):
        A = np.array([[1-p_T, p_T],
                      [0.0,   1.0]])  # U->L のみ

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

            def b(o):
                # state 0: U, state 1: L
                return np.array([ p_G if o==1 else (1-p_G),
                                  (1-p_S) if o==1 else p_S ])

            alpha = np.zeros((T,2))
            scale = np.zeros(T)
            alpha[0] = pi * b(obs[0])
            scale[0] = alpha[0].sum() + 1e-300
            alpha[0] /= scale[0]
            for t in range(1, T):
                alpha[t] = (alpha[t-1].dot(A)) * b(obs[t])
                scale[t] = alpha[t].sum() + 1e-300
                alpha[t] /= scale[t]

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
                sum_xi_U2L += 0.0 if den<=0 else num/den
                sum_gamma_U += gamma[t,0]

            sum_emit_L += gamma[:,1].sum()
            sum_emit_L_incorrect += ((1-np.array(obs)) * gamma[:,1]).sum()
            sum_emit_U += gamma[:,0].sum()
            sum_emit_U_correct   += ( np.array(obs)   * gamma[:,0]).sum()

            sum_gamma_L1 += gamma[0,1]
            total_ll += float(np.sum(np.log(scale + 1e-300)))

        nseq = max(len(sequences), 1)
        p_L0 = clip(sum_gamma_L1 / nseq)
        p_T  = clip(sum_xi_U2L   / max(sum_gamma_U, 1e-12))
        p_S  = clip(sum_emit_L_incorrect / max(sum_emit_L, 1e-12))
        p_G  = clip(sum_emit_U_correct   / max(sum_emit_U, 1e-12))

        it_used = it
        if last_ll is not None and abs(total_ll - last_ll) < tol:
            break
        last_ll = total_ll

    return {"p_L0": p_L0, "p_T": p_T, "p_S": p_S, "p_G": p_G,
            "loglik": last_ll if last_ll is not None else float('nan'),
            "iters": it_used}

def fit_by_skill(df_single):
    params_by_skill = {}
    for sid, sdf in df_single.groupby("skill_id"):
        seqs = [udf.sort_values("date")["result"].astype(int).tolist()
                for _, udf in sdf.groupby("user_id")]
        if len(seqs) == 0:
            continue
        params_by_skill[int(sid)] = bkt_em(seqs, max_iter=30)
    return params_by_skill

# ===== 3) 予測：P(L)事後と次正答確率 =====
def bkt_predict_next(obs, p):
    pL = p["p_L0"]
    for o in obs:
        pL = pL + (1 - pL) * p["p_T"]   # 事前学習
        num = ((1 - p["p_S"]) if o == 1 else p["p_S"]) * pL
        den = num + ((p["p_G"]) if o == 1 else (1 - p["p_G"])) * (1 - pL)
        pL = pL if den <= 0 else num / den
    pL_next   = pL + (1 - pL) * p["p_T"]
    p_correct = (1 - p["p_S"]) * pL_next + p["p_G"] * (1 - pL_next)
    return pL, pL_next, p_correct

def predict_user(df_single, params_by_skill, user_id):
    rows=[]
    for sid, sdf in df_single[df_single["user_id"]==user_id].groupby("skill_id"):
        obs = sdf.sort_values("date")["result"].astype(int).tolist()
        p   = params_by_skill.get(int(sid))
        if not p:
            continue
        pL, pL_next, p_corr = bkt_predict_next(obs, p)
        rows.append({"user_id":user_id, "skill_id":int(sid), "n_obs":len(obs),
                     "P(L)_posterior":pL, "P(L)_after_learn":pL_next,
                     "P(next correct)":p_corr, **p})
    out = pd.DataFrame(rows).sort_values("P(next correct)", ascending=False)
    return out

def predict_user_skill(df_single, params_by_skill, user_id, skill_id):
    sdf = df_single[(df_single["user_id"]==user_id) & (df_single["skill_id"]==skill_id)]
    obs = sdf.sort_values("date")["result"].astype(int).tolist()
    p   = params_by_skill.get(int(skill_id))
    if not p:
        # スキルに学習データがない場合でも、初期値からの予測を返す
        p = {"p_L0":0.2,"p_T":0.1,"p_S":0.1,"p_G":0.2,"loglik":float("nan"),"iters":0}
    pL, pL_next, p_corr = bkt_predict_next(obs, p)
    row = {"user_id":int(user_id), "skill_id":int(skill_id), "n_obs":len(obs),
           "P(L)_posterior":pL, "P(L)_after_learn":pL_next,
           "P(next correct)":p_corr, **p}
    return pd.DataFrame([row])

# ===== 追加: 複数スキル指定のパース =====
def _parse_skill_ids(args):
    skills = []
    # --skill_ids は空白区切りだが、カンマ区切りにも対応
    if getattr(args, "skill_ids", None):
        for tok in args.skill_ids:
            for s in str(tok).split(','):
                s = s.strip()
                if s != "":
                    skills.append(int(s))
    # 従来の --skill_id（単一）も併用可（後方互換）
    if args.skill_id is not None:
        skills.append(int(args.skill_id))
    # 重複除去（順序維持）
    out, seen = [], set()
    for k in skills:
        if k not in seen:
            out.append(k); seen.add(k)
    return out

# ===== 4) CLI（出力指定） =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="one_200_50_500_50000.csv",
                        help="入力CSV（単一スキルの行データ）")
    parser.add_argument("--nrows", type=int, default=None,
                        help="先頭から読み込む行数（例: 50000）")
    parser.add_argument("--user_id", type=int, default=None,
                        help="出力対象のユーザーID")
    parser.add_argument("--skill_id", type=int, default=None,
                        help="出力対象のスキルID（user_idと併用）")
    # 追加: 複数スキル指定
    parser.add_argument("--skill_ids", nargs='*',
                        help="複数スキル指定。例: --skill_ids 3 7 21 または --skill_ids 3,7,21")
    # 追加: 並び替えの可否と基準（スキル指定時のみ有効）
    parser.add_argument("--sort", choices=["auto","none","pnext","skill","nobs"],
                        default="auto",
                        help="スキル指定時の並び替え。auto: P(next correct)降順 / none: 入力順のまま / pnext: 次回正答確率 / skill: skill_id / nobs: 観測数")
    parser.add_argument("--ascending", action="store_true",
                        help="昇順にする（デフォルトは降順）。--sort=none の場合は無視")
    parser.add_argument("--topk", type=int, default=10,
                        help="user_idのみ指定時の表示件数")
    args = parser.parse_args()

    # CSVパスの存在確認（相対パスにも対応）
    candidates = [args.csv,
                  os.path.join(os.path.dirname(__file__), args.csv),
                  os.path.join(os.path.dirname(__file__), "../csv/one_200_50_500_50000.csv")]
    data_path = None
    for c in candidates:
        if os.path.exists(c):
            data_path = c
            break
    if data_path is None:
        raise FileNotFoundError(f"CSVが見つかりません: {args.csv}")

    df_single = load_data(data_path, nrows=args.nrows)
    params_by_skill = fit_by_skill(df_single)

    # ① user_id と 複数 skill_ids が指定された場合 → まとめて出力
    skills = _parse_skill_ids(args)
    if args.user_id is not None and len(skills) > 0:
        frames = [predict_user_skill(df_single, params_by_skill, args.user_id, sid)
                  for sid in skills]
        report = pd.concat(frames, ignore_index=True)
        # 並び替え制御
        if args.sort != "none":
            if args.sort in ("auto", "pnext"):
                key = "P(next correct)"
            elif args.sort == "skill":
                key = "skill_id"
            elif args.sort == "nobs":
                key = "n_obs"
            else:
                key = "P(next correct)"
            report = report.sort_values(key, ascending=args.ascending)
        # --sort=none のときは入力順（concat順）のまま出力
        print(report.to_string(index=False))
        return

    # （以下は従来分岐を維持）
    if args.user_id is not None and args.skill_id is not None:
        # ② ユーザー×スキルを1行で表示
        report = predict_user_skill(df_single, params_by_skill, args.user_id, args.skill_id)
        print(report.to_string(index=False))
    elif args.user_id is not None:
        # ③ ユーザー内のスキル別ランキング（従来通りP(next correct)降順でhead(topk)）
        report = predict_user(df_single, params_by_skill, user_id=args.user_id)
        print(report.head(args.topk).to_string(index=False))
    else:
        # ④ 互換用サンプル
        for uid in [68, 42, 1]:
            report = predict_user(df_single, params_by_skill, user_id=uid)
            print(report.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
