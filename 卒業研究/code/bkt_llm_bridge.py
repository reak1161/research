#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bkt_llm_bridge.py
- 単一スキルBKT（忘却なし）を用いて、ユーザー×スキルの次回正答確率などを算出
- 候補アイテム(item_id, skill_id)から学習効果が見込めるものをランキング
- LLMに渡す System/User メッセージと JSONスキーマのヒントを生成
- （任意）モックのLLM出力JSONを表示

依存: pandas, numpy
想定CSV: one_200_50_500_50000.csv（列: user_id, item_id, skill, result, date）
"""

import argparse, os, json, math, numpy as np, pandas as pd
from typing import List, Dict

# ========== BKT: EM推定 ==========
def bkt_em(sequences: List[List[int]], max_iter=30, tol=1e-4, init=None) -> Dict[str, float]:
    if init is None:
        p_L0, p_T, p_S, p_G = 0.2, 0.1, 0.1, 0.2
    else:
        p_L0, p_T, p_S, p_G = init

    def clip(x): return float(np.clip(x, 1e-5, 1-1e-5))
    last_ll, it_used = None, 0

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
            Tn = len(obs)
            if Tn == 0: 
                continue
            pi = np.array([1-p_L0, p_L0])

            def b(o):
                # state 0: U, state 1: L
                return np.array([ p_G if o==1 else (1-p_G),
                                  (1-p_S) if o==1 else p_S ])

            alpha = np.zeros((Tn,2))
            scale = np.zeros(Tn)
            alpha[0] = pi * b(obs[0])
            scale[0] = alpha[0].sum() + 1e-300
            alpha[0] /= scale[0]
            for t in range(1, Tn):
                alpha[t] = (alpha[t-1].dot(A)) * b(obs[t])
                scale[t] = alpha[t].sum() + 1e-300
                alpha[t] /= scale[t]

            beta = np.zeros((Tn,2))
            beta[-1] = 1.0 / scale[-1]
            for t in range(Tn-2, -1, -1):
                bt1 = b(obs[t+1])
                beta[t] = (A * (bt1 * beta[t+1])).sum(axis=1)
                beta[t] /= scale[t]

            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)

            for t in range(Tn-1):
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

def fit_by_skill(df_single: pd.DataFrame, skills: List[int]) -> Dict[int, Dict[str, float]]:
    out = {}
    for sid in skills:
        sdf = df_single[df_single["skill_id"]==sid]
        seqs = [udf.sort_values("date")["result"].astype(int).tolist()
                for _, udf in sdf.groupby("user_id")]
        if len(seqs) == 0:
            continue
        out[int(sid)] = bkt_em(seqs, max_iter=30)
    return out

# ========== 予測 ==========
def bkt_predict_next(obs: List[int], p: Dict[str,float]):
    pL = p["p_L0"]
    for o in obs:
        pL = pL + (1 - pL) * p["p_T"]   # 事前学習
        num = ((1 - p["p_S"]) if o == 1 else p["p_S"]) * pL
        den = num + ((p["p_G"]) if o == 1 else (1 - p["p_G"])) * (1 - pL)
        pL = pL if den <= 0 else num / den
    pL_next   = pL + (1 - pL) * p["p_T"]
    p_correct = (1 - p["p_S"]) * pL_next + p["p_G"] * (1 - pL_next)
    return pL, pL_next, p_correct

def predict_user_skill(df_single: pd.DataFrame, params_by_skill: Dict[int,Dict[str,float]],
                       user_id: int, skill_id: int) -> pd.DataFrame:
    sdf = df_single[(df_single["user_id"]==user_id) & (df_single["skill_id"]==skill_id)]
    obs = sdf.sort_values("date")["result"].astype(int).tolist()
    p   = params_by_skill.get(int(skill_id))
    if not p:
        p = {"p_L0":0.2,"p_T":0.1,"p_S":0.1,"p_G":0.2,"loglik":float("nan"),"iters":0}
    pL, pL_next, p_corr = bkt_predict_next(obs, p)
    row = {"user_id":int(user_id), "skill_id":int(skill_id), "n_obs":len(obs),
           "P(L)_posterior":pL, "P(L)_after_learn":pL_next,
           "P(next correct)":p_corr, **p}
    return pd.DataFrame([row])

# ========== 候補ランキング & LLMメッセージ ==========
def rank_candidates(report_df: pd.DataFrame, items_df: pd.DataFrame, user_id: int,
                    sweet=0.7, w_explore=0.15, topn=5) -> pd.DataFrame:
    rep = report_df[report_df['user_id']==user_id].set_index('skill_id')
    rows = []
    for _, it in items_df.iterrows():
        sid = int(it['skill_id'])
        if sid not in rep.index:
            continue
        r = rep.loc[sid]
        pnext = float(r['P(next correct)'])
        score  = -abs(pnext - sweet)                     # 目標帯に近いほど高得点
        score += w_explore/np.sqrt(float(r['n_obs'])+1)  # 観測少は探索ボーナス
        score += 0.3 * float(r['p_T']) * (1.0 - float(r['P(L)_posterior']))  # 学習余地
        rows.append({
            "item_id": str(it['item_id']),
            "skill_id": sid,
            "pnext": pnext,
            "score": score
        })
    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand
    return cand.sort_values("score", ascending=False).head(topn)

def build_llm_messages(user_id: int, ranked_df: pd.DataFrame, language="ja"):
    candidates = [
        {"item_id": str(r.item_id), "skill_id": int(r.skill_id),
         "expected_p_correct": round(float(r.pnext), 2)}
        for r in ranked_df.itertuples(index=False)
    ]
    system = (
        "あなたは教育向けの推薦アシスタントです。指定された候補問題リストからのみ選び、"
        "出力は必ず指定のJSONスキーマに厳密に従って日本語で返してください。候補外のIDは使用禁止です。"
        "助言は短く具体的に、数値は小数第2位まで。"
    )
    user_payload = {
        "task": "次に解くべき問題の推薦と、学習者向けフィードバック文の生成",
        "user_id": user_id,
        "policy": {
            "selection_rule": ("候補から最大5問。expected_p_correctが0.6〜0.8に近いものを優先。"
                               "直近の成功確率が高すぎる(>0.9)ものは除外、低すぎる(<0.4)ものはreview扱い。"),
            "types": ["review","practice","challenge"],
            "style": {"student_msg_len":"<=200文字","teacher_note_len":"<=100文字","lang":language}
        },
        "candidates": candidates
    }
    schema_hint = (
        '必ず次のJSONのみを返すこと:\n'
        '{ "schema_version":"1.0", "user_id":<int>, '
        '"recommendations":[{"item_id":"<id>","skill_id":<int>,"reason":"<短文>",'
        '"expected_p_correct":<num>,"type":"review|practice|challenge"}],'
        '"feedback_to_student":"<200文字以内>", "note_to_teacher":"<100文字以内>", "next_checkin_days":<int> }'
    )
    return system, json.dumps(user_payload, ensure_ascii=False, indent=2), schema_hint, user_payload

def build_mock_llm_output(user_id: int, ranked_df: pd.DataFrame) -> str:
    recs = []
    for r in ranked_df.itertuples(index=False):
        t = "practice"
        if r.pnext >= 0.85: t = "challenge"
        if r.pnext <= 0.50: t = "review"
        recs.append({
            "item_id": str(r.item_id),
            "skill_id": int(r.skill_id),
            "reason": f"P(next correct)={round(float(r.pnext),2)}で目標帯に近い。",
            "expected_p_correct": round(float(r.pnext),2),
            "type": t
        })
    out = {
        "schema_version": "1.0",
        "user_id": int(user_id),
        "recommendations": recs,
        "feedback_to_student": "次は少し難しい問題に挑戦しましょう。間違えた箇所は手順を分解し、根拠を必ず書き出すこと。復習は24時間以内に1問。",
        "note_to_teacher": "成功確率0.6〜0.8帯を中心に推薦。観測が少ないスキルに探索を付与。",
        "next_checkin_days": 2
    }
    return json.dumps(out, ensure_ascii=False, indent=2)

# ========== ユーティリティ ==========
def load_single_skill_df(csv_path: str, nrows=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows)
    # 列名の揺れを補正
    rename_map = {}
    if "skill_id" in df.columns and "skill" not in df.columns:
        rename_map["skill_id"] = "skill"
    if "correct" in df.columns and "result" not in df.columns:
        rename_map["correct"] = "result"
    if rename_map:
        df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"])
    df["skill_id"] = df["skill"].astype(int)
    df = df.sort_values(["user_id","skill_id","date"]).reset_index(drop=True)
    return df

def pick_skills_for_user(df: pd.DataFrame, user_id: int, skill_ids: List[int], auto_pick: int) -> List[int]:
    if skill_ids:
        return list(dict.fromkeys([int(s) for s in skill_ids]))  # 重複除去し順序維持
    # 観測数の多いスキル上位から自動選択
    s = (df[df["user_id"]==user_id]
         .groupby("skill_id").size()
         .sort_values(ascending=False).head(auto_pick).index.tolist())
    return [int(x) for x in s]

def build_items_pool(df: pd.DataFrame, skills: List[int], seed: int, per_skill: int, items_csv: str=None) -> pd.DataFrame:
    if items_csv and os.path.exists(items_csv):
        items_df = pd.read_csv(items_csv)
        needed = items_df[items_df["skill_id"].isin(skills)][["item_id","skill_id"]].drop_duplicates()
        return needed
    # データ内の item_id, skill_id から作る（簡易プール）
    base = df[df["skill_id"].isin(skills)][["item_id","skill_id"]].drop_duplicates()
    rs = np.random.RandomState(seed)
    chunks = []
    for sid in skills:
        sub = base[base["skill_id"]==sid]
        if sub.empty:
            continue
        k = min(per_skill, sub.shape[0])
        pick_idx = rs.choice(sub.index, size=k, replace=False)
        chunks.append(sub.loc[pick_idx])
    if not chunks:
        return pd.DataFrame(columns=["item_id","skill_id"])
    return pd.concat(chunks, ignore_index=True)

# ========== メイン ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="one_200_50_500_50000.csv", help="学習履歴CSV（単一スキル行）")
    ap.add_argument("--nrows", type=int, default=None, help="読み込む先頭行数（例: 50000）")
    ap.add_argument("--user_id", type=int, required=True, help="対象ユーザーID")
    ap.add_argument("--skill_ids", nargs="*", help="対象スキル（空白またはカンマ区切り可: 3 7 21 / 3,7,21）")
    ap.add_argument("--auto_pick_skills", type=int, default=3, help="skill_ids未指定時に自動で選ぶ上位スキル数")
    ap.add_argument("--items_csv", default=None, help="候補アイテムCSV（列: item_id, skill_id）。未指定なら履歴から作成")
    ap.add_argument("--per_skill_items", type=int, default=6, help="スキルごとに拾う候補数（items_csv未指定時）")
    ap.add_argument("--seed", type=int, default=0, help="候補サンプリングの乱数シード")
    # ランキング・出力制御
    ap.add_argument("--sweet", type=float, default=0.7, help="目標の次回正答確率")
    ap.add_argument("--w_explore", type=float, default=0.15, help="探索ボーナス係数")
    ap.add_argument("--topn", type=int, default=5, help="LLMに渡す候補の最大件数")
    ap.add_argument("--print_report", action="store_true", help="ユーザー×スキルのBKTレポートを表示")
    ap.add_argument("--export_llm_payload", default=None, help="UserペイロードJSONを保存するパス（未指定なら保存しない）")
    ap.add_argument("--mock_llm", action="store_true", help="モックのLLM出力JSONも表示する")
    args = ap.parse_args()

    # skill_ids のパース（空白/カンマ混在対応）
    skill_ids = []
    if args.skill_ids:
        for tok in args.skill_ids:
            for s in str(tok).split(","):
                s = s.strip()
                if s:
                    skill_ids.append(int(s))

    # 1) データ読み込み
    df = load_single_skill_df(args.csv, nrows=args.nrows)

    # 2) スキル選定
    skills = pick_skills_for_user(df, args.user_id, skill_ids, args.auto_pick_skills)
    if not skills:
        raise SystemExit("指定ユーザーに対応するスキルが見つかりません。")

    # 3) パラメータ推定
    params_by_skill = fit_by_skill(df, skills)

    # 4) ユーザー×各スキルのBKT出力
    rows = []
    for sid in skills:
        rep = predict_user_skill(df, params_by_skill, args.user_id, sid)
        rows.append(rep)
    report_df = pd.concat(rows, ignore_index=True)
    # 表示（任意）
    if args.print_report:
        print("# BKT Report (user×skills)")
        print(report_df.to_string(index=False))
        print()

    # 5) 候補アイテムプール
    items_df = build_items_pool(df, skills, seed=args.seed, per_skill=args.per_skill_items, items_csv=args.items_csv)
    if items_df.empty:
        raise SystemExit("候補アイテムが見つかりません。items_csvを指定するか履歴を確認してください。")

    # 6) ランキング
    ranked = rank_candidates(report_df, items_df, args.user_id, sweet=args.sweet,
                             w_explore=args.w_explore, topn=args.topn)
    print("# Ranked Candidates (for LLM)")
    if ranked.empty:
        print("(empty)")
        return
    print(ranked.to_string(index=False))
    print()

    # 7) LLMメッセージ生成
    system_msg, user_msg, schema_hint, user_payload = build_llm_messages(args.user_id, ranked)
    print("--- System Message ---")
    print(system_msg)
    print("\n--- User Message (payload) ---")
    print(user_msg)
    print("\n--- Schema Hint ---")
    print(schema_hint)

    # 8) 保存オプション
    if args.export_llm_payload:
        with open(args.export_llm_payload, "w", encoding="utf-8") as f:
            json.dump(user_payload, f, ensure_ascii=False, indent=2)
        print(f"\n[Saved] LLM user payload -> {args.export_llm_payload}")

    # 9) モック応答（任意）
    if args.mock_llm:
        mock_json = build_mock_llm_output(args.user_id, ranked)
        print("\n--- Mock LLM Output JSON ---")
        print(mock_json)

if __name__ == "__main__":
    main()
