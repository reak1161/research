#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bkt_llm_bridge_v2.py
- BKTで user×skill の学習度と次回正答確率を推定
- 目的に応じた2モードの推薦:
  * teach: 教員の到達目標（skill/tags）を優先
  * motivate: ちょうど良い難易度×エンゲージメントを優先
  * hybrid: teachとmotivateの線形合成
- 候補アイテム(item_id, skill_id[, kc_tags, engagement, est_sec, bloom, a,b,c...])からランキング
- LLMに渡すメッセージとペイロードJSONを生成。外部API呼び出しは行わない。

依存: pandas, numpy
学習履歴CSV: 列 user_id, item_id, skill, result, date （列名ゆれは自動補正）
"""

import argparse, os, json, math, numpy as np, pandas as pd
from typing import List, Dict, Optional

# ========= 基本BKT =========
def bkt_em(sequences: List[List[int]], max_iter=30, tol=1e-4, init=None) -> Dict[str, float]:
    if init is None:
        p_L0, p_T, p_S, p_G = 0.2, 0.1, 0.1, 0.2
    else:
        p_L0, p_T, p_S, p_G = init
    def clip(x): return float(np.clip(x, 1e-5, 1-1e-5))
    last_ll, it_used = None, 0
    for it in range(1, max_iter+1):
        A = np.array([[1-p_T, p_T],[0.0,1.0]])
        sum_gamma_L1 = sum_xi_U2L = sum_gamma_U = 0.0
        sum_emit_L = sum_emit_L_incorrect = 0.0
        sum_emit_U = sum_emit_U_correct = 0.0
        total_ll = 0.0
        for obs in sequences:
            Tn = len(obs)
            if Tn == 0: 
                continue
            pi = np.array([1-p_L0, p_L0])
            def b(o):
                return np.array([p_G if o==1 else (1-p_G),
                                 (1-p_S) if o==1 else p_S])
            alpha = np.zeros((Tn,2)); scale = np.zeros(Tn)
            alpha[0] = pi * b(obs[0]); scale[0] = alpha[0].sum() + 1e-300; alpha[0] /= scale[0]
            for t in range(1, Tn):
                alpha[t] = (alpha[t-1].dot(A)) * b(obs[t])
                scale[t] = alpha[t].sum() + 1e-300; alpha[t] /= scale[t]
            beta = np.zeros((Tn,2)); beta[-1] = 1.0 / scale[-1]
            for t in range(Tn-2, -1, -1):
                bt1 = b(obs[t+1])
                beta[t] = (A * (bt1 * beta[t+1])).sum(axis=1)
                beta[t] /= scale[t]
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)
            for t in range(Tn-1):
                bt1 = b(obs[t+1])
                num = alpha[t,0]*p_T*bt1[1]*beta[t+1,1]
                den = (alpha[t,0]*(1-p_T)*bt1[0]*beta[t+1,0] +
                       alpha[t,0]*p_T    *bt1[1]*beta[t+1,1] +
                       alpha[t,1]*1.0    *bt1[1]*beta[t+1,1])
                sum_xi_U2L += 0.0 if den<=0 else num/den
                sum_gamma_U += gamma[t,0]
            sum_emit_L += gamma[:,1].sum()
            sum_emit_L_incorrect += ((1-np.array(obs))*gamma[:,1]).sum()
            sum_emit_U += gamma[:,0].sum()
            sum_emit_U_correct += (np.array(obs)*gamma[:,0]).sum()
            total_ll += float(np.sum(np.log(scale + 1e-300)))
            sum_gamma_L1 += gamma[0,1]
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

def bkt_predict_next(obs: List[int], p: Dict[str,float]):
    pL = p["p_L0"]
    for o in obs:
        pL = pL + (1 - pL) * p["p_T"]
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
    return pd.DataFrame([{
        "user_id":int(user_id), "skill_id":int(skill_id), "n_obs":len(obs),
        "P(L)_posterior":pL, "P(L)_after_learn":pL_next, "P(next correct)":p_corr, **p
    }])

# ========= データユーティリティ =========
def load_single_skill_df(csv_path: str, nrows=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows)
    rename = {}
    if "skill_id" in df.columns and "skill" not in df.columns: rename["skill_id"]="skill"
    if "correct"  in df.columns and "result" not in df.columns: rename["correct"]="result"
    if rename: df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"])
    df["skill_id"] = df["skill"].astype(int)
    return df.sort_values(["user_id","skill_id","date"]).reset_index(drop=True)

def parse_tags(val: Optional[str]) -> List[str]:
    if pd.isna(val): return []
    s = str(val).strip()
    if not s: return []
    return [t.strip() for t in s.split(",") if t.strip()]

def build_items_pool(df: pd.DataFrame, skills: List[int], seed: int, per_skill: int, items_csv: str=None) -> pd.DataFrame:
    """
    items_csv があれば優先して使う。期待列:
      - item_id, skill_id
      - （任意）kc_tags, engagement(0-1), est_sec(秒), bloom
      - （任意）a,b,c (IRT)
    無い列はデフォルト補完。
    """
    if items_csv and os.path.exists(items_csv):
        items = pd.read_csv(items_csv)
        items = items[items["skill_id"].isin(skills)].copy()
        if "kc_tags" not in items.columns: items["kc_tags"] = ""
        if "engagement" not in items.columns: items["engagement"] = 0.5
        if "est_sec" not in items.columns: items["est_sec"] = 60
        return items[["item_id","skill_id","kc_tags","engagement","est_sec"] + 
                     [c for c in ["a","b","c","bloom"] if c in items.columns]
                    ].drop_duplicates()
    # メタが無い場合は履歴から最小限生成
    base = df[df["skill_id"].isin(skills)][["item_id","skill_id"]].drop_duplicates()
    rs = np.random.RandomState(seed)
    chunks = []
    for sid in skills:
        sub = base[base["skill_id"]==sid]
        if sub.empty: continue
        k = min(per_skill, len(sub))
        take = sub.sample(k, random_state=rs)
        take = take.assign(kc_tags="", engagement=0.5, est_sec=60)
        chunks.append(take)
    return pd.concat(chunks, ignore_index=True) if chunks else base

# ========= 推薦スコア（teach/motivate/hybrid） =========
def score_teach(pnext, n_obs, pT, pL_post, item_row, goal_skills:set, goal_tags:set,
                sweet=0.65):
    # 到達目標の達成を優先：目標KC/skillのカバー、未習得度、学習遷移、やや挑戦的な難易度
    sid = int(item_row["skill_id"])
    tags = set(parse_tags(item_row.get("kc_tags","")))
    coverage = 0.0
    if sid in goal_skills: coverage += 1.0
    if goal_tags: coverage += 0.5 * sum(1 for t in tags if t in goal_tags)
    deficit = (1.0 - pL_post)                 # 未習得度
    learn_gain = pT * (1.0 - pL_post)         # 学習余地
    sweet_term = -abs(pnext - sweet)          # ほどよい難易度（やや控えめ）
    explore = 0.10/np.sqrt(n_obs+1.0)         # 観測が少ない所へ軽い探索
    return (1.2*coverage) + (0.8*deficit) + (0.6*learn_gain) + (0.3*sweet_term) + explore

def score_motivate(pnext, n_obs, pT, pL_post, item_row, sweet=0.72):
    # モチベ重視：ジャスト難易度、短時間、エンゲージメント高め、少しのバラエティ
    engagement = float(item_row.get("engagement", 0.5))
    est_sec    = float(item_row.get("est_sec", 60.0))
    sweet_term = -abs(pnext - sweet)
    time_term  = - (est_sec/120.0)            # 短いほど良い（~2分基準）
    explore    = 0.20/np.sqrt(n_obs+1.0)
    return (1.0*sweet_term) + (0.6*engagement) + (0.2*explore) + (0.2*pT) + (0.2*(-time_term))

def rank_candidates(report_df: pd.DataFrame, items_df: pd.DataFrame, user_id: int,
                    mode="motivate", goal_skills: List[int]=None, goal_tags: List[str]=None,
                    sweet_motivate=0.72, sweet_teach=0.65, topn=5, alpha=0.5) -> pd.DataFrame:
    """
    mode:
      - 'teach'    : 教員目標に特化
      - 'motivate' : ちょうどよい難易度・体験重視
      - 'hybrid'   : teach と motivate の線形合成（alphaで重み付け）
    """
    rep = report_df[report_df["user_id"]==user_id].set_index("skill_id")
    gskills = set(int(x) for x in (goal_skills or []))
    gtags   = set(t for t in (goal_tags or []) if t)

    rows = []
    for _, it in items_df.iterrows():
        sid = int(it["skill_id"])
        if sid not in rep.index: continue
        r = rep.loc[sid]
        pnext = float(r["P(next correct)"])
        n_obs = float(r["n_obs"]); pT = float(r["p_T"]); pL = float(r["P(L)_posterior"])

        if mode == "teach":
            score = score_teach(pnext, n_obs, pT, pL, it, gskills, gtags, sweet=sweet_teach)
        elif mode == "motivate":
            score = score_motivate(pnext, n_obs, pT, pL, it, sweet=sweet_motivate)
        else:  # hybrid
            s_teach = score_teach(pnext, n_obs, pT, pL, it, gskills, gtags, sweet=sweet_teach)
            s_moti  = score_motivate(pnext, n_obs, pT, pL, it, sweet=sweet_motivate)
            score   = alpha * s_teach + (1.0 - alpha) * s_moti

        rows.append({
            "item_id": str(it["item_id"]),
            "skill_id": sid,
            "pnext": pnext,
            "score": score
        })
    cand = pd.DataFrame(rows)
    if cand.empty: return cand
    return cand.sort_values("score", ascending=False).head(topn)

# ========= LLMメッセージ =========
def build_llm_messages(user_id: int, ranked_df: pd.DataFrame, mode="motivate",
                       goals=None, language="ja"):
    candidates = [
        {"item_id": str(r.item_id), "skill_id": int(r.skill_id),
         "expected_p_correct": round(float(r.pnext), 2)}
        for r in ranked_df.itertuples(index=False)
    ]
    goal_hint = goals or {}
    system = (
        "あなたは教育向けの推薦アシスタントです。指定された候補問題リストからのみ選び、"
        "出力は必ず指定のJSONスキーマに厳密に従って日本語で返してください。候補外のIDは使用禁止です。"
        "助言は短く具体的に、数値は小数第2位まで。"
    )
    # 目的に応じてスタイル指針を変える
    if mode == "teach":
        style_hint = "学習目標の達成に直結する理由を書き、必要なら前提の再確認を促す。"
    elif mode == "motivate":
        style_hint = "成功体験を積ませる口調で短く励ます。次の一歩を具体的に指示。"
    else:
        style_hint = "目標達成と成功体験のバランスを意識して理由付けをする。"

    user_payload = {
        "task": "次に解くべき問題の推薦と、学習者向けフィードバック文の生成",
        "objective_mode": mode,
        "user_id": user_id,
        "goals": goal_hint,  # 例: {"skills":[21,35], "tags":["分数の約分","筆算"]}
        "policy": {
            "selection_rule": ("候補から最大5問。expected_p_correctが0.6〜0.8に近いものを優先。"
                               "高すぎる(>0.9)は除外、低すぎる(<0.4)はreview扱い。"),
            "types": ["review","practice","challenge"],
            "style": {"student_msg_len":"<=200文字","teacher_note_len":"<=100文字","lang":language,
                      "tone_hint": style_hint}
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

def build_mock_llm_output(user_id: int, ranked_df: pd.DataFrame, mode="motivate") -> str:
    recs = []
    for r in ranked_df.itertuples(index=False):
        t = "practice"
        if r.pnext >= 0.85: t = "challenge"
        if r.pnext <= 0.50: t = "review"
        reason = f"P(next correct)={round(float(r.pnext),2)}で目標帯に近い。"
        if mode == "teach":
            reason += " 目標知識の定着に有効。"
        elif mode == "motivate":
            reason += " 成功体験を得やすい難易度。"
        recs.append({
            "item_id": str(r.item_id),
            "skill_id": int(r.skill_id),
            "reason": reason,
            "expected_p_correct": round(float(r.pnext),2),
            "type": t
        })
    out = {
        "schema_version": "1.0",
        "user_id": int(user_id),
        "recommendations": recs,
        "feedback_to_student": (
            "この調子！次は達成できそうな一問に挑戦しましょう。解法の根拠を声に出して説明し、"
            "できたら似た問題を1問だけ復習してください。"),
        "note_to_teacher": "目的モードに基づき推薦を生成。閾値は0.6〜0.8帯を中心。",
        "next_checkin_days": 2
    }
    return json.dumps(out, ensure_ascii=False, indent=2)

# ========= メイン =========
def pick_skills_for_user(df: pd.DataFrame, user_id: int, skill_ids: List[int], auto_pick: int) -> List[int]:
    if skill_ids:
        # 重複除去し順序維持
        seen=set(); out=[]
        for s in skill_ids:
            for tok in str(s).split(","):
                tok=tok.strip()
                if not tok: continue
                v=int(tok)
                if v in seen: continue
                seen.add(v); out.append(v)
        return out
    s = (df[df["user_id"]==user_id].groupby("skill_id").size()
         .sort_values(ascending=False).head(auto_pick).index.tolist())
    return [int(x) for x in s]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="one_200_50_500_50000.csv")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--user_id", type=int, required=True)
    ap.add_argument("--skill_ids", nargs="*")
    ap.add_argument("--auto_pick_skills", type=int, default=3)
    ap.add_argument("--items_csv", default=None,
                    help="候補アイテムCSV: item_id, skill_id[, kc_tags, engagement, est_sec, ...]")
    ap.add_argument("--per_skill_items", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    # 目的モードと目標
    ap.add_argument("--mode", choices=["teach","motivate","hybrid"], default="motivate")
    ap.add_argument("--goal_skills", nargs="*", help="教員が到達させたいskill_id（空白/カンマ混在OK）")
    ap.add_argument("--goal_tags", nargs="*", help="教員が到達させたいkcタグ（空白/カンマ混在OK）")
    ap.add_argument("--alpha", type=float, default=0.5, help="hybrid時のteach重み (0~1)")
    ap.add_argument("--sweet_motivate", type=float, default=0.72)
    ap.add_argument("--sweet_teach", type=float, default=0.65)

    # 出力
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--print_report", action="store_true")
    ap.add_argument("--export_llm_payload", default=None)
    ap.add_argument("--mock_llm", action="store_true")
    args = ap.parse_args()

    # 1) 学習履歴
    df = load_single_skill_df(args.csv, nrows=args.nrows)

    # 2) 対象スキル
    skills = pick_skills_for_user(df, args.user_id, args.skill_ids, args.auto_pick_skills)
    if not skills:
        raise SystemExit("指定ユーザーの対象スキルが見つかりません。")

    # 3) BKT推定
    params = fit_by_skill(df, skills)

    # 4) レポート生成
    rows=[]
    for sid in skills:
        rows.append(predict_user_skill(df, params, args.user_id, sid))
    report_df = pd.concat(rows, ignore_index=True)
    if args.print_report:
        print("# BKT Report (user×skills)")
        print(report_df.to_string(index=False))
        print()

    # 5) 候補アイテム
    items_df = build_items_pool(df, skills, seed=args.seed, per_skill=args.per_skill_items, items_csv=args.items_csv)
    if items_df.empty:
        raise SystemExit("候補アイテムが見つかりません。")

    # 6) 目標のパース
    def _parse_mixed(lst):
        out=[]
        if not lst: return out
        for tok in lst:
            for s in str(tok).split(","):
                s=s.strip()
                if s: out.append(s)
        return out

    goal_skills = [int(x) for x in _parse_mixed(args.goal_skills)]
    goal_tags   = _parse_mixed(args.goal_tags)

    # 7) ランキング
    ranked = rank_candidates(
        report_df, items_df, args.user_id, mode=args.mode,
        goal_skills=goal_skills, goal_tags=goal_tags,
        sweet_motivate=args.sweet_motivate, sweet_teach=args.sweet_teach,
        topn=args.topn, alpha=args.alpha
    )
    print(f"# Ranked Candidates (mode={args.mode})")
    print(ranked.to_string(index=False) if not ranked.empty else "(empty)")
    print()

    # 8) LLMメッセージ
    goals_hint = {"skills": goal_skills, "tags": goal_tags}
    system_msg, user_msg, schema_hint, payload = build_llm_messages(
        args.user_id, ranked, mode=args.mode, goals=goals_hint
    )
    print("--- System Message ---")
    print(system_msg)
    print("\n--- User Message (payload) ---")
    print(user_msg)
    print("\n--- Schema Hint ---")
    print(schema_hint)

    if args.export_llm_payload:
        with open(args.export_llm_payload, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[Saved] LLM user payload -> {args.export_llm_payload}")

    if args.mock_llm:
        mock_json = build_mock_llm_output(args.user_id, ranked, mode=args.mode)
        print("\n--- Mock LLM Output JSON ---")
        print(mock_json)

if __name__ == "__main__":
    main()
