#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bkt_llm_bridge_v3_2.py  (adds learner_profile exposure)

- --expose_profile {none,summary,full} で学習者状態をLLMに共有可能
- payloadに "learner_profile" を追加（出力スキーマは従来通り）
- それ以外の機能は v3.1 と同等（重複排除・配分保証・難易度タグ補正等）

依存: pandas, numpy
"""

import argparse, os, json, numpy as np, pandas as pd
from typing import List, Dict, Optional

# ===== BKT（同じ） =====
def bkt_em(sequences: List[List[int]], max_iter=30, tol=1e-4, init=None) -> Dict[str, float]:
    if init is None: p_L0, p_T, p_S, p_G = 0.2, 0.1, 0.1, 0.2
    else:            p_L0, p_T, p_S, p_G = init
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
            if Tn == 0: continue
            pi = np.array([1-p_L0, p_L0])
            def b(o): return np.array([p_G if o==1 else (1-p_G),(1-p_S) if o==1 else p_S])
            alpha = np.zeros((Tn,2)); scale = np.zeros(Tn)
            alpha[0] = pi * b(obs[0]); scale[0] = alpha[0].sum()+1e-300; alpha[0] /= scale[0]
            for t in range(1, Tn):
                alpha[t] = (alpha[t-1].dot(A)) * b(obs[t]); scale[t] = alpha[t].sum()+1e-300; alpha[t] /= scale[t]
            beta = np.zeros((Tn,2)); beta[-1] = 1.0 / scale[-1]
            for t in range(Tn-2, -1, -1):
                bt1 = b(obs[t+1]); beta[t] = (A * (bt1 * beta[t+1])).sum(axis=1); beta[t] /= scale[t]
            gamma = alpha * beta; gamma /= (gamma.sum(axis=1, keepdims=True)+1e-300)
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
            total_ll += float(np.sum(np.log(scale+1e-300)))
            sum_gamma_L1 += gamma[0,1]
        nseq = max(len(sequences),1)
        p_L0 = clip(sum_gamma_L1 / nseq)
        p_T  = clip(sum_xi_U2L   / max(sum_gamma_U,1e-12))
        p_S  = clip(sum_emit_L_incorrect / max(sum_emit_L,1e-12))
        p_G  = clip(sum_emit_U_correct   / max(sum_emit_U,1e-12))
        it_used = it
        if last_ll is not None and abs(total_ll-last_ll) < tol: break
        last_ll = total_ll
    return {"p_L0":p_L0,"p_T":p_T,"p_S":p_S,"p_G":p_G,"loglik":last_ll if last_ll is not None else float('nan'),"iters":it_used}

def fit_by_skill(df_single: pd.DataFrame, skills: List[int]) -> Dict[int, Dict[str,float]]:
    out={}
    for sid in skills:
        sdf = df_single[df_single["skill_id"]==sid]
        seqs = [udf.sort_values("date")["result"].astype(int).tolist() for _,udf in sdf.groupby("user_id")]
        if len(seqs)==0: continue
        out[int(sid)] = bkt_em(seqs, max_iter=30)
    return out

def bkt_predict_next(obs: List[int], p: Dict[str,float]):
    pL = p["p_L0"]
    for o in obs:
        pL = pL + (1 - pL) * p["p_T"]
        num = ((1 - p["p_S"]) if o==1 else p["p_S"]) * pL
        den = num + ((p["p_G"]) if o==1 else (1 - p["p_G"])) * (1 - pL)
        pL = pL if den<=0 else num/den
    pL_next = pL + (1 - pL) * p["p_T"]
    p_corr  = (1 - p["p_S"]) * pL_next + p["p_G"] * (1 - pL_next)
    return pL, pL_next, p_corr

def predict_user_skill(df_single, params_by_skill, user_id, skill_id) -> pd.DataFrame:
    sdf = df_single[(df_single["user_id"]==user_id) & (df_single["skill_id"]==skill_id)]
    obs = sdf.sort_values("date")["result"].astype(int).tolist()
    p   = params_by_skill.get(int(skill_id)) or {"p_L0":0.2,"p_T":0.1,"p_S":0.1,"p_G":0.2,"loglik":float("nan"),"iters":0}
    pL, pL_next, p_corr = bkt_predict_next(obs, p)
    return pd.DataFrame([{"user_id":int(user_id), "skill_id":int(skill_id), "n_obs":len(obs),
                          "P(L)_posterior":pL, "P(L)_after_learn":pL_next, "P(next correct)":p_corr, **p}])

# ===== Data Utils =====
def load_single_skill_df(csv_path, nrows=None):
    df = pd.read_csv(csv_path, nrows=nrows)
    rn={}
    if "skill_id" in df.columns and "skill" not in df.columns: rn["skill_id"]="skill"
    if "correct"  in df.columns and "result" not in df.columns: rn["correct"]="result"
    if rn: df = df.rename(columns=rn)
    df["date"] = pd.to_datetime(df["date"])
    df["skill_id"] = df["skill"].astype(int)
    return df.sort_values(["user_id","skill_id","date"]).reset_index(drop=True)

def parse_tags(val: Optional[str]) -> List[str]:
    if pd.isna(val): return []
    s=str(val).strip()
    return [t.strip() for t in s.split(",") if t.strip()] if s else []

def build_items_pool(df, skills, seed, per_skill, items_csv=None):
    if items_csv and os.path.exists(items_csv):
        items = pd.read_csv(items_csv)
        items = items[items["skill_id"].isin(skills)].copy()
        for col, default in [("kc_tags",""),("engagement",0.5),("est_sec",60),("difficulty_tag","med")]:
            if col not in items.columns: items[col]=default
        cols=["item_id","skill_id","kc_tags","engagement","est_sec","difficulty_tag"]
        cols += [c for c in ["a","b","c","bloom"] if c in items.columns]
        return items[cols].drop_duplicates()
    base = df[df["skill_id"].isin(skills)][["item_id","skill_id"]].drop_duplicates()
    rs=np.random.RandomState(seed); chunks=[]
    for sid in skills:
        sub=base[base["skill_id"]==sid]
        if sub.empty: continue
        k=min(per_skill,len(sub))
        take=sub.sample(k, random_state=rs).assign(kc_tags="", engagement=0.5, est_sec=60, difficulty_tag="med")
        chunks.append(take)
    return pd.concat(chunks, ignore_index=True) if chunks else base

# ===== Scoring =====
def score_teach(pnext,n_obs,pT,pL,item_row,goal_skills:set,goal_tags:set,sweet=0.65):
    sid=int(item_row["skill_id"]); tags=set(parse_tags(item_row.get("kc_tags","")))
    coverage=0.0
    if sid in goal_skills: coverage+=1.0
    if goal_tags: coverage+=0.5*sum(1 for t in tags if t in goal_tags)
    deficit=(1.0-pL); learn=pT*(1.0-pL); sweet_term=-abs(pnext-sweet); explore=0.10/np.sqrt(n_obs+1.0)
    return 1.2*coverage + 0.8*deficit + 0.6*learn + 0.3*sweet_term + explore

def score_motivate(pnext,n_obs,pT,pL,item_row,sweet=0.72):
    engagement=float(item_row.get("engagement",0.5)); est_sec=float(item_row.get("est_sec",60.0))
    sweet_term=-abs(pnext-sweet); time_term=-(est_sec/120.0); explore=0.20/np.sqrt(n_obs+1.0)
    return 1.0*sweet_term + 0.6*engagement + 0.2*explore + 0.2*pT + 0.2*(-time_term)

# ===== Difficulty tag =====
def parse_tag_deltas(s: str): 
    vals=[float(x) for x in str(s).split(",")]
    if len(vals)!=3: raise ValueError("--tag_deltas は easy,med,hard の3要素（例 +0.12,0,-0.12）")
    return {"easy":vals[0],"med":vals[1],"hard":vals[2]}

def p_item_from_tag(pnext, tag, deltas): 
    delta=deltas.get(str(tag).lower(),0.0)
    return float(np.clip(pnext+delta,1e-3,0.999))

# ===== Ranking =====
def rank_candidates(report_df, items_df, user_id, mode="motivate",
                    goal_skills=None, goal_tags=None,
                    sweet_motivate=0.72, sweet_teach=0.65, topn=12, alpha=0.5,
                    difficulty_mode="tag", tag_deltas=None):
    rep = report_df[report_df["user_id"]==user_id].set_index("skill_id")
    gskills=set(int(x) for x in (goal_skills or [])); gtags=set(t for t in (goal_tags or []) if t)
    tag_deltas=tag_deltas or {"easy":0.12,"med":0.0,"hard":-0.12}
    rows=[]
    for _,it in items_df.iterrows():
        sid=int(it["skill_id"])
        if sid not in rep.index: continue
        r=rep.loc[sid]
        pnext=float(r["P(next correct)"]); n_obs=float(r["n_obs"]); pT=float(r["p_T"]); pL=float(r["P(L)_posterior"])
        if mode=="teach": score=score_teach(pnext,n_obs,pT,pL,it,gskills,gtags,sweet=sweet_teach)
        elif mode=="motivate": score=score_motivate(pnext,n_obs,pT,pL,it,sweet=sweet_motivate)
        else:
            sT=score_teach(pnext,n_obs,pT,pL,it,gskills,gtags,sweet=sweet_teach)
            sM=score_motivate(pnext,n_obs,pT,pL,it,sweet=sweet_motivate)
            score=alpha*sT+(1.0-alpha)*sM
        p_for_llm = p_item_from_tag(pnext, it.get("difficulty_tag","med"), tag_deltas) if difficulty_mode=="tag" else pnext
        rows.append({"item_id":str(it["item_id"]), "skill_id":sid, "pnext":pnext, "p_for_llm":p_for_llm, "score":score})
    cand=pd.DataFrame(rows)
    if cand.empty: return cand
    return cand.sort_values("score", ascending=False).head(topn)

# ===== Mix & diversity =====
def enforce_mix(ranked_df, mix_str="3,1,1", review_thr=0.40, challenge_thr=0.85, center=0.70, prefer_unique_skills=True):
    need_p,need_r,need_c=[int(x) for x in mix_str.split(",")]
    total_need=need_p+need_r+need_c
    df=ranked_df.copy()
    if "row_id" not in df.columns: df=df.reset_index().rename(columns={"index":"row_id"})
    rev  = df[df["p_for_llm"]<review_thr].copy()
    prac = df[(df["p_for_llm"]>=review_thr)&(df["p_for_llm"]<challenge_thr)].copy()
    chal = df[df["p_for_llm"]>=challenge_thr].copy()
    prac = prac.assign(dist=(prac["p_for_llm"]-center).abs()).sort_values(["dist","score"], ascending=[True,False])
    rev  = rev.sort_values("score", ascending=False)
    chal = chal.sort_values("score", ascending=False)
    used_rowids, used_keys, used_skills = set(), set(), set()
    def pick(block,k,prefer_unique):
        out=[]
        if k<=0 or block.empty: return out
        for _,row in block.iterrows():
            rid=int(row["row_id"]); key=(str(row["item_id"]),int(row["skill_id"])); sid=int(row["skill_id"])
            if rid in used_rowids or key in used_keys: continue
            if prefer_unique and sid in used_skills: continue
            out.append(row); used_rowids.add(rid); used_keys.add(key); used_skills.add(sid)
            if len(out)>=k: break
        return out
    chosen=[]
    chosen+=pick(prac,need_p,prefer_unique_skills)
    chosen+=pick(rev, need_r,prefer_unique_skills)
    chosen+=pick(chal,need_c,prefer_unique_skills)
    if len(chosen)<total_need:
        frames=[x for x in [prac,rev,chal] if not x.empty]
        remain=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=df.columns)
        if not remain.empty:
            remain=remain[~remain["row_id"].isin(used_rowids)].copy()
            remain=remain.assign(dist=(remain["p_for_llm"]-center).abs()).sort_values(["dist","score"], ascending=[True,False])
            for _,row in remain.iterrows():
                if len(chosen)>=total_need: break
                rid=int(row["row_id"]); key=(str(row["item_id"]),int(row["skill_id"])); sid=int(row["skill_id"])
                if rid in used_rowids or key in used_keys: continue
                if prefer_unique_skills and sid in used_skills: continue
                chosen.append(row); used_rowids.add(rid); used_keys.add(key); used_skills.add(sid)
    if not chosen: return pd.DataFrame(columns=df.columns)
    out=pd.DataFrame(chosen).reset_index(drop=True)
    if "dist" not in out.columns: out["dist"]=np.nan
    if len(out)>total_need: out=out.iloc[:total_need].copy()
    return out

# ===== Learner Profile (NEW) =====
def build_learner_profile(df_history: pd.DataFrame, report_df: pd.DataFrame, user_id: int, level: str, candidate_skills: List[int]):
    rep = report_df[report_df["user_id"]==user_id].copy()
    rep = rep[rep["skill_id"].isin(candidate_skills)] if candidate_skills else rep
    prof = {"level": level, "note": "Use only for phrasing feedback; do not recompute probabilities."}
    if rep.empty:
        prof["summary"] = {"avg_P_next": None, "avg_P_L": None}
        return prof
    # summary
    prof["summary"] = {
        "skills_considered": sorted(map(int, rep["skill_id"].unique())),
        "avg_P_next": float(rep["P(next correct)"].mean()),
        "avg_P_L": float(rep["P(L)_posterior"].mean()),
        "low_mastery_skills": sorted(map(int, rep.loc[rep["P(L)_posterior"]<0.4,"skill_id"].unique())),
        "data_scarce_skills": sorted(map(int, rep.loc[rep["n_obs"]<5,"skill_id"].unique())),
        "slip_risk_skills":   sorted(map(int, rep.loc[rep["p_S"]>0.3,"skill_id"].unique())),
        "guess_risk_skills":  sorted(map(int, rep.loc[rep["p_G"]>0.3,"skill_id"].unique())),
    }
    if level=="full":
        per={}
        for r in rep.itertuples(index=False):
            per[int(r.skill_id)] = {
                "n_obs": int(r.n_obs),
                "P_L_posterior": round(float(r._3), 3) if hasattr(r,'_3') else round(float(r._asdict()["P(L)_posterior"]),3),
                "P_next": round(float(r._5), 3) if hasattr(r,'_5') else round(float(r._asdict()["P(next correct)"]),3),
                "p_T": round(float(r.p_T),3),
                "p_S": round(float(r.p_S),3),
                "p_G": round(float(r.p_G),3)
            }
        prof["per_skill"] = per
    return prof

# ===== LLM messages =====
def build_llm_messages(user_id, ranked_df, mode="motivate", goals=None, language="ja", learner_profile=None):
    candidates=[{"item_id":str(r.item_id),"skill_id":int(r.skill_id),"expected_p_correct":round(float(r.p_for_llm),2)} for r in ranked_df.itertuples(index=False)]
    system=("あなたは教育向けの推薦アシスタントです。指定された候補問題リストからのみ選び、"
            "出力は必ず指定のJSONスキーマに厳密に従って日本語で返してください。候補外のIDは使用禁止です。"
            "助言は短く具体的に、数値は小数第2位まで。"
            "learner_profile はフィードバックや理由の表現調整のみに用い、確率の再計算や候補の改変には用いないこと。")
    style_hint=({"teach":"学習目標の達成に直結する理由を書き、必要なら前提の再確認を促す。",
                 "motivate":"成功体験を積ませる口調で短く励ます。次の一歩を具体的に指示。",
                 "hybrid":"目標達成と成功体験のバランスを意識して理由付けをする."}[mode])
    user_payload={
        "task":"次に解くべき問題の推薦と、学習者向けフィードバック文の生成",
        "objective_mode":mode,
        "user_id":user_id,
        "goals":goals or {},
        "policy":{"selection_rule":"候補から最大5問。expected_p_correctが0.6〜0.8に近いものを優先。高すぎる(>0.9)は除外、低すぎる(<0.4)はreview扱い。",
                  "types":["review","practice","challenge"],
                  "style":{"student_msg_len":"<=200文字","teacher_note_len":"<=100文字","lang":language,"tone_hint":style_hint}},
        "candidates":candidates
    }
    if learner_profile is not None:
        user_payload["learner_profile"]=learner_profile
    schema_hint=('必ず次のJSONのみを返すこと:\n'
                 '{ "schema_version":"1.0", "user_id":<int>, '
                 '"recommendations":[{"item_id":"<id>","skill_id":<int>,"reason":"<短文>",'
                 '"expected_p_correct":<num>,"type":"review|practice|challenge"}],'
                 '"feedback_to_student":"<200文字以内>", "note_to_teacher":"<100文字以内>", "next_checkin_days":<int> }')
    return system, json.dumps(user_payload, ensure_ascii=False, indent=2), schema_hint, user_payload

def build_mock_llm_output(user_id, ranked_df, mode="motivate")->str:
    recs=[]
    for r in ranked_df.itertuples(index=False):
        p=float(r.p_for_llm); t="practice"
        if p>=0.85: t="challenge"
        if p<=0.50: t="review"
        reason=f"P(next correct)={round(p,2)}で目標帯に近い。"
        if mode=="teach": reason+=" 目標知識の定着に有効。"
        if mode=="motivate": reason+=" 成功体験を得やすい難易度。"
        recs.append({"item_id":str(r.item_id),"skill_id":int(r.skill_id),"reason":reason,"expected_p_correct":round(p,2),"type":t})
    out={"schema_version":"1.0","user_id":int(user_id),"recommendations":recs,
         "feedback_to_student":"この調子！次は達成できそうな一問に挑戦しましょう。解法の根拠を声に出して説明し、できたら似た問題を1問だけ復習してください。",
         "note_to_teacher":"目的モードに基づき推薦を生成。閾値は0.6〜0.8帯を中心。","next_checkin_days":2}
    return json.dumps(out, ensure_ascii=False, indent=2)

# ===== Main =====
def pick_skills_for_user(df, user_id, skill_ids, auto_pick)->List[int]:
    if skill_ids:
        seen=set(); out=[]
        for s in skill_ids:
            for tok in str(s).split(","):
                tok=tok.strip()
                if not tok: continue
                v=int(tok)
                if v in seen: continue
                seen.add(v); out.append(v)
        return out
    s=(df[df["user_id"]==user_id].groupby("skill_id").size().sort_values(ascending=False).head(auto_pick).index.tolist())
    return [int(x) for x in s]

def main():
    ap=argparse.ArgumentParser()
    # 入力
    ap.add_argument("--csv", default="one_200_50_500_50000.csv")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--user_id", type=int, required=True)
    ap.add_argument("--skill_ids", nargs="*")
    ap.add_argument("--auto_pick_skills", type=int, default=3)
    ap.add_argument("--items_csv", default=None)
    ap.add_argument("--per_skill_items", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    # 目的
    ap.add_argument("--mode", choices=["teach","motivate","hybrid"], default="motivate")
    ap.add_argument("--goal_skills", nargs="*")
    ap.add_argument("--goal_tags", nargs="*")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--sweet_motivate", type=float, default=0.72)
    ap.add_argument("--sweet_teach", type=float, default=0.65)
    # 難易度
    ap.add_argument("--difficulty_mode", choices=["none","tag"], default="tag")
    ap.add_argument("--tag_deltas", default="+0.12,0,-0.12")
    # 配分
    ap.add_argument("--mix", default="3,1,1")
    ap.add_argument("--review_thr", type=float, default=0.40)
    ap.add_argument("--challenge_thr", type=float, default=0.85)
    ap.add_argument("--center", type=float, default=0.70)
    ap.add_argument("--prefer_unique_skills", action="store_true")
    # 出力
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--print_report", action="store_true")
    ap.add_argument("--export_llm_payload", default=None)
    ap.add_argument("--export_ranked", default=None)
    ap.add_argument("--mock_llm", action="store_true")
    # NEW: 学習者パラメータの露出レベル
    ap.add_argument("--expose_profile", choices=["none","summary","full"], default="summary",
                    help="LLMに渡す learner_profile の粒度。none:渡さない / summary:要約のみ / full:候補スキルのBKT数値付き")
    args=ap.parse_args()

    # 1) 履歴
    df=load_single_skill_df(args.csv, nrows=args.nrows)
    # 2) スキル
    skills=pick_skills_for_user(df, args.user_id, args.skill_ids, args.auto_pick_skills)
    if not skills: raise SystemExit("指定ユーザーの対象スキルが見つかりません。")
    # 3) BKT
    params=fit_by_skill(df, skills)
    # 4) レポート
    report_df=pd.concat([predict_user_skill(df, params, args.user_id, sid) for sid in skills], ignore_index=True)
    if args.print_report:
        print("# BKT Report (user×skills)"); print(report_df.to_string(index=False)); print()
    # 5) 候補
    items_df=build_items_pool(df, skills, seed=args.seed, per_skill=args.per_skill_items, items_csv=args.items_csv)
    if items_df.empty: raise SystemExit("候補アイテムが見つかりません。")
    # 6) 目標
    def _parse_mixed(lst):
        out=[]; 
        if not lst: return out
        for tok in lst:
            for s in str(tok).split(","):
                s=s.strip(); 
                if s: out.append(s)
        return out
    goal_skills=[int(x) for x in _parse_mixed(args.goal_skills)]
    goal_tags=_parse_mixed(args.goal_tags)
    # 7) ランキング
    deltas=parse_tag_deltas(args.tag_deltas)
    ranked0=rank_candidates(report_df, items_df, args.user_id, mode=args.mode,
                            goal_skills=goal_skills, goal_tags=goal_tags,
                            sweet_motivate=args.sweet_motivate, sweet_teach=args.sweet_teach,
                            topn=args.topn, alpha=args.alpha,
                            difficulty_mode=args.difficulty_mode, tag_deltas=deltas)
    if ranked0.empty: raise SystemExit("ランキング候補が空です。データや条件を確認してください。")
    ranked0=ranked0.reset_index().rename(columns={"index":"row_id"})
    # 8) 配分
    ranked=enforce_mix(ranked_df=ranked0, mix_str=args.mix,
                       review_thr=args.review_thr, challenge_thr=args.challenge_thr,
                       center=args.center, prefer_unique_skills=args.prefer_unique_skills)
    print(f"# Ranked Candidates after enforce_mix (mode={args.mode}, mix={args.mix})")
    print(ranked.to_string(index=False) if not ranked.empty else "(empty)"); print()
    if args.export_ranked:
        ranked.to_json(args.export_ranked, orient="records", force_ascii=False, indent=2)
        print(f"[Saved] ranked candidates -> {args.export_ranked}")
    # 9) Learner Profile（必要なら作成）
    learner_profile=None
    if args.expose_profile!="none":
        cand_skills=sorted(set(int(s) for s in ranked["skill_id"].unique()))
        learner_profile=build_learner_profile(df, report_df, args.user_id, args.expose_profile, cand_skills)
    # 10) LLMメッセージ
    goals_hint={"skills":goal_skills,"tags":goal_tags}
    system_msg, user_msg, schema_hint, payload = build_llm_messages(args.user_id, ranked, mode=args.mode, goals=goals_hint, learner_profile=learner_profile)
    print("--- System Message ---"); print(system_msg)
    print("\n--- User Message (payload) ---"); print(user_msg)
    print("\n--- Schema Hint ---"); print(schema_hint)
    if args.export_llm_payload:
        with open(args.export_llm_payload,"w",encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[Saved] LLM user payload -> {args.export_llm_payload}")
    # 11) モック
    if args.mock_llm:
        print("\n--- Mock LLM Output JSON ---"); print(build_mock_llm_output(args.user_id, ranked, mode=args.mode))

if __name__=="__main__":
    main()
