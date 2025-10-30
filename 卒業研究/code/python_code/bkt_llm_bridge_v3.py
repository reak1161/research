#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bkt_llm_bridge_v3.py

目的：
- BKTで user×skill の学習度・次回正答確率を推定
- 候補アイテム（item_id, skill_id[, difficulty_tag, kc_tags, engagement, est_sec...]）からランキング
- 目的別モード（teach / motivate / hybrid）
- 出力配分（practice, review, challenge）を enforce_mix で保証
- 可能な範囲で skill_id の重複を避けて多様化
- アイテム難易度タグ（easy/med/hard）で per-item の expected_p_correct を補正
- LLMに渡す System/User メッセージと JSONスキーマのヒントを生成
- （任意）モックのLLM出力JSONを表示

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
                alpha[t] = (alpha[t-1].dot(A)) * b(obs[t]); scale[t] = alpha[t].sum() + 1e-300; alpha[t] /= scale[t]
            beta = np.zeros((Tn,2)); beta[-1] = 1.0 / scale[-1]
            for t in range(Tn-2, -1, -1):
                bt1 = b(obs[t+1])
                beta[t] = (A * (bt1 * beta[t+1])).sum(axis=1); beta[t] /= scale[t]
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
      - 任意: difficulty_tag ∈ {easy,med,hard}, kc_tags, engagement(0-1), est_sec(秒)
      - 任意: a,b,c,bloom（使わなくてもOK）
    """
    if items_csv and os.path.exists(items_csv):
        items = pd.read_csv(items_csv)
        items = items[items["skill_id"].isin(skills)].copy()
        # 欠損列の補完
        for col, default in [
            ("kc_tags",""), ("engagement",0.5), ("est_sec",60), ("difficulty_tag","med")
        ]:
            if col not in items.columns: items[col] = default
        return items[["item_id","skill_id","kc_tags","engagement","est_sec","difficulty_tag"] +
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
        take = take.assign(kc_tags="", engagement=0.5, est_sec=60, difficulty_tag="med")
        chunks.append(take)
    return pd.concat(chunks, ignore_index=True) if chunks else base

# ========= スコア（teach/motivate/hybrid） =========
def score_teach(pnext, n_obs, pT, pL_post, item_row, goal_skills:set, goal_tags:set,
                sweet=0.65):
    sid = int(item_row["skill_id"])
    tags = set(parse_tags(item_row.get("kc_tags","")))
    coverage = 0.0
    if sid in goal_skills: coverage += 1.0
    if goal_tags: coverage += 0.5 * sum(1 for t in tags if t in goal_tags)
    deficit = (1.0 - pL_post)
    learn_gain = pT * (1.0 - pL_post)
    sweet_term = -abs(pnext - sweet)
    explore = 0.10/np.sqrt(n_obs+1.0)
    return (1.2*coverage) + (0.8*deficit) + (0.6*learn_gain) + (0.3*sweet_term) + explore

def score_motivate(pnext, n_obs, pT, pL_post, item_row, sweet=0.72):
    engagement = float(item_row.get("engagement", 0.5))
    est_sec    = float(item_row.get("est_sec", 60.0))
    sweet_term = -abs(pnext - sweet)
    time_term  = - (est_sec/120.0)
    explore    = 0.20/np.sqrt(n_obs+1.0)
    return (1.0*sweet_term) + (0.6*engagement) + (0.2*explore) + (0.2*pT) + (0.2*(-time_term))

# ========= 難易度タグによる per-item 補正 =========
def parse_tag_deltas(tag_deltas_str: str):
    # 例: "+0.12,0,-0.12" → {"easy":0.12,"med":0.0,"hard":-0.12}
    vals = [float(x) for x in tag_deltas_str.split(",")]
    if len(vals) != 3: raise ValueError("--tag_deltas は 'e, m, h' の3要素で指定してください")
    return {"easy": vals[0], "med": vals[1], "hard": vals[2]}

def p_item_from_tag(pnext: float, tag: str, deltas: Dict[str,float]):
    delta = deltas.get(str(tag).lower(), 0.0)
    return float(np.clip(pnext + delta, 1e-3, 0.999))

# ========= 候補ランキング =========
def rank_candidates(report_df: pd.DataFrame, items_df: pd.DataFrame, user_id: int,
                    mode="motivate", goal_skills: List[int]=None, goal_tags: List[str]=None,
                    sweet_motivate=0.72, sweet_teach=0.65, topn=12, alpha=0.5,
                    difficulty_mode="tag", tag_deltas=None) -> pd.DataFrame:
    """
    difficulty_mode: "none" | "tag"
      - none: per-item補正なし（p_for_llm = pnext）
      - tag : difficulty_tag を使って p_for_llm を補正（easy/med/hard）
    """
    rep = report_df[report_df["user_id"]==user_id].set_index("skill_id")
    gskills = set(int(x) for x in (goal_skills or []))
    gtags   = set(t for t in (goal_tags or []) if t)
    tag_deltas = tag_deltas or {"easy":0.12,"med":0.0,"hard":-0.12}

    rows = []
    for _, it in items_df.iterrows():
        sid = int(it["skill_id"])
        if sid not in rep.index: continue
        r = rep.loc[sid]
        pnext = float(r["P(next correct)"])
        n_obs = float(r["n_obs"]); pT = float(r["p_T"]); pL = float(r["P(L)_posterior"])

        # モード別スコア
        if mode == "teach":
            score = score_teach(pnext, n_obs, pT, pL, it, gskills, gtags, sweet=sweet_teach)
        elif mode == "motivate":
            score = score_motivate(pnext, n_obs, pT, pL, it, sweet=sweet_motivate)
        else:
            s_teach = score_teach(pnext, n_obs, pT, pL, it, gskills, gtags, sweet=sweet_teach)
            s_moti  = score_motivate(pnext, n_obs, pT, pL, it, sweet=sweet_motivate)
            score   = alpha * s_teach + (1.0 - alpha) * s_moti

        # per-item の expected を決める
        if difficulty_mode == "tag":
            p_for_llm = p_item_from_tag(pnext, it.get("difficulty_tag","med"), tag_deltas)
        else:
            p_for_llm = pnext

        rows.append({
            "item_id": str(it["item_id"]),
            "skill_id": sid,
            "pnext": pnext,            # skill基準
            "p_for_llm": p_for_llm,    # アイテム基準（LLMに渡す期待正答率）
            "score": score
        })
    cand = pd.DataFrame(rows)
    if cand.empty: return cand
    return cand.sort_values("score", ascending=False).head(topn)

# ========= 配分保証 & 多様化 =========
def _select_unique_top(df: pd.DataFrame, k: int, seen_skills: set):
    """ skill重複をできるだけ避けて上位kを選ぶ（足りなければ重複も許容） """
    out = []
    # 1周目：未出スキルを優先
    for _, row in df.iterrows():
        if len(out) >= k: break
        sid = int(row["skill_id"])
        if sid in seen_skills: continue
        out.append(row)
        seen_skills.add(sid)
    # 2周目：まだ足りなければ重複も許容
    if len(out) < k:
        for _, row in df.iterrows():
            if len(out) >= k: break
            key = (row["item_id"], int(row["skill_id"]))
            if any((r["item_id"], int(r["skill_id"])) == key for r in out): continue
            out.append(row)
    if not out:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(out)

def enforce_mix(ranked_df: pd.DataFrame, mix_str="3,1,1",
                review_thr=0.40, challenge_thr=0.85, center=0.70,
                prefer_unique_skills=True) -> pd.DataFrame:
    """ recommendations の配分（practice, review, challenge）を満たしつつ、多様性を考慮 """
    p_need, r_need, c_need = [int(x) for x in mix_str.split(",")]
    # しきい値は per-item の期待（p_for_llm）で判定
    rev  = ranked_df[ranked_df["p_for_llm"] < review_thr].copy()
    chal = ranked_df[ranked_df["p_for_llm"] >= challenge_thr].copy()
    prac = ranked_df[(ranked_df["p_for_llm"] >= review_thr) & (ranked_df["p_for_llm"] < challenge_thr)].copy()

    # practiceは中心帯に近い順、他はscore順
    prac = prac.assign(dist=(prac["p_for_llm"]-center).abs()).sort_values(["dist","score"], ascending=[True,False]).drop(columns="dist", errors="ignore")
    rev  = rev.sort_values("score", ascending=False)
    chal = chal.sort_values("score", ascending=False)

    out_frames = []
    seen = set() if prefer_unique_skills else set()
    # 可能な限りスキル重複を避けて選ぶ
    out_frames.append(_select_unique_top(prac, p_need, seen))
    out_frames.append(_select_unique_top(rev,  r_need, seen))
    out_frames.append(_select_unique_top(chal, c_need, seen))
    out = pd.concat(out_frames, ignore_index=True)

    # 足りない分を中心帯に近い順で充填（skill多様性を再度試みる）
    total_need = p_need + r_need + c_need
    if len(out) < total_need:
        remain = pd.concat([
            prac[~prac.index.isin(out.index)],
            rev[~rev.index.isin(out.index)],
            chal[~chal.index.isin(out.index)]
        ], ignore_index=True)
        if not remain.empty:
            remain = remain.assign(dist=(remain["p_for_llm"]-center).abs()).sort_values(["dist","score"], ascending=[True,False])
            out_extra = _select_unique_top(remain, total_need - len(out), seen)
            out = pd.concat([out, out_extra], ignore_index=True)

    # 念のため上位 total_need に切り詰め
    if len(out) > total_need:
        out = out.iloc[:total_need].copy()

    return out.reset_index(drop=True)

# ========= LLMメッセージ =========
def build_llm_messages(user_id: int, ranked_df: pd.DataFrame, mode="motivate",
                       goals=None, language="ja"):
    candidates = [
        {"item_id": str(r.item_id), "skill_id": int(r.skill_id),
         "expected_p_correct": round(float(r.p_for_llm), 2)}
        for r in ranked_df.itertuples(index=False)
    ]
    goal_hint = goals or {}
    system = (
        "あなたは教育向けの推薦アシスタントです。指定された候補問題リストからのみ選び、"
        "出力は必ず指定のJSONスキーマに厳密に従って日本語で返してください。候補外のIDは使用禁止です。"
        "助言は短く具体的に、数値は小数第2位まで。"
    )
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
        "goals": goal_hint,
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
        p = float(r.p_for_llm)
        t = "practice"
        if p >= 0.85: t = "challenge"
        if p <= 0.50: t = "review"
        reason = f"P(next correct)={round(p,2)}で目標帯に近い。"
        if mode == "teach": reason += " 目標知識の定着に有効。"
        if mode == "motivate": reason += " 成功体験を得やすい難易度。"
        recs.append({
            "item_id": str(r.item_id),
            "skill_id": int(r.skill_id),
            "reason": reason,
            "expected_p_correct": round(p,2),
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
    # 入力
    ap.add_argument("--csv", default="one_200_50_500_50000.csv")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--user_id", type=int, required=True)
    ap.add_argument("--skill_ids", nargs="*")
    ap.add_argument("--auto_pick_skills", type=int, default=3)
    ap.add_argument("--items_csv", default=None,
                    help="候補アイテムCSV: item_id, skill_id[, difficulty_tag, kc_tags, engagement, est_sec, ...]")
    ap.add_argument("--per_skill_items", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)

    # 目的／スコア
    ap.add_argument("--mode", choices=["teach","motivate","hybrid"], default="motivate")
    ap.add_argument("--goal_skills", nargs="*", help="教員が到達させたいskill_id（空白/カンマ混在OK）")
    ap.add_argument("--goal_tags", nargs="*", help="教員が到達させたいkcタグ（空白/カンマ混在OK）")
    ap.add_argument("--alpha", type=float, default=0.5, help="hybrid時のteach重み (0~1)")
    ap.add_argument("--sweet_motivate", type=float, default=0.72)
    ap.add_argument("--sweet_teach", type=float, default=0.65)

    # 難易度補正
    ap.add_argument("--difficulty_mode", choices=["none","tag"], default="tag",
                    help="per-item 期待正答率の補正方法（none:補正なし / tag: difficulty_tag を用いる）")
    ap.add_argument("--tag_deltas", default="+0.12,0,-0.12",
                    help="difficulty_tag が easy,med,hard のときの補正値（例: +0.12,0,-0.12）")

    # 配分・多様性
    ap.add_argument("--mix", default="3,1,1",
                    help="出力配分 practice,review,challenge（合計=5が目安）")
    ap.add_argument("--review_thr", type=float, default=0.40)
    ap.add_argument("--challenge_thr", type=float, default=0.85)
    ap.add_argument("--center", type=float, default=0.70,
                    help="practiceの中心帯（0.6〜0.8目安）")
    ap.add_argument("--prefer_unique_skills", action="store_true",
                    help="可能な範囲で skill_id の重複を避ける")

    # 出力
    ap.add_argument("--topn", type=int, default=12,
                    help="enforce_mix 前のランキング上位を何件まで候補に渡すか（配分後に最終的に5件程度）")
    ap.add_argument("--print_report", action="store_true")
    ap.add_argument("--export_llm_payload", default=None)
    ap.add_argument("--export_ranked", default=None, help="配分適用後の候補を保存するパス")
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

    # 7) ランキング（per-item補正のための deltas 準備）
    deltas = parse_tag_deltas(args.tag_deltas)
    ranked0 = rank_candidates(
        report_df, items_df, args.user_id, mode=args.mode,
        goal_skills=goal_skills, goal_tags=goal_tags,
        sweet_motivate=args.sweet_motivate, sweet_teach=args.sweet_teach,
        topn=args.topn, alpha=args.alpha,
        difficulty_mode=args.difficulty_mode, tag_deltas=deltas
    )
    if ranked0.empty:
        raise SystemExit("ランキング候補が空です。データや条件を確認してください。")

    # 8) 配分保証 & 多様化
    ranked = enforce_mix(
        ranked_df=ranked0,
        mix_str=args.mix,
        review_thr=args.review_thr,
        challenge_thr=args.challenge_thr,
        center=args.center,
        prefer_unique_skills=args.prefer_unique_skills
    )

    print(f"# Ranked Candidates after enforce_mix (mode={args.mode}, mix={args.mix})")
    print(ranked.to_string(index=False) if not ranked.empty else "(empty)")
    print()

    # 保存（任意）
    if args.export_ranked:
        ranked.to_json(args.export_ranked, orient="records", force_ascii=False, indent=2)
        print(f"[Saved] ranked candidates -> {args.export_ranked}")

    # 9) LLMメッセージ
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

    # 10) モック応答（任意）
    if args.mock_llm:
        mock_json = build_mock_llm_output(args.user_id, ranked, mode=args.mode)
        print("\n--- Mock LLM Output JSON ---")
        print(mock_json)

if __name__ == "__main__":
    main()
