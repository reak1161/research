#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BKT×IRT: 複数スキル向け 推薦(geom, lam=0.05, target_band=auto)
- ダウンロード不要: 画面にTop-10とΔ統計を出力のみ（ファイル保存なし）
- あなたの bkt_mw.py を使用（同じディレクトリ or パスを通す）
- まずは高速モード（上位Nスキルに絞り、EM反復少なめ）で試し、
  問題なければ FULL モード（全スキル＆反復増やす）に切替えてください。

使い方（例）:
  python run_recs_geom.py --csv ../csv/200_50_500_100000.csv
オプション:
  --fast true/false         : 高速モード（既定: true）
  --n_keep 15               : 高速モードで残すスキル数
  --max_iter 7              : EM反復（高速）
  --full_max_iter 20        : EM反復（フル）
  --user -1                 : ユーザーID（-1なら最頻出ユーザーを自動選択）
  --topk 10                 : 表示するTop-K
  --k_cand 200              : recommend_topKのK（候補の上限）
  --max_candidates 1000     : 候補母集団の上限
"""

import argparse
import importlib.util
import os
import sys
import json
import numpy as np
import pandas as pd

# --------- 引数 ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="学習記録CSVのパス（user_id,item_id,skill,result,date）")
    p.add_argument("--fast", type=str, default="true", help="高速モード: true/false")
    p.add_argument("--n_keep", type=int, default=15, help="高速モードで残すスキル数")
    p.add_argument("--max_iter", type=int, default=7, help="高速モードのEM反復回数")
    p.add_argument("--full_max_iter", type=int, default=20, help="フルモードのEM反復回数")
    p.add_argument("--user", type=int, default=-1, help="対象ユーザーID（-1なら自動選択）")
    p.add_argument("--topk", type=int, default=10, help="表示するTop-K件数")
    p.add_argument("--k_cand", type=int, default=200, help="recommend_topKのK")
    p.add_argument("--max_candidates", type=int, default=1000, help="候補母集団の上限")
    return p.parse_args()

# --------- 表示ユーティリティ ----------
def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_df(df, max_rows=20, cols=None):
    if cols is not None:
        df = df.loc[:, cols]
    with pd.option_context("display.max_rows", max_rows,
                           "display.max_columns", None,
                           "display.width", 120):
        print(df)

def qsafe(s, q):
    try:
        return float(s.quantile(q))
    except Exception:
        return float("nan")

# --------- メイン ----------
def main():
    args = parse_args()
    FAST = str(args.fast).lower() in ("1","true","t","yes","y")

    # bkt_mw.py の動的インポート（このスクリプトと同ディレクトリ想定。パス調整は適宜）
    here = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(here, "bkt_mw.py"),
        os.path.join(os.getcwd(), "bkt_mw.py"),
        "/mnt/data/bkt_mw.py",
    ]
    bkt_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            bkt_path = p
            break
    if bkt_path is None:
        print("ERROR: bkt_mw.py が見つかりません。スクリプトのある場所に置くか、パスを調整してください。")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("bkt_mw", bkt_path)
    bkt_mw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bkt_mw)

    # 1) データ読込
    print_section("1) Load data")
    df_all = bkt_mw.load_data(args.csv)
    print(f"Rows: {len(df_all):,}, Users: {df_all['user_id'].nunique()}, Items: {df_all['item_id'].nunique()}")

    # 2) explode（1/m重み。重みを変えたければここで再計算して差し替え）
    print_section("2) Prepare exploded (1/m weights)")
    df_ex_full = bkt_mw.prepare_exploded(df_all)
    print(f"Exploded rows: {len(df_ex_full):,}, Skills: {df_ex_full['skill_id'].nunique()}")

    # 3) モード別：高速 or フル
    if FAST:
        print_section("3) FAST MODE: 上位スキルに絞って学習")
        # 効率サンプル数（重み合計）が多い上位Nスキルを残す
        eff = df_ex_full.groupby("skill_id")["weight"].sum().sort_values(ascending=False)
        keep_skills = set(eff.head(args.n_keep).index.astype(int).tolist())

        # アイテム側：そのアイテムのスキル集合が keep_skills ⊆ のものに絞る
        def item_skills_in_set(df, allowed):
            skills = df["skill"].astype(str)
            def ok(s):
                return all(int(x) in allowed for x in s.split(","))
            return skills.apply(ok)

        df_all_filt = df_all[item_skills_in_set(df_all, keep_skills)].copy()
        df_ex = df_ex_full[df_ex_full["skill_id"].isin(keep_skills)].copy()

        print(f"Kept skills (count): {len(keep_skills)} -> {sorted(list(keep_skills))}")
        print(f"Filtered items: {df_all_filt['item_id'].nunique()}  (rows: {len(df_all_filt):,})")

        max_iter = args.max_iter
    else:
        print_section("3) FULL MODE: 全スキルで学習")
        df_all_filt = df_all
        df_ex = df_ex_full
        max_iter = args.full_max_iter

    # 4) 重み付きEM + MAP推定（bkt_mwに依存）
    print_section(f"4) Fit per-skill (max_iter={max_iter})")
    params_by_skill = bkt_mw.fit_per_skill_weighted(
        df_exploded=df_ex,
        max_iter=max_iter,
        min_eff_samples=50 if FAST else 200
    )
    print(f"Trained skills: {len(params_by_skill)}")

    # 5) 対象ユーザーの決定
    if args.user >= 0:
        sample_user = int(args.user)
    else:
        sample_user = int(df_ex["user_id"].value_counts().idxmax())
    print(f"Target user: {sample_user}")

    # 6) 推薦（指定：geom, target_band=auto, lam=0.05）
    print_section("5) Recommend (combine_mode='geom', target_band='auto', lam=0.05)")
    raw = bkt_mw.recommend_topK(
        user_id=sample_user,
        K=args.k_cand,
        df_all=df_all_filt,
        df_exploded=df_ex,
        params_by_skill=params_by_skill,
        combine_mode="geom",
        alpha=0.5,             # geom では未使用
        target_band="auto",
        lam=0.05,
        max_candidates=args.max_candidates
    )

    # 7) Top-K の表示
    cols = ["item_id", "p_correct", "delta", "score", "skills"]
    top = raw.head(args.topk).copy()
    print_section(f"6) Top-{args.topk} recommendations (geom, lam=0.05, target_band=auto)")
    print_df(top, cols=cols)

    # 8) Δ統計の表示
    print_section("7) Δ (expected mastery gain) statistics")
    if "delta" not in raw.columns or raw["delta"].empty:
        print("No candidates were scored (delta not available).")
    else:
        ds = raw["delta"].astype(float)
        stats = {
            "count": int(ds.shape[0]),
            "mean": float(ds.mean()) if ds.shape[0] else float("nan"),
            "std": float(ds.std(ddof=1)) if ds.shape[0] > 1 else float("nan"),
            "min": float(ds.min()) if ds.shape[0] else float("nan"),
            "q10": qsafe(ds, 0.10),
            "q25": qsafe(ds, 0.25),
            "median": float(ds.median()) if ds.shape[0] else float("nan"),
            "q75": qsafe(ds, 0.75),
            "q90": qsafe(ds, 0.90),
            "max": float(ds.max()) if ds.shape[0] else float("nan"),
        }
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    # 9) 参考：p_correct の基本統計
    if "p_correct" in raw.columns and not raw["p_correct"].empty:
        print_section("8) p_correct statistics (参考)")
        pc = raw["p_correct"].astype(float)
        ref = {
            "mean": float(pc.mean()),
            "median": float(pc.median()),
            "min": float(pc.min()),
            "max": float(pc.max()),
        }
        print(json.dumps(ref, ensure_ascii=False, indent=2))

    print_section("DONE")

if __name__ == "__main__":
    main()
