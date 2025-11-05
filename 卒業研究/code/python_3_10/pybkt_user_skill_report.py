# pybkt_user_skill_report.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, pandas as pd, numpy as np
from pyBKT.models import Model

def extract_skill_params(m):
    """
    pyBKT 1.4.1 を想定。学習済みパラメータを
    skill_name ごとの DataFrame(p_T,p_S,p_G,p_L0)で返す。
    """
    P = getattr(m, "params", None)
    if isinstance(P, pd.DataFrame):
        df = P.rename(columns={
            "learns":"p_T","slips":"p_S","guesses":"p_G","prior":"p_L0",
            "learn":"p_T","slip":"p_S","guess":"p_G","L0":"p_L0"
        })
        if "p_T" not in df.columns:
            raise RuntimeError("pyBKT params 形式が想定外です。print(model.params)で列名を確認してください。")
        df.index.name = "skill_name"
        return df.reset_index()
    elif isinstance(P, dict):
        # もし dict なら {param_name: {skill: value}} 形式を想定
        learns  = P.get("learns")  or P.get("learn")  or {}
        slips   = P.get("slips")   or P.get("slip")   or {}
        guesses = P.get("guesses") or P.get("guess")  or {}
        priors  = P.get("prior")   or P.get("L0")     or {}
        skills = sorted(set(learns)|set(slips)|set(guesses)|set(priors))
        rows=[]
        for s in skills:
            rows.append(dict(skill_name=str(s),
                             p_T=float(learns.get(s,np.nan)),
                             p_S=float(slips.get(s,np.nan)),
                             p_G=float(guesses.get(s,np.nan)),
                             p_L0=float(priors.get(s,np.nan))))
        return pd.DataFrame(rows)
    else:
        raise RuntimeError("model.params が見つかりませんでした。pyBKT 1.4.1 を想定しています。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--user", required=True, help="例: 68 または u68（入力CSVに合わせる）")
    ap.add_argument("--skill", help="特定スキルだけ表示したい場合（任意）")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--order-col", default="order_id")
    ap.add_argument("--user-col",  default="user_id")
    ap.add_argument("--skill-col", default="skill_name")
    ap.add_argument("--correct-col", default="correct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-fits", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # 列名マッピング
    cols = dict(order_id=args.order_col, user_id=args.user_col,
                skill_name=args.skill_col, correct=args.correct_col)

    # order_id が整数でなければ自動で連番を作る（user内）
    import pandas as pd
    if not pd.api.types.is_integer_dtype(df[cols["order_id"]]):
        # 文字列日付ならdatetimeに
        if df[cols["order_id"]].dtype == "object":
            df[cols["order_id"]] = pd.to_datetime(df[cols["order_id"]], errors="coerce")
        df = df.sort_values([cols["user_id"], cols["order_id"]])
        new_col = "__order_id_int__"
        df[new_col] = df.groupby(cols["user_id"]).cumcount()
        cols["order_id"] = new_col  # 以降はこれを使う

    # pyBKT fit
    m = Model(seed=args.seed, num_fits=args.num_fits).fit(data=df, defaults=cols)

    # 予測（state_predictions / correct_predictions）
    pred = m.predict(data=df, defaults=cols).sort_values([cols["user_id"], cols["skill_name"], cols["order_id"]])

    # 学習済みパラメータ（スキル別 T,S,G,L0）
    par = extract_skill_params(m)
    par[cols["skill_name"]] = par["skill_name"].astype(str)

    # ユーザー抽出（必要ならスキルも）
    sub = pred[pred[cols["user_id"]].astype(str) == str(args.user)]
    if args.skill:
        sub = sub[sub[cols["skill_name"]].astype(str) == str(args.skill)]
    if sub.empty:
        raise SystemExit("該当ユーザー（/スキル）のデータがありません。")

    # 各 user×skill の最終試行だけ取り出す
    last = sub.groupby([cols["user_id"], cols["skill_name"]], as_index=False).tail(1)

    # スキル別パラメータを付与
    last = last.merge(par, how="left", left_on=cols["skill_name"], right_on=cols["skill_name"])

    # 列を計算
    last = last.rename(columns={
        "state_predictions":"P(L)_posterior",
        "correct_predictions":"_tmp_correct"
    })
    last["P(L)_after_learn"] = last["P(L)_posterior"] + (1 - last["P(L)_posterior"]) * last["p_T"]
    last["P(next correct)"]  = (1 - last["p_S"]) * last["P(L)_after_learn"] + last["p_G"] * (1 - last["P(L)_after_learn"])
    last["n_obs"] = pred.groupby([cols["user_id"], cols["skill_name"]]).size().reindex(last.set_index([cols["user_id"], cols["skill_name"]]).index).to_numpy()


    out = last[[cols["user_col"], cols["skill_col"], "n_obs",
                "P(L)_posterior", "P(L)_after_learn", "P(next correct)",
                "p_L0","p_T","p_S","p_G"]].copy()

    # 表示整形
    for c in ["P(L)_posterior","P(L)_after_learn","P(next correct)","p_L0","p_T","p_S","p_G"]:
        out[c] = out[c].astype(float)
    out = out.sort_values("P(L)_posterior", ascending=False).head(args.top)

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"saved -> {args.out}")
    else:
        # 画面表示
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

if __name__ == "__main__":
    main()
