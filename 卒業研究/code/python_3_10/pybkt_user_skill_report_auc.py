# aucによる忘却オンオフの評価

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyBKT で学習→予測し、ユーザー×スキルの要約を出すユーティリティ。
- 入力: 縦持ちCSV（デフォルト: order_id,user_id,skill_name,correct）
- 列名が違う場合は --order-col/--user-col/--skill-col/--correct-col でマッピング
- order_id が整数でない（例: date 文字列）ときは、ユーザー内の連番を自動生成
- 出力列: user_id, skill_name, n_obs, P(L)_posterior, P(L)_after_learn, P(next correct), p_L0, p_T, p_S, p_G

想定 pyBKT: 1.4.1

追加機能:
- `--compare-forgets` で **忘却なし vs 忘却あり** の AUC 比較（全体/スキル別/ユーザー別/ユーザー×スキル別）
- `--auc-by` で集計粒度、`--eval-out` でCSV保存
"""
import argparse
import sys
import os
import pandas as pd
import numpy as np
from pyBKT.models import Model


def extract_skill_params(model: Model) -> pd.DataFrame:
    """pyBKT 学習済みパラメータを skill_name ごとに取り出す（多バージョン対応）。
    戻り: columns=[skill_name, p_T, p_S, p_G, p_L0]
    受け取り型: DataFrame / dict / list-array のいずれでも可。
    """
    P = getattr(model, "params", None)
    if P is None:
        raise RuntimeError("model.params が見つかりません。pyBKTのバージョンを確認してください。")
    # 一部バージョンはメソッドなので呼び出す
    if callable(P):
        P = P()

    # skills 候補（位置対応のときに使う）
    skill_names = None
    for key in ("skills", "unique_skills", "KC_list", "kcs"):
        v = getattr(model, key, None)
        if v is not None:
            try:
                skill_names = [str(x) for x in list(v)]
            except Exception:
                pass
            break

    # DataFrame 形式
    if isinstance(P, pd.DataFrame):
        df = P.copy()
        rename_map = {
            "learns": "p_T", "learn": "p_T",
            "slips": "p_S", "slip": "p_S",
            "guesses": "p_G", "guess": "p_G",
            "prior": "p_L0", "L0": "p_L0",
            "forgets": "p_F", "forget": "p_F",
        }
        df = df.rename(columns=rename_map)

        # --- Index の正規化（MultiIndex 含む）---
        if isinstance(df.index, pd.MultiIndex):
            level_names = list(df.index.names)
            pick_name = None
            for cand in ("skill", "skill_name", "KC", "kc", "kcid", "KC_id", "KCs", "skills"):
                if cand in level_names:
                    pick_name = cand
                    break
            if pick_name is not None:
                names = df.index.get_level_values(pick_name).astype(str)
            else:
                names = df.index.get_level_values(0).astype(str)
            df = df.reset_index()
            # もう一方のレベル名（パラメータ名）を推定
            other_levels = [n for n in level_names if n != pick_name]
            param_level = other_levels[0] if other_levels else None
            if param_level and param_level in df.columns:
                df = df.rename(columns={pick_name: "skill_name", param_level: "param"})
            else:
                df = df.rename(columns={pick_name: "skill_name", df.columns[1]: "param"})
        else:
            # 単一Indexの場合
            if df.index.name and df.index.name not in df.columns:
                df = df.reset_index().rename(columns={df.index.name: "skill_name"})
            if "skill_name" not in df.columns:
                if skill_names is not None and len(skill_names) == len(df):
                    df.insert(0, "skill_name", skill_names)
                else:
                    df = df.reset_index()
                    df = df.rename(columns={df.columns[0]: "skill_name"})

        # --- ワイド/ロング判定＆ピボット ---
        keep = ["skill_name", "p_T", "p_S", "p_G", "p_L0"]
        if not any(k in df.columns for k in ["p_T","p_S","p_G","p_L0"]):
            # ロング形式を想定（param, value系）
            # パラメータ名の列候補
            cand_param = None
            for c in df.columns:
                if str(c).lower() in ("param","parameter","name","variable","par"):
                    cand_param = c; break
            # 値の列候補
            cand_value = None
            # 数値カラムから候補を選ぶ
            for c in df.columns:
                if c in ("skill_name", cand_param):
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    cand_value = c; break
            if cand_param is not None and cand_value is not None:
                # パラメータ名を正規化
                df[cand_param] = df[cand_param].astype(str).str.lower()
                piv = df.pivot_table(index="skill_name", columns=cand_param, values=cand_value, aggfunc="first").reset_index()
                piv.columns = [str(x) for x in piv.columns]
                # 標準名に合わせる
                rename_map2 = {
                    "learns":"p_T", "learn":"p_T", "t":"p_T",
                    "slips":"p_S", "slip":"p_S", "s":"p_S",
                    "guesses":"p_G", "guess":"p_G", "g":"p_G",
                    "prior":"p_L0", "l0":"p_L0", "prior_l0":"p_L0"
                }
                piv = piv.rename(columns=rename_map2)
                for k in keep:
                    if k not in piv.columns:
                        piv[k] = np.nan
                piv["skill_name"] = piv["skill_name"].astype(str)
                df = piv[keep]
            else:
                # どうしても解釈できない場合は空の形にして返す
                out = pd.DataFrame({k: [] for k in keep})
                return out
        else:
            # 既にワイド形式
            for k in keep:
                if k not in df.columns:
                    df[k] = np.nan
            if "skill_name" not in df.columns:
                if skill_names is not None and len(skill_names) == len(df):
                    df.insert(0, "skill_name", skill_names)
                else:
                    df.insert(0, "skill_name", df.index.astype(str))
            df["skill_name"] = df["skill_name"].astype(str)
            df = df[keep]

        # 同一 skill が複数行ある場合は平均で集約
        df = df.groupby("skill_name", as_index=False).agg({"p_T":"mean","p_S":"mean","p_G":"mean","p_L0":"mean"})
        return df

    # dict 形式
    if isinstance(P, dict):
        # ネスト整形
        src = P
        for key in ("by_skill", "skills", "kcs"):
            if key in src and isinstance(src[key], dict):
                src = src[key]
                break
        # パラメータ候補を取り出す（配列 or dict）
        def to_series(val):
            if isinstance(val, dict):
                s = pd.Series(val)
                s.index = s.index.astype(str)
                return s
            arr = list(val) if hasattr(val, "__len__") else []
            if skill_names is None:
                raise RuntimeError("params が配列形式ですが skill_names を特定できません。")
            if len(arr) != len(skill_names):
                raise RuntimeError("params 配列長と skills 長が一致しません。")
            return pd.Series(arr, index=[str(x) for x in skill_names])
        # 候補キー
        kT = next((k for k in ("learns","learn","T") if k in src), None)
        kS = next((k for k in ("slips","slip","S") if k in src), None)
        kG = next((k for k in ("guesses","guess","G") if k in src), None)
        kL = next((k for k in ("prior","L0","prior_L0") if k in src), None)
        d = {}
        if kT: d["p_T"] = to_series(src[kT])
        if kS: d["p_S"] = to_series(src[kS])
        if kG: d["p_G"] = to_series(src[kG])
        if kL: d["p_L0"] = to_series(src[kL])
        df = pd.DataFrame(d)
        df.index.name = "skill_name"
        df = df.reset_index()
        for k in ["p_T","p_S","p_G","p_L0"]:
            if k not in df.columns:
                df[k] = np.nan
        return df[["skill_name","p_T","p_S","p_G","p_L0"]]

    # list/tuple/配列 → 位置対応
    if hasattr(P, "__len__"):
        if skill_names is None:
            raise RuntimeError("params が配列形式ですが skill_names を特定できません。")
        # 代表カラム名が判別できないため NaN で返す（後段の計算はフォールバックへ）
        return pd.DataFrame({
            "skill_name": [str(x) for x in skill_names],
            "p_T": np.nan, "p_S": np.nan, "p_G": np.nan, "p_L0": np.nan
        })

    raise RuntimeError(f"未対応の params 型: {type(P)}")


def _rankdata_tieavg(a: np.ndarray) -> np.ndarray:
    """平均順位のrankをnumpyのみで（tiesは平均）。1始まり。
    """
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(a, dtype=float)
    n = len(a)
    i = 0
    while i < n:
        j = i
        ai = a[order[i]]
        while j + 1 < n and a[order[j+1]] == ai:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based ranks
        ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks


def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_tieavg(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _ensure_score_series(pred: pd.DataFrame, cols: dict, par: pd.DataFrame) -> pd.Series:
    if "_pred_correct_now" in pred.columns:
        return pred["_pred_correct_now"].astype(float)
    # fallback: compute from p_S, p_G and state_predictions
    par_map = par.set_index("skill_name")[['p_S','p_G']]
    sub = pred[[cols["skill_name"], "state_predictions"]].copy()
    sub = sub.join(par_map, on=cols["skill_name"])  # align by skill
    score = (1.0 - sub["p_S"].astype(float)) * sub["state_predictions"].astype(float) \
            + sub["p_G"].astype(float) * (1.0 - sub["state_predictions"].astype(float))
    return score


def _evaluate_auc(pred: pd.DataFrame, cols: dict, par: pd.DataFrame, by: str) -> pd.DataFrame:
    score = _ensure_score_series(pred, cols, par)
    y = pred[cols["correct"]].astype(int)
    if by == "overall":
        auc = _roc_auc_safe(y.to_numpy(), score.to_numpy())
        pos_rate = float(y.mean()) if len(y) else float("nan")
        return pd.DataFrame([{ "scope":"overall", "n": len(y), "pos_rate": pos_rate, "auc": auc }])
    if by == "skill":
        rows = []
        for sk, g in pred.groupby(cols["skill_name"]):
            auc = _roc_auc_safe(g[cols["correct"]].to_numpy(), score.loc[g.index].to_numpy())
            rows.append({"skill": str(sk), "n": int(g.shape[0]), "pos_rate": float(g[cols["correct"]].mean()), "auc": auc})
        return pd.DataFrame(rows)
    if by == "user":
        rows = []
        for u, g in pred.groupby(cols["user_id"]):
            auc = _roc_auc_safe(g[cols["correct"]].to_numpy(), score.loc[g.index].to_numpy())
            rows.append({"user": str(u), "n": int(g.shape[0]), "pos_rate": float(g[cols["correct"]].mean()), "auc": auc})
        return pd.DataFrame(rows)
    if by == "user-skill":
        rows = []
        for (u, sk), g in pred.groupby([cols["user_id"], cols["skill_name"]]):
            auc = _roc_auc_safe(g[cols["correct"]].to_numpy(), score.loc[g.index].to_numpy())
            rows.append({"user": str(u), "skill": str(sk), "n": int(g.shape[0]), "pos_rate": float(g[cols["correct"]].mean()), "auc": auc})
        return pd.DataFrame(rows)
    raise ValueError("unknown by: " + str(by))


def _fit_predict(df: pd.DataFrame, cols: dict, seed: int, num_fits: int, forgets: bool):
    model = Model(seed=seed, num_fits=num_fits)
    # 一部バージョンは forgets 未対応なので try で包む
    try:
        model.fit(data=df, defaults=cols, forgets=forgets)
    except TypeError:
        if forgets:
            print("[warn] この pyBKT 版は forgets 未対応のため、忘却なしで学習します。", file=sys.stderr)
        model.fit(data=df, defaults=cols)
    try:
        pred = model.predict(data=df, defaults=cols)
    except TypeError:
        pred = model.predict(data=df)
    if "correct_predictions" in pred.columns:
        pred = pred.rename(columns={"correct_predictions":"_pred_correct_now"})
    param = extract_skill_params(model)
    if not param.empty:
        param = param.groupby("skill_name", as_index=False).agg({"p_T":"mean","p_S":"mean","p_G":"mean","p_L0":"mean"})
    return pred, param


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--user", required=False, help="例: 68 または u68（入力CSVに合わせる）。※AUC比較(--compare-forgets)では不要")
    ap.add_argument("--skill", help="特定スキルだけ表示（任意）")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--order-col", default="order_id")
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--skill-col", default="skill_name")
    ap.add_argument("--correct-col", default="correct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-fits", type=int, default=1)
    ap.add_argument("--out", default="", help="CSVで保存するパス（未指定なら標準出力）")
    # === Plot options ===
    ap.add_argument("--plot-out", default="", help="時系列プロットPNGの保存先。指定時に描画を実行")
    ap.add_argument("--skills", nargs='*', help="プロット対象スキル（空なら --skill / ユーザー出現上位）")
    ap.add_argument("--max-plots", type=int, default=6, help="自動選択時の最大スキル数")
    # === AUC compare options ===
    ap.add_argument("--compare-forgets", action="store_true", help="忘却なし vs 忘却ありでAUC比較を実行")
    ap.add_argument("--auc-by", choices=["overall","skill","user","user-skill"], default="overall")
    ap.add_argument("--eval-out", default="", help="AUC比較のCSV保存先（省略時は標準出力）")
    args = ap.parse_args()

    # --user は通常モードでのみ必須
    if not args.compare_forgets and not args.user:
        print("error: --user は通常モードで必須です (--compare-forgets のときは不要)", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv)

    # ===== AUC 比較モード（早期リターン） =====
    if args.compare_forgets:
        cols = dict(order_id=args.order_col, user_id=args.user_col, skill_name=args.skill_col, correct=args.correct_col)
        # order 整形（予測に必要）
        if cols["order_id"] not in df.columns:
            print(f"[info] 指定した order-col '{cols['order_id']}' が見つかりません。列: {df.columns.tolist()}", file=sys.stderr)
            sys.exit(1)
        if not pd.api.types.is_integer_dtype(df[cols["order_id"]]):
            if df[cols["order_id"]].dtype == object:
                df[cols["order_id"]] = pd.to_datetime(df[cols["order_id"]], errors="coerce")
            df = df.sort_values([cols["user_id"], cols["order_id"]])
            new_col = "__order_id_int__"
            df[new_col] = df.groupby(cols["user_id"]).cumcount()
            cols["order_id"] = new_col
            print("[info] 非整数の order_id を検出 → ユーザー内連番に置換しました。", file=sys.stderr)
        df[cols["skill_name"]] = df[cols["skill_name"]].astype(str)

        pred0, par0 = _fit_predict(df, cols, seed=args.seed, num_fits=args.num_fits, forgets=False)
        pred1, par1 = _fit_predict(df, cols, seed=args.seed, num_fits=args.num_fits, forgets=True)

        t0 = _evaluate_auc(pred0, cols, par0 if not par0.empty else par1, by=args.auc_by)
        t0.insert(0, "model", "no_forget")
        t1 = _evaluate_auc(pred1, cols, par1 if not par1.empty else par0, by=args.auc_by)
        t1.insert(0, "model", "forget")

        # マージして差分
        key_cols = [c for c in ["scope","skill","user"] if c in t0.columns or c in t1.columns]
        m = pd.merge(t0, t1, on=key_cols, how="outer", suffixes=("_noF","_F"))
        # 整形
        if "n_noF" in m.columns and "n_F" in m.columns:
            m["n"] = m[["n_noF","n_F"]].max(axis=1)
        if "pos_rate_noF" in m.columns and "pos_rate_F" in m.columns:
            m["pos_rate"] = m[["pos_rate_noF","pos_rate_F"]].mean(axis=1)
        if "auc_noF" in m.columns and "auc_F" in m.columns:
            m["delta_auc"] = m["auc_F"] - m["auc_noF"]
        # 並び替え
        sort_keys = [c for c in ["scope","user","skill"] if c in m.columns] + (["delta_auc"] if "delta_auc" in m.columns else [])
        m = m.sort_values(sort_keys)

        # 出力
        if args.eval_out:
            m.to_csv(args.eval_out, index=False)
            print(f"saved -> {args.eval_out}")
        else:
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(m.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        return

    # ===== 通常モード（レポート/プロット） =====
    cols = dict(order_id=args.order_col, user_id=args.user_col,
                skill_name=args.skill_col, correct=args.correct_col)

    cols = dict(order_id=args.order_col, user_id=args.user_col,
                skill_name=args.skill_col, correct=args.correct_col)

    # order_id が整数でなければ、ユーザー内の連番を生成（date等も許容）
    if cols["order_id"] not in df.columns:
        print(f"[info] 指定した order-col '{cols['order_id']}' が見つかりません。列: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)
    if not pd.api.types.is_integer_dtype(df[cols["order_id"]]):
        # 文字列日付なら datetime にしてから並べ替え
        if df[cols["order_id"]].dtype == object:
            df[cols["order_id"]] = pd.to_datetime(df[cols["order_id"]], errors="coerce")
        df = df.sort_values([cols["user_id"], cols["order_id"]])
        new_col = "__order_id_int__"
        df[new_col] = df.groupby(cols["user_id"]).cumcount()
        cols["order_id"] = new_col
        print("[info] 非整数の order_id を検出 → ユーザー内連番に置換しました。", file=sys.stderr)

    # 型の揃え（skill は文字列に）
    df[cols["skill_name"]] = df[cols["skill_name"]].astype(str)

    # 学習
    model = Model(seed=args.seed, num_fits=args.num_fits)
    model.fit(data=df, defaults=cols)

    # 予測（state_predictions / correct_predictions など付与）
    try:
        pred = model.predict(data=df, defaults=cols)
    except TypeError:
        pred = model.predict(data=df)
    pred = pred.sort_values([cols["user_id"], cols["skill_name"], cols["order_id"]])
    pred[cols["skill_name"]] = pred[cols["skill_name"]].astype(str)
    # 後でフォールバックに使えるよう保持
    if "correct_predictions" in pred.columns:
        pred = pred.rename(columns={"correct_predictions": "_pred_correct_now"})
    pred[cols["skill_name"]] = pred[cols["skill_name"]].astype(str)

    # スキル別パラメータ（重複行があっても一意化）
    par = extract_skill_params(model)
    if not par.empty:
        par = par.groupby("skill_name", as_index=False).agg({"p_T":"mean","p_S":"mean","p_G":"mean","p_L0":"mean"})

    # 対象ユーザー（/スキル）抽出
    sub = pred[pred[cols["user_id"]].astype(str) == str(args.user)]
    if args.skill:
        sub = sub[sub[cols["skill_name"]].astype(str) == str(args.skill)]
    if sub.empty:
        print("該当ユーザー（/スキル）のデータがありません。", file=sys.stderr)
        sys.exit(1)

    # 各 user×skill の最終試行（order_id 最大の行を厳密に1件取得）
    grp_keys = [cols["user_id"], cols["skill_name"]]
    idx = sub.groupby(grp_keys)[cols["order_id"]].idxmax()
    last = sub.loc[idx].copy()

    # n_obs を付与（確実にキー結合）
    counts_df = pred.groupby(grp_keys).size().reset_index(name="n_obs")
    last = last.merge(counts_df, on=grp_keys, how="left")
    # cast to nullable integer for clean printing
    try:
        last["n_obs"] = last["n_obs"].astype("Int64")
    except Exception:
        pass

    # パラメータ結合（キー正規化: 文字列/数値どちらでも一致させる）
    par_key_str = par["skill_name"].astype(str).str.strip()
    last_key_str = last[cols["skill_name"]].astype(str).str.strip()
    par1 = par.copy(); par1["__key"] = par_key_str
    last1 = last.copy(); last1["__key"] = last_key_str
    merged = last1.merge(par1.drop(columns=["skill_name"]).rename(columns={"__key":"__key_par"}), left_on="__key", right_on="__key_par", how="left")

    # まだ揃わない場合は数値化キーで再試行
    if merged[["p_T","p_S","p_G","p_L0"]].isna().all().all():
        par2 = par.copy(); par2["__key_num"] = pd.to_numeric(par2["skill_name"], errors="coerce")
        last2 = last.copy(); last2["__key_num"] = pd.to_numeric(last2[cols["skill_name"]], errors="coerce")
        merged = last2.merge(par2.drop(columns=["skill_name"]).rename(columns={"__key_num":"__key_num_par"}), left_on="__key_num", right_on="__key_num_par", how="left")

    last = merged

    # 欠損列（古いpyBKTなど）に対応
    for k in ("p_T","p_S","p_G","p_L0"):
        if k not in last.columns:
            last[k] = np.nan

    # 列名と指標計算
    if "state_predictions" not in last.columns:
        raise RuntimeError("predict 結果に 'state_predictions' が見つかりません。pyBKTのバージョンを確認してください。")
    last = last.rename(columns={"state_predictions": "P(L)_posterior"})
    # 念のため user_id / skill を保持（文字列化は出力時に丸める）
    last[cols["user_id"]] = last[cols["user_id"]].astype(str)
    last[cols["skill_name"]] = last[cols["skill_name"]].astype(str)
    last["P(L)_posterior"] = last["P(L)_posterior"].astype(float)
    # 次時点の習得確率（学習遷移適用）
    if last["p_T"].notna().any():
        last["P(L)_after_learn"] = last["P(L)_posterior"] + (1.0 - last["P(L)_posterior"]) * last["p_T"].fillna(0.0)
    else:
        last["P(L)_after_learn"] = np.nan

    # 次時点の正答確率
    if last[["p_S","p_G"]].notna().any().all():
        last["P(next correct)"] = (1.0 - last["p_S"]) * last["P(L)_after_learn"] + last["p_G"] * (1.0 - last["P(L)_after_learn"])
    else:
        # フォールバック：直近の予測確率（時点tでの予測）を代用
        if "_pred_correct_now" in last.columns:
            last["P(next correct)"] = last["_pred_correct_now"].astype(float)
        else:
            last["P(next correct)"] = np.nan

    # 出力整形
    out_cols = [cols["user_id"], cols["skill_name"], "n_obs",
                "P(L)_posterior", "P(L)_after_learn", "P(next correct)",
                "p_L0", "p_T", "p_S", "p_G"]
    out = last[out_cols].copy()

    # 並び替え・上位抽出
    out = out.sort_values("P(L)_posterior", ascending=False).head(args.top)

    # 出力
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"saved -> {args.out}")
    else:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # === Plot timeseries (optional) ===
    if args.plot_out:
        # skill 選定
        if args.skills and len(args.skills) > 0:
            skills_to_plot = [str(s) for s in args.skills]
        elif args.skill:
            skills_to_plot = [str(args.skill)]
        else:
            # 該当ユーザーの出現上位から自動選択
            cnt_user = pred[pred[cols["user_id"]].astype(str) == str(args.user)]
            top = cnt_user.groupby(cols["skill_name"]).size().sort_values(ascending=False)
            skills_to_plot = [str(s) for s in top.index[:args.max_plots].tolist()]
        _plot_user_skills_timeseries(pred, par, args.user, skills_to_plot, cols, args.plot_out)
        print(f"plot saved -> {args.plot_out}")


def _plot_user_skills_timeseries(pred: pd.DataFrame, par: pd.DataFrame, user, skills, cols, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib を読み込めませんでした: {e}", file=sys.stderr)
        return
    user_str = str(user)
    dfu = pred[pred[cols["user_id"]].astype(str) == user_str].copy()
    # パラメータをスキルに付与（P(correct)計算や posterior用）
    par_use = par.copy()
    par_use["skill_name"] = par_use["skill_name"].astype(str)
    dfu[cols["skill_name"]] = dfu[cols["skill_name"]].astype(str)
    dfu = dfu.merge(par_use, how="left", left_on=cols["skill_name"], right_on="skill_name")

    skills = [s for s in skills if (dfu[cols["skill_name"]] == str(s)).any()]
    n = len(skills)
    if n == 0:
        print("[warn] プロット対象のスキル行が見つかりません。", file=sys.stderr)
        return
    ncols = 2 if n >= 2 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3.6*nrows), squeeze=False)

    for i, sk in enumerate(skills):
        ax = axes[i//ncols][i%ncols]
        sub = dfu[dfu[cols["skill_name"]] == str(sk)].sort_values(cols["order_id"]).copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(sub))
        pL = sub["state_predictions"].astype(float).to_numpy()
        # p_correct (予測) は列があればそれ、なければ p_S/p_G から計算
        if "_pred_correct_now" in sub.columns:
            pC = sub["_pred_correct_now"].astype(float).to_numpy()
        else:
            pS = sub["p_S"].astype(float).to_numpy()
            pG = sub["p_G"].astype(float).to_numpy()
            pC = (1.0 - pS) * pL + pG * (1.0 - pL)
        y = sub[cols["correct"]].astype(int).to_numpy() if cols["correct"] in sub.columns else sub["correct"].astype(int).to_numpy()
        # 観測後 posterior の参考線（p_S/p_G がある時のみ）
        pS = sub["p_S"].astype(float)
        pG = sub["p_G"].astype(float)
        if pS.notna().any() and pG.notna().any():
            p_post = []
            for pp, yy, ss, gg in zip(pL, y, pS.fillna(0.0), pG.fillna(0.0)):
                if yy == 1:
                    num = pp*(1-ss); den = pp*(1-ss) + (1-pp)*gg
                else:
                    num = pp*ss;     den = pp*ss + (1-pp)*(1-gg)
                p_post.append(num/(den+1e-12))
            p_post = np.array(p_post)
        else:
            p_post = None

        ax.plot(x, pL, linewidth=2.5, label="P(L) prior")
        ax.plot(x, pC, linestyle="--", linewidth=1.8, label="P(correct)")
        if p_post is not None:
            ax.plot(x, p_post, linewidth=1.8, alpha=0.7, label="P(L) posterior")
        # 正誤散布
        ax.scatter(x[y==1], np.ones((y==1).sum()), marker="o", s=18, label="correct", alpha=0.8)
        ax.scatter(x[y==0], np.zeros((y==0).sum()), marker="x", s=22, label="incorrect", alpha=0.8)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, len(x)-0.5)
        ax.grid(True, alpha=0.3)
        n_obs = int(sub.shape[0])
        ax.set_title(f"user={user_str}  skill={sk}  n={n_obs}")
        if i == 0:
            ax.legend(loc="best", fontsize=9)

    # 余った軸を消す
    for j in range(i+1, nrows*ncols):
        axes[j//ncols][j%ncols].set_visible(False)

    plt.tight_layout()
    # ensure output directory exists
    try:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"[warn] 出力ディレクトリ作成に失敗しました: {e}", file=sys.stderr)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
