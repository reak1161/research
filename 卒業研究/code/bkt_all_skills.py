# bkt_all_skills.py — 全スキル一括：multiprior 基準、必要なら multilearn/KT-IDEM も切替可
from pyBKT.models import Model
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import time

from pathlib import Path
import re

import warnings
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    module=r"pyBKT\.fit\.EM_fit"
)

# ===== 拡張トグル（ここだけON/OFFすれば全体に効きます） =====
USE_MULTIPRIOR = True            # 学習者ごとに prior を分ける（"user_id" を渡す）
USE_MULTILEARN = False           # 問題グループごとに learns を分ける（"problem_name" を渡す）
USE_KT_IDEM   = False            # アイテム別 guess/slip（KT-IDEM）
MIN_ITEM_N    = 5                # KT-IDEM時、各itemの最小出現数
# ============================================================


#CSV = r"data/ct.csv"
CSV = r"data/ct_from_in1e5_de1.csv"

def load_base(csv=CSV):
    df = pd.read_csv(csv, encoding="latin-1").rename(columns={
        "Anon Student Id": "user_id",
        "KC(Default)": "skill_name",
        "Problem Name": "problem_name",
        "Step Name": "step_name",
        "Correct First Attempt": "correct",
        "First Transaction Time": "ts",
    })
    # --- 正誤を0/1に ---
    t = {1,"1",True,"true","True","correct","Correct","Yes","yes"}
    f = {0,"0",False,"false","False","incorrect","Incorrect","No","no"}
    df = df[df["correct"].isin(t|f)].copy()
    df["correct"] = df["correct"].apply(lambda x: 1 if x in t else 0).astype(int)

    # === ここから：複数スキル対応（KC分解→explode） ===
    df["skill_raw"] = df["skill_name"].astype(str)

    def split_skills(s: str):
        s = s.strip()
        # 代表的な区切りを順にチェック（ASSISTments: "~~" が多い）
        for sep in ("~~", ";", "|", ","):
            if sep in s:
                toks = [tok.strip() for tok in s.split(sep)]
                return [tok for tok in toks if tok]
        return [s] if s else []

    df["__skills"] = df["skill_raw"].map(split_skills)
    df = df.explode("__skills")
    df["skill_name"] = df["__skills"].astype(str).str.strip()
    df = df.drop(columns=["__skills"])

    # 空/NAスキルを落とす
    df = df[df["skill_name"].notna() & (df["skill_name"] != "")]
    # === ここまで ===

    # --- ts が無い/壊れている場合のフォールバック ---
    if "ts" not in df.columns:
        ts = None
        for cand in ["First Transaction Time", "Step Start Time", "Correct Transaction Time",
                     "timestamp", "date", "time"]:
            if cand in df.columns:
                ts = pd.to_datetime(df[cand], errors="coerce"); break
        if ts is None or ts.isna().mean() > 0.5:
            df["_ord"] = df.groupby("user_id").cumcount()
            ts = pd.to_datetime(df["_ord"], unit="s", origin="2024-01-01")
            df.drop(columns=["_ord"], inplace=True)
        df["ts"] = ts

    # 時刻と並び替え
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    df = ensure_ts(df, user_col="user_id")

    df = df.sort_values(["user_id","ts","problem_name","step_name"], kind="mergesort")

    # （任意）KC分解後に opp を再計算したい場合だけ↓を有効化
    # df["Opportunity(Default)"] = df.groupby(["user_id","skill_name"]).cumcount() + 1

    # スキル全体が全0/全1は除外
    ok_skill = df.groupby("skill_name")["correct"].transform(lambda s: s.max()!=s.min())
    return df[ok_skill].copy()


def fit_one_skill(df_skill, skill):
    
    df_skill = ensure_ts(df_skill, user_col="user_id")  # ← 追加

    # === item 粒度（KT-IDEMなら Problem+Step を連結、そうでなければ Problem名） ===
    if USE_KT_IDEM:
        df_skill = df_skill.copy()
        df_skill["problem_id"] = df_skill["problem_name"].astype(str) + " :: " + df_skill["step_name"].astype(str)
    else:
        df_skill = df_skill.copy()
        df_skill["problem_id"] = df_skill["problem_name"].astype(str)   # ← コピーにする

    # === 安定化フィルタ（ユーザの極端系列は除外） ===
    by_user = df_skill.groupby("user_id")["correct"]
    df_skill = df_skill[by_user.transform(lambda s: s.max() != s.min())].copy()

    # KT-IDEM時のみ：極端/少数 item を軽く除外
    if USE_KT_IDEM:
        by_item = df_skill.groupby("problem_id")["correct"]
        df_skill = df_skill[by_item.transform(lambda s: s.max() != s.min())].copy()
        cnt_item = df_skill.groupby("problem_id")["correct"].transform("size")
        df_skill = df_skill[cnt_item >= MIN_ITEM_N].copy()

    # データが小さすぎたらスキップ（安全弁）
    if len(df_skill) < 30 or df_skill["correct"].nunique() < 2:
        raise RuntimeError("too few usable rows after stabilization")


    defaults = {"prior":0.15, "learns":0.08, "forgets":0.0, "guesses":0.20, "slips":0.10}
    m = Model(seed=42, num_fits=1, parallel=False)   # ← Windowsは並列OFF

    m.fit(
        data=df_skill, skills=skill,
        multigs=("problem_id" if USE_KT_IDEM else False),   # ← True ではなく列名 or False
        multiprior=("user_id" if USE_MULTIPRIOR else False),
        multilearn=("problem_name" if USE_MULTILEARN else False),
        forgets=False, defaults=defaults, parallel=False
    )
    
    
    # 予測と指標
    yhat = m.predict(data=df_skill)
    y = df_skill["correct"].to_numpy(int)
    p = np.clip(yhat["correct_predictions"].to_numpy(float), 1e-15, 1-1e-15)
    auc = roc_auc_score(y, p)
    ll  = log_loss(y, p)
    acc = accuracy_score(y, (p>=0.5).astype(int))
    return m, auc, ll, acc, yhat

def main():

    start = time.perf_counter() #計測開始

    run_id = time.strftime("%Y%m%d_%H%M%S")           # 実行ごとに時刻でフォルダ分け
    OUTDIR = Path("runs") / run_id
    PRED_DIR = OUTDIR / "pred"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    def safe_name(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z._-]+", "_", s)[:80]

    df = load_base()
    skills = sorted(df["skill_name"].unique())
    print("skills:", len(skills), skills[:5], "...")

    rows = []
    all_params = []
    for i, skill in enumerate(skills, 1):
        sdf = df[df["skill_name"]==skill]
        try:
            m, auc, ll, acc, yhat = fit_one_skill(sdf, skill)
            rows.append({"skill":skill, "n":len(sdf), "auc":auc, "logloss":ll, "acc@0.5":acc})
            # パラメータを縦持ちで回収
            params = m.params()
            for param_name, table in params.items():
                # table は「スキル名→値」 or DataFrame 形式に近いもの
                if isinstance(table, dict) and skill in table:
                    all_params.append({"skill":skill, "param":param_name, "value":table[skill]})
            # 各スキルの予測を保存（必要なら）
            yhat[["user_id","problem_id","skill_name","correct",
                "correct_predictions","state_predictions"]].to_csv(
                PRED_DIR / f"pred_{i:02d}_{safe_name(skill)}.csv",
                index=False, encoding="utf-8"
            )

            print(f"[{i}/{len(skills)}] {skill}: AUC={auc:.3f} LL={ll:.3f} Acc={acc:.3f} (n={len(sdf)})")
        except Exception as e:
            print(f"[{i}/{len(skills)}] {skill}: ERROR -> {e}")

    pd.DataFrame(rows).to_csv(OUTDIR / "per_skill_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(all_params).to_csv(OUTDIR / "per_skill_params.csv", index=False, encoding="utf-8")
    print(f"saved to: {OUTDIR} (pred files are under {PRED_DIR})")


    # --- ここから追加：ユーザ×スキルの最新P(L)を集計して保存 ---
    all_pred = []
    for p in PRED_DIR.glob("pred_*.csv"):
        dfp = pd.read_csv(p, usecols=["user_id","skill_name","state_predictions"])
        # ファイル内の行順（=時系列順）でランク付け
        dfp["row_order"] = dfp.groupby(["user_id","skill_name"]).cumcount()
        all_pred.append(dfp)

    if all_pred:
        pred = pd.concat(all_pred, ignore_index=True)
        # 各 user×skill で最後の行（最新P(L)）だけ残す
        last = pred.sort_values("row_order").groupby(
            ["user_id","skill_name"], as_index=False
        ).tail(1).rename(columns={"state_predictions": "mastery"})
        last[["user_id","skill_name","mastery"]].to_csv(
            OUTDIR / "user_skill_mastery.csv", index=False, encoding="utf-8"
        )
        print("saved:", OUTDIR / "user_skill_mastery.csv")
    # --- 追加ここまで ---


    cp = time.perf_counter() # チェックポイント
    t = cp - start
    print(int(t/60), "min", t%60, "s")

    # --- 上位Kスキルを自動選定して OOS を回す ---
    K = 5  # 必要に応じて変更
    topk = (
        df.groupby("skill_name", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=False)
        .head(K)["skill_name"]
        .tolist()
    )

    for s in topk:
        sdf = df[df["skill_name"] == s]
        # テストが成立する最低条件
        if len(sdf) >= 50 and sdf["correct"].nunique() == 2:
            try:
                print(f"[OOS baseline] skill={s} ->",   fit_eval_oos(sdf, s, mode="baseline"))
                # 他モードを試すなら↓を適宜ON
                # print(f"[OOS multilearn] skill={s} ->", fit_eval_oos(sdf, s, mode="multilearn"))
                # print(f"[OOS multiprior] skill={s} ->", fit_eval_oos(sdf, s, mode="multiprior"))
                # print(f"[OOS ktidem]    skill={s} ->", fit_eval_oos(sdf, s, mode="ktidem"))
            except Exception as e:
                print(f"[OOS] skill={s} skipped due to: {e}")
        else:
            print(f"[OOS] skill={s} skipped (too small or degenerate)")



# --- 置き換え：単一モード専用の OOS 評価 ---
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np

def time_holdout_by_user(df_skill, train_ratio=0.8):
    parts = []
    for uid, g in df_skill.sort_values("ts").groupby("user_id", sort=False):
        n = len(g); k = max(1, int(n * train_ratio))
        parts.append((g.iloc[:k].copy(), g.iloc[k:].copy()))
    train = pd.concat([a for a,b in parts if len(a)], ignore_index=True)
    test  = pd.concat([b for a,b in parts if len(b)], ignore_index=True)
    return train, test


def ensure_ts(df: pd.DataFrame, user_col="user_id") -> pd.DataFrame:
    """
    df に 'ts' (datetime64) 列を必ず作る。
    候補列があればそれを使い、なければユーザ内の行順から擬似タイムスタンプを生成。
    """
    if "ts" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts"]):
        return df

    # 候補から作る
    cand_cols = [
        "ts", "First Transaction Time", "Step Start Time",
        "Correct Transaction Time", "timestamp", "date", "time"
    ]
    ts = None
    for c in cand_cols:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce")
            break

    # 候補が無い/NaT が多すぎる場合は擬似的に生成
    if ts is None or ts.isna().mean() > 0.5:
        order = df.groupby(user_col).cumcount()
        ts = pd.to_datetime(order, unit="s", origin="2024-01-01")

    df = df.copy()
    df["ts"] = ts
    # 型保証
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


def ensure_time_order(df: pd.DataFrame, user_col="user_id", time_col="ts") -> pd.DataFrame:
    """
    ユーザ内の時系列順序を必ず作る。
    ts が有効なら ts で並べ、無ければ元の行順で並べつつ、__ord を付与する。
    """
    df = df.copy()
    if time_col in df.columns and df[time_col].notna().any():
        # ts が有効にある場合はそれでソート
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values([user_col, time_col], kind="mergesort")
    else:
        # ts が無い/全部NaT の場合は、ユーザ単位で元の順序を保ったまま
        df = df.sort_values([user_col], kind="mergesort")
    # ユーザ内の順序インデックス
    df["__ord"] = df.groupby(user_col).cumcount()
    return df


def fit_eval_oos(df_skill, skill, mode="baseline", min_item_n=5, train_ratio=0.8):
    df_skill = ensure_ts(df_skill, user_col="user_id")          # ← 既存
    df_skill = ensure_time_order(df_skill, user_col="user_id")  # ← これを追加

    # --- item 列の用意（既存のまま） ---
    if mode == "ktidem":
        df_skill["problem_id"] = df_skill["problem_name"].astype(str) + " :: " + df_skill["step_name"].astype(str)
    else:
        df_skill["problem_id"] = df_skill["problem_name"].astype(str)

    # ユーザ極端系列の除外（既存のまま）
    by_user = df_skill.groupby("user_id")["correct"]
    df_skill = df_skill[by_user.transform(lambda s: s.max()!=s.min())].copy()

    # KT-IDEM の極端/少数アイテム除外（既存のまま）
    if mode == "ktidem":
        by_item = df_skill.groupby("problem_id")["correct"]
        df_skill = df_skill[by_item.transform(lambda s: s.max()!=s.min())].copy()
        cnt_item = df_skill.groupby("problem_id")["correct"].transform("size")
        df_skill = df_skill[cnt_item >= min_item_n].copy()

    # ★★★ ここを置き換え：ts ではなく __ord を使ってホールドアウト ★★★
    parts = []
    for uid, g in df_skill.groupby("user_id", sort=False):
        g = g.sort_values("__ord", kind="mergesort")
        n = len(g)
        k = max(2, int(n*train_ratio))
        if k >= n:
            k = n-1
        parts.append((g.iloc[:k].copy(), g.iloc[k:].copy()))
    train = pd.concat([a for a,b in parts if len(a)], ignore_index=True)
    test  = pd.concat([b for a,b in parts if len(b)], ignore_index=True)

    # ★ KT-IDEM: train の item カテゴリに test を揃える（ズレ防止）
    if mode == "ktidem":
        train["problem_id"] = train["problem_id"].astype("category")
        cats = train["problem_id"].cat.categories
        test = test[test["problem_id"].isin(cats)].copy()
        test["problem_id"] = pd.Categorical(test["problem_id"], categories=cats)

    # multilearn: 未学習グループを test から除外（既に実装済みならそのまま）
    if mode == "multilearn":
        seen = set(train["problem_name"].unique())
        test = test[test["problem_name"].isin(seen)].copy()

    if len(test) < 30 or test["correct"].nunique() < 2:
        raise RuntimeError("test fold too small/degenerate after filtering")

    defaults = {"prior":0.15, "learns":0.08, "forgets":0.0, "guesses":0.20, "slips":0.10}
    m = Model(seed=42, num_fits=1, parallel=False)

    # 1モードだけ指定
    if mode == "baseline":
        kwargs = dict(multigs=False,        multiprior=False,         multilearn=False)
    elif mode == "multiprior":
        kwargs = dict(multigs=False,        multiprior="user_id",     multilearn=False)
    elif mode == "multilearn":
        kwargs = dict(multigs=False,        multiprior=False,         multilearn="problem_name")
    elif mode == "ktidem":
        kwargs = dict(multigs="problem_id", multiprior=False,         multilearn=False)
    else:
        raise ValueError("unknown mode")

    m.fit(data=train, skills=skill, forgets=False, defaults=defaults, parallel=False, **kwargs)

    # ★ preload=True を付けて、学習時のリソース配列を使う
    yhat = m.predict(data=test)

    # スコア
    import numpy as np
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
    y = test["correct"].to_numpy(int)
    p = np.clip(yhat["correct_predictions"].to_numpy(float), 1e-15, 1-1e-15)

    from sklearn.metrics import roc_auc_score, log_loss

    # 既存：y = test["correct"].to_numpy(int)
    # 既存：p = np.clip(yhat["correct_predictions"].to_numpy(float), 1e-15, 1-1e-15)

    ll      = log_loss(y, p)
    ll_flip = log_loss(1 - y, p)
    auc     = roc_auc_score(y, p)
    auc_flp = roc_auc_score(1 - y, p)

    print(f"[label check] LL(as-is)={ll:.4f}  LL(flip)={ll_flip:.4f}  "
        f"AUC(as-is)={auc:.3f}  AUC(flip)={auc_flp:.3f}")

    if ll_flip + 1e-6 < ll:
        print("→ ひっくり返した方が良い（ラベルの向きが逆の可能性が高い）")
    else:
        print("→ 現状の向きでOK")


    pos_rate = test["correct"].mean()
    print(f"[{mode}] test size={len(test)}, positive_rate={pos_rate:.3f}")
    return {"auc": roc_auc_score(y, p), "logloss": log_loss(y, p), "acc@0.5": (p>=0.5).astype(int).mean(),
            "n_train": len(train), "n_test": len(test)}




if __name__ == "__main__":

    start = time.perf_counter() #計測開始
    
    main()

    end = time.perf_counter() #計測終了
    t = end - start
    print(int(t/60), "min", t%60, "s")