# ファイル名例: bkt_with_pybkt.py  （※ pyBKT.py などパッケージ名と同名はNG）
import pandas as pd
import numpy as np
from pyBKT.models import Model, Roster  # Rosterが無ければ後述のB案で

def main():
    # === データ読み込み ===
    df = pd.read_csv("../csv/200_50_500_100000.csv")

    # ここで date の厳密な型変換が不要なら、to_datetime は省略してOK
    # df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 単一スキルのみに絞る（カンマ区切りを除外）
    df = df[df["skill"].astype(str).str.count(",") == 0].copy()

    # 列を pyBKT に合わせる
    df["skill_name"] = df["skill"].astype(str)
    df["Anon Student Id"] = df["user_id"]
    # 安定ソートで順序保証
    df = df.sort_values(["Anon Student Id", "skill_name", "date"], kind="mergesort")
    df["order_id"] = df.groupby(["Anon Student Id", "skill_name"]).cumcount() + 1
    df["correct"] = df["result"].astype(int)

    defaults = {
        "order_id": "order_id",
        "skill_name": "skill_name",
        "correct": "correct",
        "user_id": "Anon Student Id",
    }

    # === 並列OFFで学習（Windows安心設定） ===
    m = Model(seed=0, num_fits=1, parallel=False)
    m.fit(data=df, defaults=defaults, parallel=False)

    # === 一人のユーザに対して「次問正答確率」を出す例 ===
    uid = 68
    rows = []
    user_df = df[df["Anon Student Id"] == uid]
    for skill, sdf in user_df.groupby("skill_name"):
        obs = sdf.sort_values("date")["correct"].to_numpy()
        try:
            roster = Roster(students=[uid], skills=skill, model=m)
            state = roster.update_state(skill, uid, obs)
            p_next = float(state.get_correct_prob())
            pL_post = float(state.get_mastery_prob())
        except Exception:
            # Rosterが無い/不調なら predict + 手計算に切替（B案）
            preds = m.predict(data=sdf, defaults=defaults)
            # 直近の修得確率
            if "state_predictions" in preds.columns:
                pL_post = float(preds["state_predictions"].iloc[-1])
            else:
                pL_post = float(preds["correct_predictions"].iloc[-1])
            params = m.params()
            p_T = params["learns"][skill]
            p_S = params["slips"][skill]
            p_G = params["guesses"][skill]
            pL_next = pL_post + (1 - pL_post) * p_T
            p_next = (1 - p_S) * pL_next + p_G * (1 - pL_next)

        rows.append({
            "user_id": uid, "skill": skill, "n_obs": int(len(obs)),
            "P(L)_posterior": pL_post, "P(next correct)": p_next
        })

    out = pd.DataFrame(rows).sort_values("P(next correct)", ascending=False)
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    # Windowsのexe化などを考慮するなら freeze_support() もOK
    # import multiprocessing as mp; mp.freeze_support()
    main()
