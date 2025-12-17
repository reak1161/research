

import os, sys, time
import pandas as pd
import numpy as np
from pyBKT.models import Model
try:
    from pyBKT.models import Roster
    HAS_ROSTER = True
except Exception:
    HAS_ROSTER = False
# （以下はあなたの Safe Mode 本文そのまま）


# ==== 追加（pyBKT import の“あと”に置く）====
import multiprocessing.pool as mp_pool
import multiprocessing.dummy as mp_dummy
from pyBKT.fit import EM_fit  # ← ここ重要：EM_fit 内で import された Pool も上書きしたい

# 退避
_ORIG_POOL = mp_pool.Pool
_ORIG_EMFIT_POOL = getattr(EM_fit, "Pool", None)

# context= を受け取れる ThreadPool 互換ラッパ
def _ThreadPoolCompat(processes=None, initializer=None, initargs=(), maxtasksperchild=None, context=None):
    # multiprocessing.dummy.Pool は context/maxtasksperchild を受けないので無視してOK
    return mp_dummy.Pool(processes=processes, initializer=initializer, initargs=initargs)

class UseThreadPoolDuringFit:
    def __enter__(self):
        # multiprocessing.context.Pool -> multiprocessing.pool.Pool(context=...) 経由なので
        # “pool.Pool” を互換関数で置換し、かつ EM_fit がキャプチャしている Pool も上書き
        mp_pool.Pool = _ThreadPoolCompat
        try:
            EM_fit.Pool = _ThreadPoolCompat
        except Exception:
            pass
    def __exit__(self, exc_type, exc, tb):
        # 元に戻す
        mp_pool.Pool = _ORIG_POOL
        if _ORIG_EMFIT_POOL is not None:
            EM_fit.Pool = _ORIG_EMFIT_POOL
# ==============================================



def log(msg):
    print(msg, flush=True)

def main(smoke=True, top_k_skills=5, target_uid=None):
    import faulthandler
    faulthandler.enable()
    # 60秒ごとにスレッドダンプ（ハング位置の可視化）
    faulthandler.dump_traceback_later(60, repeat=True)

    t0 = time.perf_counter()
    log("[1/6] Loading CSV ...")

    # ★最初は軽量実行（smoke=Trueで先頭2万行だけ読む）
    csv_path = "../csv/200_50_500_100000.csv"  # あなたのCSVに合わせて
    if smoke:
        df = pd.read_csv(csv_path, nrows=20000)
    else:
        df = pd.read_csv(csv_path)

    # 必要最低限の前処理（重い日付パースは一旦省略）
    # df["date"] = pd.to_datetime(df["date"], errors="coerce")  # 必要なら有効化
    if "date" not in df.columns:
        # 日付が無いなら並び替え用のダミー順序をつくる
        df["date"] = np.arange(len(df))

    # 単一スキルのみ
    df = df[df["skill"].astype(str).str.count(",") == 0].copy()
    df["skill_name"] = df["skill"].astype(str)
    df["Anon Student Id"] = df["user_id"]
    df = df.sort_values(["Anon Student Id", "skill_name", "date"], kind="mergesort")
    df["order_id"] = df.groupby(["Anon Student Id", "skill_name"]).cumcount() + 1
    df["correct"] = df["result"].astype(int)

    # 多すぎると遅いので、まずは頻度上位のスキルだけに絞る
    vc = df["skill_name"].value_counts()
    keep_skills = vc.index[:top_k_skills].tolist()
    df = df[df["skill_name"].isin(keep_skills)].copy()

    defaults = {
        "order_id": "order_id",
        "skill_name": "skill_name",
        "correct": "correct",
        "user_id": "Anon Student Id",
    }

    log(f"[2/6] Rows={len(df):,}, Users={df['Anon Student Id'].nunique()}, "
        f"Skills={df['skill_name'].nunique()} (top {top_k_skills})")

    # ===== 学習 =====
    t1 = time.perf_counter()
    log("[3/6] Fitting pyBKT (num_fits=1, parallel=False) ...")

    
    m = Model(seed=0, num_fits=1, parallel=False)

    with UseThreadPoolDuringFit():      # ★ この1行だけでOK
        m.fit(data=df, defaults=defaults, parallel=False)


    t2 = time.perf_counter()
    log(f"[4/6] Fit done in {t2 - t1:.2f}s  ({len(df)/(t2-t1):,.0f} rows/s)")

    # ===== 予測（次問正答確率） =====
    # 適当に1名選ぶ or 指定ユーザ
    if target_uid is None:
        target_uid = int(df["Anon Student Id"].iloc[0])
    user_df = df[df["Anon Student Id"] == target_uid]
    if user_df.empty:
        log(f"[WARN] user {target_uid} has no rows in filtered data.")
        return

    rows = []
    log(f"[5/6] Predicting for user={target_uid} ...")
    for skill, sdf in user_df.groupby("skill_name"):
        sdf = sdf.sort_values("date", kind="mergesort")
        obs = sdf["correct"].to_numpy()
        try:
            if HAS_ROSTER:
                roster = Roster(students=[target_uid], skills=skill, model=m)
                state = roster.update_state(skill, target_uid, obs)
                p_next = float(state.get_correct_prob())
                pL_post = float(state.get_mastery_prob())
            else:
                preds = m.predict(data=sdf, defaults=defaults)  # 各時刻の正答確率など
                if "state_predictions" in preds.columns:
                    pL_post = float(preds["state_predictions"].iloc[-1])
                else:
                    # 最悪のフォールバック：正答確率を近似
                    pL_post = float(np.clip(preds["correct_predictions"].iloc[-1], 1e-6, 1-1e-6))
                params = m.params()
                p_T = params["learns"][skill]
                p_S = params["slips"][skill]
                p_G = params["guesses"][skill]
                pL_next = pL_post + (1 - pL_post) * p_T
                p_next = (1 - p_S) * pL_next + p_G * (1 - pL_next)

            rows.append({
                "user_id": target_uid, "skill": skill, "n_obs": int(len(obs)),
                "P(L)_posterior": pL_post, "P(next correct)": p_next
            })
        except Exception as e:
            log(f"[ERROR] skill={skill}: {e}")

    out = pd.DataFrame(rows).sort_values("P(next correct)", ascending=False)
    log("[6/6] Result (top 10):")
    # 強制フラッシュしてすぐ見えるように
    print(out.head(10).to_string(index=False), flush=True)

    t3 = time.perf_counter()
    log(f"[DONE] Total time {t3 - t0:.2f}s")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main(smoke=True, top_k_skills=5)
