from bkt_visualize import compute_bkt_log, plot_bkt_progress
import pandas as pd

df = pd.read_csv("interactions.csv")  # 必須列: user_id, skill_id, correct(0/1), 任意: t or timestamp
params = pd.read_csv("skill_params.csv")  # 必須列: skill_id, p_init, learn, slip, guess

log_df = compute_bkt_log(
    df,
    params,
    time_col="t",        # 無ければ自動で並び替え
    correct_col="correct"
)

# 学習者を2人（68, 58）、スキルを2つ（6, 24）に絞って出力、画像保存もする
plot_bkt_progress(
    log_df,
    user_ids=[68, 58],
    skill_ids=[6, 24],
    save_dir="figs",          # 保存しないなら None
    show=True                 # Jupyter/VScode で即表示
)
