CLI Quiz Distribution Set

How to run:
  ./run_generate_sample.sh [n_users] [interactions_per_user] [seed]
  ./run_fit_bkt.sh [log_csv] [out_params_csv] [out_report_csv]
  ./run_fit_irt.sh [log_csv] [out_items_csv] [out_theta_csv]
  ./run_offline_predict.sh [log_csv] [out_scores_csv]
  ./run_evaluate.sh [scores_csv] [out_metrics_csv]
  ./run_build_payloads.sh [scores_csv] [out_dir] [top_k] [max_candidates]
  ./run_plot_history.sh <user_id> <skill_name> [log_csv] [params_csv] [out_dir]
  ./run_render_selection_chain.sh [json_dir] [user_id] [out_prefix] [fmt]
  ./run_quiz.sh <user_id>
  ./run_full_flow.sh
  Example:
    ./run_generate_sample.sh 80 120 42
    ./run_full_flow.sh

What run_full_flow.sh does:
1) Fit BKT params from csv/sim_logs.csv
2) Fit IRT params from csv/sim_logs.csv
3) Build runs/sim_user_item_scores.csv
4) Finish (no quiz start)

Notes:
- Uses OpenAI via OPENAI_API_KEY env var.
- Logs are written to csv/sim_online_logs.csv
- Initial history is loaded from csv/sim_logs.csv

Suggested order:
1) ./run_generate_sample.sh
2) ./run_fit_bkt.sh
3) ./run_fit_irt.sh
4) ./run_offline_predict.sh
5) ./run_evaluate.sh
6) ./run_quiz.sh <user_id>
7) (optional) ./run_plot_history.sh <user_id> <skill_name>
8) (optional) ./run_render_selection_chain.sh

Files:
- code/ : CLI scripts
- csv/items_sample_lpic_tier.csv : item bank
- csv/bkt_params_multi.csv : BKT params
- csv/irt_items_estimated.csv : IRT item params
- csv/sim_logs.csv : initial history
- csv/sim_online_logs.csv : output log (created if missing)
