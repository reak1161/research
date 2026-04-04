CLI Quiz Distribution Set

How to run:
  ./run_quiz.sh <user_id>
  ./run_full_flow.sh <user_id> [llm_model]
  Example:
    ./run_full_flow.sh user000 gpt-4o-mini

What run_full_flow.sh does:
1) Fit BKT params from csv/sim_logs.csv
2) Fit IRT params from csv/sim_logs.csv
3) Build runs/sim_user_item_scores.csv
4) Start interactive quiz

Notes:
- Uses OpenAI via OPENAI_API_KEY env var.
- Logs are written to csv/sim_online_logs.csv
- Initial history is loaded from csv/sim_logs.csv

Files:
- code/ : CLI scripts
- csv/items_sample_lpic_tier.csv : item bank
- csv/bkt_params_multi.csv : BKT params
- csv/irt_items_estimated.csv : IRT item params
- csv/sim_logs.csv : initial history
- csv/sim_online_logs.csv : output log (created if missing)
