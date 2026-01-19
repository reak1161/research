CLI Quiz Distribution Set

How to run:
  ./run_quiz.sh <user_id>

Notes:
- Uses OpenAI via OPENAI_API_KEY env var.
- Logs are written to csv/sim_online_logs.csv
- Initial history is loaded from csv/sim_logs.csv

Files:
- code/ : CLI scripts
- csv/items_sample_lpic_tier.csv : item bank
- csv/bkt_params_multi.csv : BKT params
- csv/sim_logs.csv : initial history
- csv/sim_online_logs.csv : output log (created if missing)
