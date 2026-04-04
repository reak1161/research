# 引き継ぎ用ミニマム構成（卒業研究）

このディレクトリは、今回の研究で実際に使った処理を **最小限** で再実行できるように整理したものです。  
対象は次の流れです。

1. 疑似ログ生成（任意）
2. BKT パラメータ推定
3. IRT（3PL）パラメータ推定
4. BKT+IRT のオフライン正答確率推定（`P_final`）
5. 指標評価（AUC / Logloss / MSE / RMSE）
6. CLI チューター実行（オンライン更新）

---

## ディレクトリ構成

```text
卒業研究/
  code/
    bkt_core.py
    irt_core.py
    data_io.py
    llm_client.py
    llm_payload.py
    simulate_user_logs.py
    offline_fit_bkt.py
    fit_irt_params.py
    offline_predict_scores.py
    evaluate_scores.py
    interactive_lpic_quiz.py
    plot_user_history.py
    report_llm_latency.py
  csv/
    sim_online_logs.csv        # 初回起動用ヘッダのみ作成済み
  runs/
  run_generate_sample.sh
  run_fit_bkt.sh
  run_fit_irt.sh
  run_offline_predict.sh
  run_evaluate.sh
  run_full_flow.sh
  run_quiz.sh
  run_plot_history.sh
  run_llm_latency.sh
  requirements.txt
```

---

## 前提環境

- Python 3.10 系（3.11 でも概ね可）
- Linux / macOS（bash 前提）

### セットアップ

```bash
cd 卒業研究
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

OpenAI を使う場合のみ API キーを設定してください。

```bash
export OPENAI_API_KEY="..."
```

---

## 必須CSV（最小）

この構成では、以下の入力を想定します。

- `csv/items_sample_lpic_tier.csv`（問題マスタ）
- `csv/sim_logs.csv`（学習ログ）

`sim_logs.csv` が無い場合は `run_generate_sample.sh` で作れます。

### すでに同梱済みの推定結果（すぐ実行できるように配置）

- `csv/bkt_params_multi.csv`
- `csv/irt_items_estimated.csv`
- `csv/irt_theta_estimated.csv`
- `runs/sim_user_item_scores.csv`
- `runs/sim_metrics_overall.csv`
- `runs/bkt_metrics.csv`
- `runs/irt_fit_history.csv`

---

## 実行手順（標準）

### 0) 疑似ログ生成（必要な場合のみ）

```bash
./run_generate_sample.sh 80 120 42
```

- 80ユーザー
- 1ユーザーあたり120インタラクション
- seed=42

### 1) BKT推定

```bash
./run_fit_bkt.sh
```

出力:
- `csv/bkt_params_multi.csv`
- `runs/bkt_metrics.csv`

### 2) IRT推定

```bash
./run_fit_irt.sh
```

出力:
- `csv/irt_items_estimated.csv`
- `csv/irt_theta_estimated.csv`
- `runs/irt_fit_history.csv`
- `runs/irt_fit_plots/*`

### 3) オフライン正答確率（P_final）推定

```bash
./run_offline_predict.sh
```

出力:
- `runs/sim_user_item_scores.csv`

### 4) 評価

```bash
./run_evaluate.sh
```

出力:
- `runs/sim_metrics_overall.csv`

### 5) CLIチューター（オンライン）

```bash
./run_quiz.sh user000
```

出力:
- `csv/sim_online_logs.csv`（追記）
- `runs/json_history/<user_id>/...`（LLM入出力履歴）

---

## 主要プログラムの役割

- `simulate_user_logs.py`  
  問題マスタから疑似ログを生成します。

- `offline_fit_bkt.py` + `bkt_core.py`  
  L2 / L2_tier のBKTパラメータを推定します。

- `fit_irt_params.py` + `irt_core.py`  
  3PLの項目パラメータ（a,b,c）とユーザー能力（theta）を推定します。

- `offline_predict_scores.py`  
  BKT（L2, L2_tier）とIRTを線形結合して `P_final` を算出します。

- `evaluate_scores.py`  
  予測結果に対して AUC/Logloss/MSE/RMSE を計算します。

- `interactive_lpic_quiz.py`  
  CLIで出題・採点・BKTオンライン更新・LLM推薦/フィードバックを実行します。

---

## 引き継ぎ時の注意点

1. **列名依存**  
   CSVの列名（`item_id`, `L2`, `tier`, `correct`, `timestamp` など）が変わると失敗しやすいです。

2. **OpenAIキー**  
   配布先にキーを渡さない運用にする場合は、`--use-llm` を外して実行してください。

3. **日本語フォント警告（IRTプロット）**  
   環境によっては日本語グリフ警告が出ますが、CSVと画像出力自体は可能です。

4. **オンラインログ初期化**  
   `csv/sim_online_logs.csv` はヘッダだけ作成済みです。初回実行でも読み込みエラーになりにくくしています。

---

## 最短実行（前処理を一括）

```bash
./run_full_flow.sh
```

このスクリプトは次を順に実行します。
- BKT推定
- IRT推定
- オフラインスコア計算

（クイズ開始はしません）
