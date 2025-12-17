# BKT × IRT 推薦システム要件ドキュメント

> **更新日**: 2025-11-19 (JST)  
> **目的**: BKT（Bayesian Knowledge Tracing）と IRT（Item Response Theory）を組み合わせ、学習ログから次問題の正答確率と推薦を行うシステムの要件を明文化する。

---

## 1. ビジョンと成功指標

### 1.1 プロダクトビジョン
- 学習者の行動ログを入力に、**分野ごとに正答確率と学習度を推定**し、難易度整合を保った問題推薦と短い日本語フィードバックを生成する。
- モデル推定から推薦 JSON、可視化、評価までを**再現性のある CLI パイプライン**として提供する。
- 将来的に IRT パラメータと連続能力推定を取り込み、BKT の離散遷移と IRT の連続能力の強みを統合する。

### 1.2 MVP の成功指標
- **AUC ≥ 0.65**, **Logloss ≤ 0.62**（単一分野平均、同一 seed で再現 ±0.001）。
- 推薦 JSON で `practice` 帯（P(next) 0.60–0.80）の問題を 5 件以内で返却し、帯外は review/challenge として区別。
- CLI で `--compare-forgets` により忘却オン/オフの差分を CSV でエクスポートできる。
- 学習度可視化（ユーザー×複数分野）を PNG 出力できる。

---

## 2. スコープと非スコープ

### 2.1 スコープ（MVP）
1. **データ前処理**: long 形式 CSV（`order_id,user_id,domain,correct`）を想定し、ユーザー内シーケンスを正規化。
2. **BKT モデリング**: 単一分野を単位とし、`L0,T,S,G` を EM で推定。`forgets` は比較用オプション。
3. **評価**: overall / domain / user / user×domain 粒度で AUC・Logloss を算出。
4. **推薦ロジック**: P(next) に基づく帯分けと分野サンプリング（`1 - P(L)_after_learn` 重み）。
5. **LLM 連携**: 問題候補＋分野エビデンスを渡し、「候補外 ID 禁止」ルールで最終選定と短文フィードバックを生成。
6. **再現性**: 依存バージョン固定（`numpy<2`, `pyBKT==1.4.1`, `pybind11>=2.12`）、`runs/yyyymmdd_HHMM/` 以下に成果物を整理。

### 2.2 非スコープ（今後検討）
- マルチスキルの Q マトリクス推定、深層モデル（DKT, SAKT 等）。
- オンライン学習やリアルタイム API。現状はバッチ CLI のみ。
- 自動で最適な LLM プロンプト生成・評価。
- IRT パラメータ推定自体（外部ツールまたはシミュレータで取得済みを前提）。

---

## 3. ユーザーとユースケース

| ロール | 目的 | 主な操作 |
| --- | --- | --- |
| 分析担当者 | 学習ログからモデルを学習し、分野別の学習度と正答確率を把握 | CLI で BKT 学習・評価、`runs/` の指標確認 |
| カリキュラム設計者 | 学習進捗を踏まえた次問題を選び、対話型学習に投入 | 推薦 JSON を教材管理システムへ転記、LLM で助言生成 |
| LLM パイプライン | 候補問題の中から最大 5 件を選定し、フィードバックを生成 | `P_for_llm`（BKT×IRT重み付き）と補足テキストを入力に使用 |

ユースケース例:
1. 定期的に CSV ログを取り込み、BKT を再学習して成績をトレンド確認。
2. 指定ユーザー×分野の習得度グラフを生成し、面談資料に使用。
3. BKT に IRT の項目パラメータを重ね、`practice/review/challenge` のバランスで推薦 JSON を自動生成。

---

## 4. データ要件

### 4.1 入力 CSV（long 形式）

| 列名 | 型 / 例 | 必須 | 説明 |
| --- | --- | --- | --- |
| `order_id` | int または ISO 日時 | ✔ | ユーザー内の解答順序。非整数の場合は内部で連番化。 |
| `user_id` | str/int | ✔ | 学習者 ID。 |
| `domain` | str | ✔ | 分野（旧 skill）。CLI では `--skill-col` で指定。 |
| `item_id` | str/int | △ | 推薦や IRT 連携で使用。 |
| `correct` | 0/1 | ✔ | 正誤。 |
| `timestamp` | ISO | △ | ログ順序の補助。 |
| `tags / difficulty_hint` | str | △ | 分野内でのタグ・難易度（IRT 補助）。 |

品質要件:
- 全体正答率は **0.55–0.70** に収まるようデータ生成 or フィルタリング。
- 分野あたりの最小観測数を設け、極端に少ない分野は警告のうえ除外。
- 単一分野 CSV を束ねる際は、列名を統一し BOM なし UTF-8 を使用。

### 4.2 シミュレーション / IRT データ
- IRT 項目パラメータ: `item_id, domain, a, b, c`。
- 能力推定: `user_id, domain, theta`。
- 問題マスタ: `item_id, domain, stem, solution_outline, hint`。
- 生成器は θ 分布、スキル結合、当て推量 `c`、難易度 `b`、識別力 `a` を制御可能とする。

---

## 5. システム構成（論理）

1. **Data Prep**: CSV 読み込み、列マッピング、欠損と order 補正。
2. **BKT Trainer (`pybkt_user_domain.py`)**: モデル学習、`state_predictions` 抽出、パラメータ表生成。
3. **Evaluator**: 指標計算（AUC/Logloss）、忘却オンオフ比較、CSV 出力。
4. **Visualizer**: ユーザー×分野サブプロット（P(L) prior/posterior、P(correct)、観測点）。
5. **Recommendation Core**: P(next) による帯分け、分野サンプリング、多様性制約（同一分野 2 連まで）。
6. **LLM Adapter**: 候補問題・分野エビデンスを JSON 形式で出力し、LLM に最終選定とフィードバック生成を依頼。
7. **BKT × IRT Bridge (`bkt_irt_bridge.py`)**: BKT の P(next) と IRT の 3PL 予測を重み付きで合成し、`P_for_llm` を算出。

---

## 6. 機能要件詳細

### 6.1 Data Prep / CLI
- `--order-col/--user-col/--skill-col/--correct-col` で列名を切り替え可能。
- `order_id` が非整数の場合、ユーザー単位で昇順ソートし 0 始まりの連番を付与。
- 欠損/異常値:
  - `correct` が 0/1 以外なら除外し、ログに ID を記録。
  - `domain` 欠損行は自動ドロップ（要件: 95% 以上残存）。
- CLI から出力ディレクトリを明示 (`--out-dir runs/yyyymmdd_HHMM`)、メタ情報（seed, コマンド, Git commit）を JSON で保存。

### 6.2 BKT 学習・推論
- モデル: `pyBKT.models.Model` の `fit/predict` を使用。`estimation_procedure="EM"`。
- 推定パラメータは `skill_name, p_T, p_S, p_G, p_L0` で CSV 保存。複数行のときは平均で集約。
- 出力列:
  - `P(L)_prior`, `P(L)_posterior`, `P(L)_after_learn`
  - `P(correct_now)`, `P(next)` (= `p_next_bkt`)
  - `n_obs`, `recent_correct_seq`, `trend_last5`
- 忘却比較: `--compare-forgets` 指定時は `forgets=False/True` を同一データで学習し、差分表を生成。

### 6.3 推薦ロジック
- 推薦候補集合: 分野ごとに最新 `P(next)` を計算し、`expected_p_correct` とする。
- 帯区分:
  - `review`: `P(next) < 0.40`
  - `practice`: `0.60 ≤ P(next) ≤ 0.80`（最優先）
  - `challenge`: `P(next) > 0.80` かつ IRT 難易度 `b` がユーザー θ より高い問題
- 分野サンプリング: 重み `w_domain = 1 - P(L)_after_learn` を正規化し、低学習度ドメインを優先。
- 多様性: 連続同一 domain の推薦を最大 2 件までに制限（LLM プロンプトでルール明記）。
- 推薦 JSON 例:

```json
{
  "schema_version": "1.0",
  "user_id": 68,
  "generated_at": "2025-11-01T20:00:00+09:00",
  "objective_mode": "motivate",
  "recommendations": [
    {
      "item_id": "Q12345",
      "type": "practice",
      "expected_p_correct": 0.72,
      "domain": "fractions",
      "skill_evidence": {
        "p_L_after": 0.63,
        "p_next": 0.72,
        "trend_last5": "+-++?"
      },
      "feedback_to_student": "桁上がりの位置に注意して、途中式を一行増やそう。",
      "note_to_teacher": "直近で繰上りミス。P=0.72。"
    }
  ]
}
```

### 6.4 IRT との連携
- `bkt_irt_bridge.py` で以下を計算:
  - `P_bkt`: domain レベルの正答確率。
  - `P_irt`: 3PL (`c + (1-c) / (1 + exp(-a(θ-b)))`)。
  - `P_for_llm = w_bkt * P_bkt + (1-w_bkt) * P_irt`、欠損時は片方をそのまま利用。
- 入力 CSV は列名カスタマイズ可能 (`--col-bkt-user` など)。
- 出力: `user_id,item_id,domain,P_bkt,P_irt,P_for_llm`。必要に応じて問題メタを merge。

### 6.5 評価・可視化
- 評価コマンド例:

```bash
python pybkt_user_domain.py \
  --csv data/log.csv \
  --order-col date --user-col user_id --skill-col domain --correct-col correct \
  --compare-forgets --auc-by overall \
  --eval-out runs/auc_logloss_overall.csv
```

- 粒度別 CSV (`overall / domain / user / user×domain`) と、忘却オン/オフ差分 (`ΔAUC, ΔLogloss`) を保存。
- 可視化: `python pybkt_user_skill_report.py --user 68 --skills algebra fractions geometry ...`
  - 太線: `P(L)_prior`、点線: `P(correct)`、散布: 実観測 0/1。
  - タイトル: `user=68 domain=fractions n=42` の形式。

---

## 7. 非機能要件

| 項目 | 要件 |
| --- | --- |
| OS/環境 | Windows 11 上の **WSL2 Ubuntu 24.04**。Python 3.10.13（pyenv）, 仮想環境 `~/repos/research/.venv`。 |
| 依存ライブラリ | `pyBKT==1.4.1`, `numpy<2`, `pybind11>=2.12`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`（AUC）。 |
| パフォーマンス | 10 万レコード・10 分野で 5 分以内に学習完了（Surface Laptop クラス想定）。 |
| ロギング | 主要ステップのINFOログ、評価結果サマリを標準出力 + `runs/.../log.txt` に保存。 |
| 再現性 | `seed` を CLI で指定。`runs/.../meta.json` に `seed, git_commit, command, timestamp` を書き出す。 |
| ファイル構成 | `卒業研究/main/` 配下に `code/`, `csv/`, `runs/` を集約。CLI は `cd ~/repos/research/research/卒業研究/main` した状態で実行し、成果物は `main/runs/yyyymmdd_HHMM/` に保存。 |

---

## 8. 既知のリスクと緩和策

| リスク | 影響 | 緩和策 |
| --- | --- | --- |
| **低正答率データでの推定不安定** | パラメータ推定が暴れる | θ初期分布の調整、分野数を抑制、当て推量 `c` と難易度 `b` を緩く設定（データ生成時）。 |
| **複数分野結合での確率過小** | 推薦が review 偏重に | AND ではなく幾何/ロジット平均を採用し、校正を実施。 |
| **AUC=NaN（片側クラス）** | 指標が欠落 | 片側クラス検知で警告し、その分野をスキップ。 |
| **NumPy 2.x で pyBKT 拡張が壊れる** | 実行時エラー | `numpy<2` で固定し、CI/起動時に import チェック。 |
| **LLM が候補外 ID を出力** | 運用事故 | プロンプトで「候補以外禁止」を明示し、検証スクリプトで弾く。 |

---

## 9. マイルストーン

| フェーズ | 内容 | 完了条件 |
| --- | --- | --- |
| 1. MVP | BKT fit/predict/eval、推薦 JSON 骨格、可視化 | AUC/Logloss 目標達成、runs ディレクトリ整備 |
| 2. 校正強化 | Holdout（cut_date / last_k）と Isotonic / Platt 校正 | 校正後 P(next) の Brier Score 向上 |
| 3. IRT PoC | `bkt_irt_bridge.py` で `P_for_llm` を算出、挑戦枠に IRT 難易度を活用 | LLM 入力 JSON に `P_irt` を添付 |
| 4. 運用化 | CLI 分割（fit/eval/recommend）、ログ自動保存、依存チェック | `bootstrap.sh` → `make recommend` で一連実行 |

---

## 10. 参考ドキュメント / スクリプト
- `docs/research_overview.md`, `docs/research_overview_2.md`
- `docs/setup_wsl_python310.md`（環境構築）
- `code/python_3_10/pybkt_user_domain.py`（BKT ユーティリティ）
- `code/python_3_10/bkt_irt_bridge.py`（BKT×IRT 合成）
- `code/main/` 配下の実装（`offline_fit_bkt.py`, `plot_user_history.py`, `offline_predict_scores.py`, `build_offline_llm_payloads.py`, `interactive_lpic_quiz.py`, `evaluate_scores.py`, `simulate_user_logs.py`）

---

## 11. サンプル実行フロー

> **前提**: `cd ~/repos/research/research/卒業研究/main` で作業し、相対パスはすべて `main/` 直下基準。

1. **ログ生成（任意）**
   ```bash
   python -m code.main.simulate_user_logs \
     --items-csv csv/items_sample_lpic.csv \
     --items-domain-col L2 \
     --out-csv csv/sim_logs.csv \
     --n-users 30 \
     --interactions-per-user 30 \
     --seed 42
   ```
2. **BKT パラメータ推定**
   ```bash
   python -m code.main.offline_fit_bkt \
     --csv csv/sim_logs.csv \
     --skill-col L1 L2 \
     --order-col order_id \
     --correct-col correct \
     --out-params csv/bkt_params_sim_multi.csv \
     --out-report runs/sim_bkt_metrics_multi.csv
   ```
3. **履歴可視化（任意）**
   ```bash
   python -m code.main.plot_user_history \
     --log-csv csv/sim_logs.csv \
     --params-csv csv/bkt_params_sim_multi.csv \
     --user-id u001 \
     --skill-col L2 \
     --order-col order_id \
     --correct-col correct \
     --out-dir runs/sim_history/
   ```
4. **IRT パラメータ推定（簡易2PL）**
   ```bash
   python -m code.main.fit_irt_params \
     --log-csv csv/sim_logs.csv \
     --user-col user_id \
     --item-col item_id \
     --domain-col L2 \
     --correct-col correct \
     --out-items csv/irt_items_estimated.csv \
     --out-theta csv/irt_theta_estimated.csv
   ```

5. **BKT×IRT 正答確率算出**
   ```bash
   python -m code.main.offline_predict_scores \
     --log-csv csv/sim_logs.csv \
     --params-csv csv/bkt_params_sim_multi.csv \
     --irt-items-csv csv/irt_items_estimated.csv \
     --items-csv csv/items_sample_lpic.csv \
     --items-domain-col L2 \
     --skill-col L2 \
     --mode hybrid_mean \
     --w-bkt 0.6 \
     --out runs/sim_user_item_scores.csv
   ```
6. **LLM ペイロード生成**
   ```bash
   python -m code.main.build_offline_llm_payloads \
     --scores-csv runs/sim_user_item_scores.csv \
     --out-dir runs/sim_payloads \
     --top-k 5
   ```
7. **評価（任意）**
   ```bash
   python -m code.main.evaluate_scores \
     --scores-csv runs/sim_user_item_scores.csv \
     --by overall \
     --out runs/sim_metrics_overall.csv
   ```
8. **CLI チューター（任意）**
   ```bash
   python -m code.main.interactive_lpic_quiz \
     --user-id u001 \
     --items-csv csv/items_sample_lpic.csv \
     --item-id-col item_id \
     --domain-col L2 \
     --bkt-params-csv csv/bkt_params_sim_multi.csv \
     --log-csv csv/sim_online_logs.csv \
     --max-questions 5 \
     --emit-llm-payload
   ```

本ドキュメントは、他のアシスタント/研究協力者が即座に作業へ入れるよう**要件と運用ルールを単一ソース**として管理する。変更が生じた場合は本ファイルの更新日と該当セクションを更新する。
