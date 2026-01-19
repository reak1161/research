# BKT×IRT 学習支援システム 要件定義書

この文書は、Python で以下の 5 つの機能を持つモジュール／スクリプト群を実装するための要件定義です。

1. オフラインで BKT パラメータを推定するコード
2. 指定したユーザーの学習状態遷移をグラフ・表で可視化するコード
3. オフラインでユーザー×問題の正答確率を推定するコード（BKT＋IRT 前提）
4. LLM にパラメータ等を渡して問題候補を受け取るためのコード（LLM ブリッジ）
5. オンラインでリアルタイムに正答確率を更新しつつ出題するコード（CLI チューター）

---

## 0. 共通仕様

### 使用環境
- 言語: Python 3.10
- 実行環境: WSL2 上の Ubuntu（ターミナルから実行）
- 仮想環境: `.venv` を想定

### 主なライブラリ
- pandas
- numpy
- matplotlib
- 必要に応じて argparse, typing

### ディレクトリ構成（前提）
- ルート: `~/repos/research/research/卒業研究/`
  - `code/python_3_10/` … Python スクリプト
  - `csv/` … 入出力 CSV
  - `runs/` … 生成画像・JSON・ログなど

以降のパスはこの構成を想定して相対パスで記述する。

---

## 1. オフライン BKT パラメータ推定ツール

### ファイル名
`code/python_3_10/offline_fit_bkt.py`

### 概要
学習ログ CSV（縦持ち）を入力として、指定したスキル列（例: domain, L1, L2, L3）ごとに単一 BKT モデルをフィットする。フィット結果として、各スキルのパラメータ（L0, T, S, G）と、必要に応じて評価指標（AUC, Logloss など）を CSV で出力する。

### 入力
学習ログ CSV ファイル（例: `csv/ct_lpic.csv`）

ログの必須列:
- `user_id` : 学習者 ID
- `item_id` : 問題 ID
- `timestamp` : 解答時刻（UNIX 時刻 or ISO 文字列）
- `skill` : スキルラベル（この列名は引数で指定、domain / L2 など）
- `correct` : 正誤フラグ（1=正解, 0=不正解）

### コマンドライン引数（例）
- `--csv` : 入力ログ CSV パス（必須）
- `--order-col` : 時系列ソートに使う列名（デフォルト: timestamp）
- `--user-col` : ユーザー ID 列名（デフォルト: user_id）
- `--skill-col` : スキル列名（例: domain, L2）
- `--correct-col` : 正誤列名（デフォルト: correct）
- `--out-params` : パラメータ出力 CSV パス（例: `csv/bkt_L2_params.csv`）
- `--out-report` : スキルごとの評価指標などの CSV パス（任意）

### 出力: パラメータ CSV 形式
列例:
- `skill` : スキル名（`--skill-col` で指定した値）
- `L0` : 初期習得確率
- `T` : 学習遷移確率
- `S` : スリップ率
- `G` : 当て推量
- 任意で AUC, Logloss など

---

## 2. 学習状態遷移の可視化ツール

### ファイル名
`code/python_3_10/plot_user_history.py`

### 概要
指定したユーザー・スキルについて、解答履歴に沿った BKT の習得度 P(L) と正答確率 P(next) の推移を

- グラフ（PNG）
- 表（CSV）

で出力する。

### 入力
- 学習ログ CSV（例: `csv/ct_lpic.csv`）
- BKT パラメータ CSV（前項 1 の出力）
  - 指定スキル列に対する L0, T, S, G

### コマンドライン引数（例）
- `--log-csv` : 学習ログ CSV
- `--params-csv` : BKT パラメータ CSV
- `--user-id` : 対象ユーザー ID（単一、複数指定も対応できると望ましい）
- `--skill` : 対象スキル名（単一 or 複数）
- `--skill-col` : スキル列名（domain / L2 など）
- `--order-col` : 時系列列（timestamp）
- `--out-dir` : 出力ディレクトリ（例: `runs/history_plots/`）

### 処理要件
- ログを user_id & skill でフィルタし、order-col でソートする。
- BKT 更新式を実装し、各ステップで
  - P(L)
  - P(next correct)
  を計算。

### グラフ
- x軸: 回答順（または timestamp）
- y軸: P(L) と P(next)（2本の線）
- 正誤は点や色でプロット（例: 正解=○, 不正解=×）

### 表
`step_index, timestamp, item_id, correct, P_L, P_next` を CSV として出力。

---

## 3. オフライン正答確率推定ツール（BKT＋IRT）

### ファイル名
`code/python_3_10/offline_predict_scores.py`

### 概要
BKT パラメータ・学習ログ・IRT パラメータを入力として、指定したスナップショット時点でのユーザー×問題ごとの正答確率を計算する。正答確率は BKT と IRT を組み合わせた \(P_{\text{final}}(u,i)\) とする。

### 想定入力
- 学習ログ CSV: `csv/ct_lpic.csv`
- BKT パラメータ CSV: `csv/bkt_L2_params.csv`
- IRT 項目パラメータ CSV: `csv/irt_items.csv`
  - 列: `item_id, domain, a, b, c`
- （任意）IRT 能力推定 CSV: `csv/irt_theta.csv`
  - 列: `user_id, domain, theta`
  - 無い場合は「BKT から擬似 θ を作る」モードも許可
- 問題マスタ CSV: `csv/items_lpic.csv`
  - 列: `item_id, domain, L1, L2, L3, question_text, ...`

### 正答確率の定義（イメージ）
- BKT による分野別正答確率（「この分野の次の問題」の正答確率）  
  \(P_{\text{BKT}}(u,d) = P(X_{t+1}=1 \mid H_{u,d})\)
- IRT（3PL）による問題別正答確率  
  \(P_{\text{IRT}}(u,i) = c_i + (1-c_i)/(1+\exp(-a_i(\theta_{u,d(i)}-b_i)))\)
- 最終的な正答確率（重み付き平均版）  
  \(P_{\text{final}}(u,i) = w \cdot P_{\text{BKT}}(u,d(i)) + (1-w) \cdot P_{\text{IRT}}(u,i)\)  
  \(w\) はコマンドライン引数で指定（デフォルト 0.5）
- もしくはシンプル版として、「BKT から logit 変換で擬似 θ を作る」モード
  - `--mode hybrid_mean` : 上の重み付き平均
  - `--mode bkt_to_theta` : BKT → θ → 3PL で \(P_{\text{final}} = P_{\text{irt}}\)

### コマンドライン引数（例）
- `--log-csv`
- `--bkt-params-csv`
- `--irt-items-csv`
- `--irt-theta-csv`（省略可）
- `--items-csv`
- `--skill-col` : BKT で使う分野列（例: L2）
- `--cut-timestamp` : この時刻以前のログのみで状態を作る（評価時点）
- `--mode` : `hybrid_mean` or `bkt_to_theta`
- `--w-bkt` : 重み w（`hybrid_mean` のとき使用）
- `--out` : 出力 CSV（例: `runs/user_item_scores.csv`）

### 出力 CSV 形式
列例:
- `user_id`
- `item_id`
- `domain`（skill）
- `P_bkt`
- `P_irt`
- `P_final`（LLMに渡すベーススコア）
- 必要に応じて `theta, a, b, c` など

---

## 4. LLM ブリッジ（ペイロード生成）ツール

### ファイル名
- コア: `code/python_3_10/llm_payload.py`
- バッチ用スクリプト: `code/python_3_10/build_offline_llm_payloads.py`

### 概要
BKT＋IRT の結果やユーザー状態・候補問題リストから、LLM に渡す JSON ペイロードを組み立てる共通モジュール。オフライン版（全ユーザー分まとめて JSON を作る）とオンライン版（1ユーザー分だけ即座に作る）の両方で使えるようにする。

### `llm_payload.py` の要件
```python
def build_llm_payload(
    user_id: str | int,
    mode: str,
    user_state: dict,
    candidates: list[dict],
    k: int = 5,
) -> dict:
    """
    LLM に渡すペイロードを共通フォーマットで作る。

    mode: "offline" or "online"
    user_state:
      例: {"L2:text_processing": 0.72, "L2:stdio_and_pipes": 0.55, ...}

    candidates:
      各問題の情報のリスト。
      例:
        {
          "item_id": 101,
          "domain": "text_processing",
          "p_final": 0.68,
          "difficulty": "medium",
          "tags": ["grep_basic"],
          "question_text": "...",
        }

    上位 k 件の候補だけを含める。
    """
```

### 出力 JSON 例
```json
{
  "schema_version": "1.0",
  "mode": "offline",
  "user_id": "u001",
  "user_state": {
    "L2:text_processing": 0.72,
    "L2:stdio_and_pipes": 0.55
  },
  "candidates": [
    {
      "item_id": 1,
      "domain": "text_processing",
      "p_final": 0.68,
      "difficulty": "easy",
      "tags": ["sort_basic"],
      "question_text": "..."
    }
  ]
}
```

### オフライン用スクリプト
`build_offline_llm_payloads.py`

`offline_predict_scores.py` の出力（ユーザ×問題スコア）を読み、ユーザーごとに候補問題リストを構築して `build_llm_payload()` を呼び、`runs/offline_payload_{user_id}.json` のように保存する。

---

## 5. オンライン出題エンジン（CLI チューター）

### ファイル名
LPIC 版例: `code/python_3_10/interactive_lpic_quiz.py`

### 概要
ターミナル上で対話的に問題を出題し、

1. 学習者の回答を受け取る
2. 正誤判定してログ CSV に記録する
3. BKT（推定済みパラメータ）を用いてリアルタイムで P(L), P(next) を更新
4. IRT パラメータを用いて問題ごとの正答確率を計算
5. 必要なら `llm_payload.build_llm_payload()` を用いて LLM に「次の問題候補」を問い合わせる

初期バージョンでは LLM 呼び出しはダミーでもよい（ペイロードを print するだけ）。

### 入力
- 問題マスタ CSV（例: `csv/items_lpic.csv`）
  - 列: `item_id, L1, L2, L3, question_text, answer_type, choices, correct_key, ...`
- BKT パラメータ CSV
- IRT 項目パラメータ CSV（あれば）
- ユーザー ID（コマンドライン引数）

### コマンドライン引数
- `--user-id` : 対象ユーザー ID
- `--items-csv` : 問題 CSV
- `--bkt-params-csv` : BKT パラメータ CSV
- `--irt-items-csv` : IRT 項目パラメータ CSV（省略可）
- `--log-csv` : 解答ログ出力 CSV（例: `csv/ct_lpic.csv`）

### 機能要件
**出題:**
- `answer_type` が `mcq` の場合:
  - 選択肢を A/B/C/D または 1/2/3/4 で受け付け、正誤判定
- `answer_type` が `text` の場合:
  - テキスト入力を受け付け、指定された `correct_key` に基づいて正誤判定

**BKT:**
- 起動時に BKT パラメータを読み込み、ユーザー×スキルの初期 P(L) を L0 で初期化
- 各解答ごとに BKT 更新式で P(L) を更新し、P(next) を計算

**IRT:**
- IRT 項目パラメータがあれば、BKTから能力 θ を計算し、3PL で問題ごとの正答確率を計算（P_irt または P_final）

**ログ:**
- `timestamp, user_id, item_id, L1, L2, L3, correct` などを `--log-csv` に追記

**LLM 連携（簡易版）:**
- 各ステップで「次の問題候補」を選ぶ直前に、
  - 現在の user_state（P(L) or P_final）
  - 候補問題（未出題 or 最近解いていない問題など）
  から `build_llm_payload()` を呼び出し、ペイロードを JSON で標準出力に出す（実際の API 呼び出しは別モジュールでも可）。
- オンラインモードでは、直近履歴から `mood_state` を判定し、`P_final` を再スコアした Top-K だけを Gemini に送るオプションを用意する（CLI 例: `--use-llm --llm-model gemini-2.5-flash --llm-min-interval 1.0`）。LLM 応答が不正/失敗のときは Python 側のトップ候補にフォールバック。
- P_final の優先供給は「オフライン計算結果（offline_predict_scores の CSV）を読み込んで item ごとに反映」を基本とし、未スコアの item は BKT の P(L) でフォールバックする。将来的にクイズ内で階層BKT＋IRT を逐次計算するリアルタイム版を検討する場合は、同等ロジックをオンライン側に組み込むこと。
- **P_final の供給方針**: 現行はオフライン計算 (`offline_predict_scores.py`) の出力 CSV を読み込んで item ごとの P_final を候補に使う運用を推奨。将来的にリアルタイム計算（クイズ内で階層BKT＋IRTを逐次評価）を検討する場合は、同じロジックをオンライン側に持ち込む。

---

## 6. もう少し良くするための構成案（オプション）

### (A) 共通ライブラリ層を作る
- `bkt_core.py`:
  - BKT の更新式・パラメータ読み書き・ユーザー状態管理を関数化
- `irt_core.py`:
  - 3PL の実装、θ の変換、数値計算を関数化
- `data_io.py`:
  - ログ CSV, items CSV, params CSV の読み書きユーティリティ

→ これらを 1 回ちゃんと作っておくと、
`offline_fit_bkt.py` / `offline_predict_scores.py` / `interactive_lpic_quiz.py`
などから同じ関数を使い回せるので、バグが減る。

### (B) 「評価スクリプト」を別に切る
3 で出した `user_item_scores.csv` を使って

- AUC, Logloss を計算する `evaluate_scores.py`
- BKT 単独 vs BKT+IRT vs BKT→θ のモードを比較するスクリプトとして独立

→ 卒研の評価パートが書きやすい。

---

## 7. 階層型 BKT（L1/L2/L3）の重み付き正答確率（追加仕様）

### 7.1 目的
既存の BKT×IRT 正答確率に対し、L1（大分類）・L2（中分類）・L3（小分類）の BKT を重み付きで統合した正答確率を導入する。データ量が多い階層と細かい概念階層をバランスよく活用することで、推薦精度を向上させる。

### 7.2 前提
- 各問題には `L1, L2, L3` の3階層ラベルが付与されている。
- 階層ごとの BKT パラメータ CSV（`bkt_L1_params.csv`, `bkt_L2_params.csv`, `bkt_L3_params.csv`）が用意されている。
- 学習ログ形式: `user_id, item_id, timestamp, L1, L2, L3, correct`。

### 7.3 階層別 BKT 正答確率
ユーザ u、問題 i に対し以下を算出（既存 BKT 更新式を流用）:
- `P_bkt_L1(u, L1(i))`
- `P_bkt_L2(u, L2(i))`
- `P_bkt_L3(u, L3(i))`

併せて、各階層における回答数 `N_L1/N_L2/N_L3` を保持する。

### 7.4 重み設計
1. **ベース重み（ハイパーパラメータ）**:  
   - `w1_base = 0.2`, `w2_base = 0.5`, `w3_base = 0.3`
2. **信頼度補正関数**:
   ```python
   def level_reliability(n_answers: int, n_min: int = 5) -> float:
       return min(1.0, n_answers / n_min)
   ```
3. **実重み**:
   - `r1 = w1_base * level_reliability(N_L1)`
   - `r2 = w2_base * level_reliability(N_L2)`
   - `r3 = w3_base * level_reliability(N_L3)`
   - `Z = r1 + r2 + r3`
   - `Z > 0` の場合は `w1=r1/Z`, `w2=r2/Z`, `w3=r3/Z`  
   - `Z == 0` の場合はベース比率をそのまま使用するか、BKT を未定義扱いにする。

### 7.5 統合 BKT 正答確率
```
P_bkt_total(u, i) = w1 * P_bkt_L1(u, L1(i))
                  + w2 * P_bkt_L2(u, L2(i))
                  + w3 * P_bkt_L3(u, L3(i))
```
階層の値が取れない場合はその階層の `r_k` を 0 にする、またはフォールバック値（例: 0.5）を使う。

### 7.6 IRT との統合
既存の `P_final = w_bkt * P_bkt + (1-w_bkt) * P_irt` の BKT 部分を `P_bkt_total` に置き換える。

### 7.7 実装メモ
- 主に `offline_predict_scores.py` で階層別 BKT パラメータを読み、ユーザ×問題ごとに `P_bkt_total` を計算する。
- 補助関数例:
  - `level_reliability(n_answers, n_min=5)`
  - `compute_hierarchical_bkt_probability(user_id, item_info, ...)`
- CLI 例（参考）:
  ```bash
  python -m code.offline_predict_scores \
    --log-csv csv/sim_logs.csv \
    --params-l1 csv/bkt_L1_params.csv \
    --params-l2 csv/bkt_L2_params.csv \
    --params-l3 csv/bkt_L3_params.csv \
    --hier-bkt-base 0.2 0.5 0.3 \
    --hier-bkt-n-min 5 \
    --irt-items-csv csv/irt_items_estimated.csv \
    --items-csv csv/items_sample_lpic.csv \
    --mode hybrid_mean --w-bkt 0.6 \
    --out runs/sim_user_item_scores.csv
  ```

---

## 8. モチベーションを考慮した出題難易度制御（追加仕様）

### 8.1 目的
固定の「正答確率 60%」ではなく、学習者の直近成績に応じて難易度レンジを動的に調整する。`P_final(u,i)` を用い、FRUSTRATED なら解けそうな問題を多めに、CONFIDENT ならチャレンジ寄り、NORMAL は 6〜7 割解けそうな問題を中心に出題する。

### 8.2 learner mood（学習者状態）の判定
- 直近 N 問（デフォルト N=5）の正答率、連続不正解数を用いて `mood_state` を決定。
- 初期案:
  - FRUSTRATED: 直近正答率 < 0.3 または 連続不正解 >= 3
  - CONFIDENT: 直近正答率 > 0.8
  - NORMAL: 上記以外
- 閾値やウィンドウはハイパーパラメータとして調整可。
- 擬似コード例:
  ```python
  def get_mood_state(history, window: int = 5) -> str:
      # history は [(correct: bool, p_final: float|None), ...] の時系列を想定
      ...
  ```

### 8.3 mood_state ごとの狙う正答確率レンジ（初期案）
- FRUSTRATED: target≈0.75, filter 0.60–0.90（解けそうな問題を多めに）
- NORMAL: target≈0.70, filter 0.60–0.85（適正難易度 6–7 割）
- CONFIDENT: target≈0.65, filter 0.50–0.85（少しチャレンジ寄り）
- 数値は実験に応じて調整可能。

### 8.4 Python 側での候補選択ロジック
- 入力: ユーザ u、候補問題リスト（item_id, P_final, L1/L2/L3, role など）、mood_state。
- 出力: mood_state のレンジ内で target に近い順にソートした上位 K 問（例: K=10）。
- 擬似コード:
  ```python
  def select_candidates_by_mood(candidates, mood: str, k: int = 10):
      if mood == "FRUSTRATED":
          target, low, high = 0.75, 0.60, 0.90
      elif mood == "CONFIDENT":
          target, low, high = 0.65, 0.50, 0.85
      else:
          target, low, high = 0.70, 0.60, 0.85
      filtered = [c for c in candidates if low <= c["p_final"] <= high]
      scored = sorted(filtered, key=lambda c: abs(c["p_final"] - target))
      return scored[:k]
  ```

### 8.5 LLM ペイロードへの組み込み
- LLM は難易度決定エンジンではなく、Python 側で選んだ候補 Top K を元に「どれを出すか＋フィードバック生成」を行う。
- `llm_payload.py` に `mood_state` と `policy_hint`（例: target_p_final, コメント）を追加して LLM に渡す。
- LLM への指示例:
  - FRUSTRATED: p_final が高め、role が review/practice を優先し、励ましのフィードバックを付ける。
  - CONFIDENT: p_final 0.55–0.7 の challenge を含めてもよい。「次のステップ」を示す。

### 8.6 まとめ
- 固定で「正答率 60%」に縛らず、直近成績から mood_state を判定。
- mood_state ごとに P_final の目標レンジを設定し、Python 側で候補をフィルタ・ソート。
- mood_state と policy_hint を LLM に渡し、難易度方針を明示したうえでフィードバック生成をさせる。

---

## 9. LLM API 利用要件（Gemini）
- **API とライブラリ**: Google Gemini（`google-genai` 公式クライアント）。モデル既定は `gemini-2.5-flash`（利用可能モデルに応じて切替可）。
- **認証**: 環境変数 `GEMINI_API_KEY` を使用（例: `export GEMINI_API_KEY="..."`）。キーはリポジトリやログに残さない。
- **タイムアウト / リトライ**: 1 リクエストあたり 30s 以内、HTTP 429/5xx は指数バックオフで最大 3 回リトライ。致命的エラー時は Python 側でフォールバック（候補をそのまま提示）。
- **レート制御**: 1 秒あたり 1 リクエスト程度にスロットルし、連続失敗時はクールダウンを入れる。
- **ログ**: プロンプトとレスポンス本文は必要最低限のみ保存し、API キー・個人情報を含めない。`runs/` 以下にメタ（モデル名、所要時間、HTTP ステータス）を残す。
- **オンラインモード方針**: 各出題ごとに BKT/IRT と直近履歴で候補を再スコアし、Top K（mood 反映済み）だけを LLM に渡す。LLM は候補外 ID を生成しない前提のプロンプトを付与する。
