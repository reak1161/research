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
