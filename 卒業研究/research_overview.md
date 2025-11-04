# 研究概要：BKT × IRT による学習者モデルと問題推薦

## 背景
- 個別最適化学習では、学習者の**現在の習得状態**と**次問題の正答確率**を推定し、適切な難易度・内容を**継続的に推薦**する仕組みが重要。
- 既存の **Bayesian Knowledge Tracing (BKT)** は手続き的知識の獲得を2状態（未習得/習得）で捉えるのに対し、**Item Response Theory (IRT)** は**連続能力**と**項目特性**（識別力 *a*、難易度 *b*、当て推量 *c*）に基づく確率モデルを提供。
- 本研究は **BKT と IRT を統合/併用**し、**将来の正答確率予測**と**次に解くべき問題推薦**を行う実運用可能なパイプラインの構築を目的とする。

## 目的
1. 学習ログから **次時点の正答確率** \(P(\text{next correct})\) を精度良く推定  
2. 推定結果に基づいて **適切な次問題**（review / practice / challenge）を推薦  
3. 将来的に IRT パラメータを取り込み、**難易度整合的**な推薦を実現

## データ
- 形式（長表）: `order_id, user_id, item_id, skill_name, correct(0/1), [timestamp]`
- 単一/複数スキルを含むデータに対応（Qマトリクスは拡張で導入）
- 人工データ生成器を実装（正答率分布を制御：θ分布、Kスキル結合、当て推量 *c*、難易度 *b*、識別力 *a*）

## モデル構成
### BKT（単一スキルを中心にMVP実装）
- パラメータ: 初期習得度 \(L_0\)、学習確率 \(T\)、スリップ \(S\)、当て推量 \(G\)  
- 学習: EM（必要に応じ **1/m** または **1/√m** の重み付け）  
- 出力: 各観測点での **事後習得確率** \(P(L_t \mid \text{履歴})\) と **正答確率** \(P(\text{correct}_t)\)

### 複数スキルの結合（拡張）
- AND（積）／OR（補集合の積）／**幾何平均**／**ロジット平均**を選択制  
- 初期デフォルトは **幾何平均**（極端な低下/上振れを抑制）

### IRT（統合予定）
- 2PL/3PL: \(P_i(\theta)= c_i + (1-c_i)\,\sigma\!\left(a_i(\theta - b_i)\right)\)  
- IRT値を **難易度補正**や推薦帯分けに利用（BKTの \(P(\text{next})\) に調整項として加味）

### 連続的習得度（オプション）
- 連続潜在能力 \(\theta_t\) を時系列更新（Dynamic IRT / PFA/Elo 近似）  
- あるいは **Beta-Bernoulli** で「得意度」 \(p_t=\alpha/(\alpha+\beta)\) を逐次更新（軽量）

## 推薦ロジック
- 目標帯域：**expected_p_correct が 0.6–0.8 を優先**
  - \(>0.9\): 除外（簡単すぎ）  
  - \(<0.4\): **review**（復習）  
- 種別: `review / practice / challenge`  
- 出力（例；最大5件・短い助言つき）:
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
      "skill_ids": ["add"],
      "feedback_to_student": "桁上がりの位置に注意して、途中式を一行増やそう。",
      "note_to_teacher": "P=0.72。直近で繰上りミス。"
    }
  ]
}
```

## 評価
- 指標: **AUC**, **Logloss**（ホールドアウト: `cut_date` / `last_k` / `ratio`）  
- 片側クラス（全0/全1）検知時は **AUC=NaN を警告＋スキップ**  
- **Definition of Done（MVP）**:  
  - 単一スキル平均で **AUC ≥ 0.65**, **Logloss ≤ 0.62**  
  - 同一 seed で再現（±0.001）

## 可視化
- **習得状態の変位**: 複数ユーザ×複数スキルのグリッドに  
  - 太線: \(P(L_t)\)（BKTの state_predictions）  
  - 破線: \(P(\text{correct}_t)\)  
  - 散布: 実観測の正誤（1/0）  
- 履歴の推移、ROC、校正曲線（calibration）を併記

## 実装・環境
- OS: Windows 11 / **WSL2 Ubuntu 24.04**  
- Python: **3.10.13**（pyenv）, 仮想環境: `~/repos/research/.venv`  
- 主要ライブラリ:  
  - `pyBKT==1.4.1`（**C++拡張 .so 有効化**）  
  - `numpy<2`, `pybind11>=2.12`（ABI安定のため固定）  
- 構成（例）:
```
project/
├─ data/               # CSV, candidates, qmatrix
├─ policies/           # 推薦ルールYAML
├─ src/
│  ├─ bkt/             # pyBKTラッパ（fit/predict）
│  ├─ recommend/       # ランキング・帯分け・JSON出力
│  ├─ eval/            # AUC/Logloss, ROC
│  └─ viz/             # 可視化スクリプト
├─ runs/yyyymmdd_HHMM/ # 出力（モデル, 図, JSON, ログ）
└─ requirements.txt
```

## 既知の課題と対策
- **低正答率データでの推定不安定**  
  - θ初期平均↑、スキル数K抑制、当て推量 *c*↑、難易度 *b* 易化、識別力 *a*↓  
  - 目標：全体正答率 **0.55–0.70** 帯を確保してからデータ量を増やす  
- **複数スキル結合での確率過小**  
  - AND（積）から **幾何/ロジット平均**へデフォルト切替、校正で補正  
- **NumPy 2.x によるネイティブ壊れ**  
  - `numpy<2` で固定、CIで import テスト  
- **AUC=NaN**  
  - 評価件数下限と片側クラス検知・警告

## マイルストーン
1. **MVP**：単一スキルBKTの fit/predict/eval、推薦JSON出力、可視化  
2. **複数スキル**：結合ルール比較（AND/幾何/ロジット）と校正  
3. **IRT統合**：難易度補正、連続能力トラッキング（Dynamic IRT 近似）  
4. **運用**：スクリプトのCLI化、ログ保存、再現性確保

## 期待される貢献
- BKT と IRT を併用した**軽量・再現性の高い推薦パイプライン**  
- 学習現場で使える**シンプルな帯分け規則**と**短い助言文テンプレ**  
- 研究再現性のための**環境固定・手順化**、**可視化**の提供
