# 研究概要：BKT × IRT による学習者モデルと問題推薦（更新版）

> **更新日**: 2025-11-19 (JST)
>
> **この文書について**: 本プロジェクトの現状と、今後の実装方針・評価方法・運用ルールをまとめたものです。**他のアシスタント/共同研究者が読んでもすぐ作業に入れる**よう、データ仕様・CLIの使い方・方針決定事項を明記しています。

---

## 0. 現在地サマリ（What we have now）

* **概念整理**: *skill* 表現は **分野（domain）** に統一。**1問=1分野** を前提に進める。
* **中核モデル**: **pyBKT==1.4.1**（単一分野BKT）で推定・予測。忘却(Forgets)は**比較用のオプション**として実装。
* **評価**: **AUC** と **Logloss** に対応。粒度は `overall / domain / user / user×domain`。忘却 **オン/オフ比較**の差分表（ΔAUC, ΔLogloss）を出力可能。
* **可視化**: ユーザーの **複数分野サブプロット**（P(L) prior/posterior、P(correct)、正誤）を出力。
* **環境**: Windows 11 / **WSL2 Ubuntu 24.04**、Python **3.10.13（pyenv）**、プロジェクト venv は `~/repos/research/.venv`。依存固定：`numpy<2`, `pybind11>=2.12`, `pyBKT==1.4.1`。
* **セットアップ手順**: `docs/setup_wsl_python310.md` に手順化済み（VS Code の venv 自動アクティブ、`code .` の徹底、つまずき対策込み）。

---

## 1. 目的とスコープ（MVP）

* **目的**: 学習ログから **次時点の正答確率 P(next)** を推定し、**分野別の学習度**を踏まえて **次に解くべき問題**を推薦。短い**日本語フィードバック**を添付。
* **スコープ**: まずは **単一分野（=単一スキル相当）**のみ。将来 IRT による難易度整合を段階的に導入。

---

## 2. データ仕様（long形式）

* 必須列: `order_id, user_id, domain(=skill_col), correct(0/1)`
* 任意列: `item_id, timestamp, tags, difficulty_hint ...`
* 備考:

  * `order_id` が日時の場合、内部で **ユーザー内連番**に変換（学習順序担保）。
  * 本文書では **"分野 (domain)"** を用語として採用。スクリプト引数は `--skill-col` に **domain列**を指定。

---

## 3. モデリング（pyBKT）

* 推定パラメータ: `L0, T(learn), S(slip), G(guess)`（**forgets** は比較用）。
* 予測の語彙:

  * **P(L)
    _prior** = `state_predictions`（観測直前）
  * **P(L)
    _posterior** = 観測後の事後（S/G と観測で更新）
  * **P(L)
    _after_learn** = 事後に学習遷移 T を適用した次時点の直前
  * **P(correct_now)** = その時点の正答確率（`(1-S)·P(L)_prior + G·(1-P(L)_prior)`）
  * **P(next)** = 次時点の正答確率（`(1-S)·P(L)_after_learn + G·(1-P(L)_after_learn)`）

---

## 4. 推薦ロジック（MVP）

* **帯分け**: `P(next)` が **0.60–0.80** を優先（practice）。`>0.90` は除外、`<0.40` は review。
* **分野配分**: 分野の **学習度 = P(L)_after_learn**。重み `1 - 学習度` で低学習度分野を多めにサンプリング。
* **多様性**: 同一分野の連投は **最大2連**まで。
* **LLM 入力**: 候補問題 + 分野エビデンス（`domain, p_L_after, p_next, trend, last5 ...`）。**候補外ID禁止**のルールで最終選定＆短文フィードバック生成。

---

## 5. 評価（AUC/Logloss + 忘却比較）

* 指標: **AUC**（順位指標）、**Logloss**（校正/確信度）。
* 粒度: `overall / domain / user / user×domain`。
* 忘却比較: 同一データに対し `forgets=False/True` で 2回学習→**横持ちマージ**→ `ΔAUC, ΔLogloss` を算出。
* 注意: AUC は校正誤差に鈍感。**差が小さい場合は Logloss も併記**して判断。
* 将来拡張: **時系列ホールドアウト**（各 user×domain の末尾k で評価）、**ブートストラップCI**。

---

## 6. 可視化（複数分野サブプロット）

* 図要素: 太線= **P(L) prior**、点線= **P(correct)**、細線= **P(L) posterior**、散布= 正誤（1/0）。
* タイトル: `user=...  分野=...  n=...`。`--display-skill-label "分野"` で見出し統一。

---

## 7. CLI チートシート（現行）

* **忘却オン/オフ比較（AUC/Logloss, overall）**

  ```bash
  python pybkt_user_skill_report.py \
    --csv <path/to.csv> \
    --order-col date --user-col user_id --skill-col domain --correct-col correct \
    --compare-forgets --auc-by overall
  ```
* **分野別でCSV保存**

  ```bash
  --compare-forgets --auc-by skill --eval-out runs/auc_logloss_by_domain.csv
  ```
* **学習状態の可視化（ユーザー×複数分野）**

  ```bash
  python pybkt_user_skill_report.py \
    --csv <path/to.csv> --user 68 --skills algebra fractions geometry \
    --order-col date --user-col user_id --skill-col domain --correct-col correct \
    --display-skill-label "分野" \
    --plot-out runs/u68_domains.png
  ```

---

## 8. IRT 導入の位置づけ（次フェーズ）

* 目的: **難易度整合**と**校正改善**。
* 使い所: `p_for_llm = calibrate( BKTの P(next), 2/3PL(a,b,c) )`、挑戦枠の選定に `a` を活用。
* データ: 生成器に `a,b,c` を持たせ、**分野×問題**で難度分布を設計。

---

## 9. 環境・再現性

* OS: Windows 11 / **WSL2 Ubuntu 24.04**
* Python: **3.10.13（pyenv）**、venv: `~/repos/research/.venv`
* 依存固定: `numpy<2`, `pybind11>=2.12`, `pyBKT==1.4.1`
* セットアップガイド: `docs/setup_wsl_python310.md`（VS Code の `.vscode/settings.json` 例、`bootstrap.sh` 付属）

---

## 10. 他のアシスタント/共同研究者向けブリーフ

* **目的**: BKTで分野別学習度と P(next) を出し、LLM が *候補問題* から**最大5件**を選定して短い日本語フィードバックを返す。**候補外IDは不可**。
* **入力CSV**: `order_id,user_id,domain,correct`（order_id が日時ならユーザー内連番化）。
* **既定ルール**: `P(next) 0.60–0.80` を優先、`>0.90` 除外、`<0.40` は review。分野配分は `1-学習度`。同一分野は最大2連。
* **評価**: AUC/Logloss を overall/粒度別で出す。`--compare-forgets` で忘却オン/オフ差分を出す。
* **可視化**: ユーザー×複数分野のサブプロットを PNG で保存。
* **動かし方**: 上記「CLI チートシート」参照。VS Code は **WSL から `code .`**。venv は `~/repos/research/.venv`。
* **注意**: NumPy は **2.x を禁止**（`numpy<2`）。`--correct-col` の列名ミスに注意。

---

## 11. マイルストーン（見直し版）

1. **MVP**: 単一分野BKTの fit/predict/eval、忘却比較、可視化、推薦JSON骨格。
2. **校正強化**: Holdout + Isotonic/Platt で `p_for_llm` を校正。
3. **IRT PoC**: 2/3PL の難易度・識別力で帯分け微調整。
4. **運用**: CLI 分割（fit/eval/recommend）、ログ自動保存、再現性メタ（seed/依存/commit）を `runs/` に出力。

---

*補足*: 以前の用語「スキル」は「分野」に統一。スクリプト引数名は `--skill-col` を流用するが、渡す列は **domain** を推奨。可視化や出力の見出しは `--display-skill-label "分野"` で揃えられる。
