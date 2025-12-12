#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BKT × IRT ブリッジスクリプト（オフライン用）

- BKT の結果 (user × domain の P(next)) と
- IRT の結果 (item の a,b,c, user × domain の θ) を統合して

ユーザ × 問題ごとのスコア:
  - P_bkt (ドメインレベルの正答確率)
  - P_irt (3PL の正答確率)
  - P_for_llm (両者の重み付き平均)

を計算して CSV に出力する。

想定する入力:
  1) BKT 出力: bkt_last.csv
     user_id, domain, p_next_bkt, ... などの列を含む

  2) IRT 項目パラメータ: irt_items.csv
     item_id, domain, a, b, c

  3) IRT 能力推定: irt_theta.csv
     user_id, domain, theta

  4) 問題マスタ: items.csv
     item_id, domain, question_text, ... （LLMに渡したいメタ情報）

※列名は --col-* オプションである程度カスタマイズできる。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def three_pl(theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """3PL モデルによる正答確率を計算する"""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--bkt-last", required=True,
                   help="BKT の user×domain の出力 CSV (例: bkt_L2_last.csv)")
    p.add_argument("--irt-items", required=True,
                   help="項目パラメータ CSV (item_id, domain, a, b, c)")
    p.add_argument("--irt-theta", required=True,
                   help="能力パラメータ CSV (user_id, domain, theta)")
    p.add_argument("--items", required=True,
                   help="問題マスタ CSV (item_id, domain, question_text など)")
    p.add_argument("--out", required=True,
                   help="出力 CSV (user×itemごとのスコア)")

    # 列名カスタマイズ（最低限）
    p.add_argument("--col-bkt-user", default="user_id")
    p.add_argument("--col-bkt-domain", default="domain")
    p.add_argument("--col-bkt-pnext", default="p_next_bkt")

    p.add_argument("--col-item-id", default="item_id")
    p.add_argument("--col-item-domain", default="domain")

    p.add_argument("--col-theta-user", default="user_id")
    p.add_argument("--col-theta-domain", default="domain")
    p.add_argument("--col-theta-val", default="theta")

    p.add_argument("--col-irt-a", default="a")
    p.add_argument("--col-irt-b", default="b")
    p.add_argument("--col-irt-c", default="c")

    # BKT と IRT をどう混ぜるかの重み
    p.add_argument("--w-bkt", type=float, default=0.5,
                   help="P_for_llm = w_bkt * P_bkt + (1-w_bkt) * P_irt の w_bkt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bkt_last_path = Path(args.bkt_last)
    irt_items_path = Path(args.irt_items)
    irt_theta_path = Path(args.irt_theta)
    items_path = Path(args.items)
    out_path = Path(args.out)

    # ===== データ読み込み =====
    bkt = pd.read_csv(bkt_last_path)
    irt_items = pd.read_csv(irt_items_path)
    irt_theta = pd.read_csv(irt_theta_path)
    items = pd.read_csv(items_path)

    # 列をわかりやすく抜き出し＆リネーム（内部では統一名を使う）
    bkt_small = bkt[[args.col_bkt_user,
                     args.col_bkt_domain,
                     args.col_bkt_pnext]].copy()
    bkt_small.columns = ["user_id", "domain", "P_bkt"]

    irt_items_small = irt_items[[args.col-item-id if False else args.col_item_id,  # noqa
                                 args.col-item-domain if False else args.col_item_domain,  # noqa
                                 args.col_irt_a,
                                 args.col_irt_b,
                                 args.col_irt_c]].copy()
    irt_items_small.columns = ["item_id", "domain", "a", "b", "c"]

    theta_small = irt_theta[[args.col_theta_user,
                             args.col_theta_domain,
                             args.col_theta_val]].copy()
    theta_small.columns = ["user_id", "domain", "theta"]

    items_small = items[[args.col_item_id, args.col_item_domain]].copy()
    items_small.columns = ["item_id", "domain"]

    # ===== ユーザ×問題の組を作る =====
    # 作り方はいろいろあるが、ここでは「全ユーザ × 全問題」の直積を作る簡単版。
    # 実際には「同じ domain の問題だけを見る」などで絞り込んだ方が良い。
    users = bkt_small["user_id"].drop_duplicates()
    # 全ユーザ×全問題
    user_df = pd.DataFrame({"user_id": users})
    user_item = user_df.merge(items_small, how="cross")  # pandas 1.2+ の cross join

    # ===== BKT の P_bkt を user×item に付与（domain で引く） =====
    user_item = user_item.merge(bkt_small, on=["user_id", "domain"], how="left")

    # ===== IRT の θ と a,b,c を user×item に付与 =====
    user_item = user_item.merge(theta_small, on=["user_id", "domain"], how="left")
    user_item = user_item.merge(irt_items_small, on=["item_id", "domain"], how="left",
                                suffixes=("", "_item"))

    # ===== 3PL で P_irt を計算 =====
    mask = user_item[["theta", "a", "b", "c"]].notna().all(axis=1)
    user_item["P_irt"] = np.nan
    user_item.loc[mask, "P_irt"] = three_pl(
        theta=user_item.loc[mask, "theta"].to_numpy(),
        a=user_item.loc[mask, "a"].to_numpy(),
        b=user_item.loc[mask, "b"].to_numpy(),
        c=user_item.loc[mask, "c"].to_numpy(),
    )

    # ===== BKT × IRT を重み付きで合成 =====
    w = float(args.w_bkt)
    user_item["P_for_llm"] = np.nan

    # BKT と IRT 両方ある行だけ合成
    mask_both = user_item[["P_bkt", "P_irt"]].notna().all(axis=1)
    user_item.loc[mask_both, "P_for_llm"] = (
        w * user_item.loc[mask_both, "P_bkt"]
        + (1.0 - w) * user_item.loc[mask_both, "P_irt"]
    )

    # どちらか欠けている場合のフォールバック（お好みで調整）
    # - BKT だけある場合は BKT をそのまま使う
    mask_only_bkt = user_item["P_for_llm"].isna() & user_item["P_bkt"].notna()
    user_item.loc[mask_only_bkt, "P_for_llm"] = user_item.loc[mask_only_bkt, "P_bkt"]

    # - IRT だけある場合は IRT をそのまま
    mask_only_irt = user_item["P_for_llm"].isna() & user_item["P_irt"].notna()
    user_item.loc[mask_only_irt, "P_for_llm"] = user_item.loc[mask_only_irt, "P_irt"]

    # ===== 出力に含める列を絞る =====
    out_cols = [
        "user_id",
        "item_id",
        "domain",
        "P_bkt",
        "P_irt",
        "P_for_llm",
    ]
    # 元の items のメタ情報を付けたい場合はここで merge してもよい
    user_item[out_cols].to_csv(out_path, index=False)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
