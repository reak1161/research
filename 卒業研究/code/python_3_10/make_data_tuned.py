#!/usr/bin/env python
# coding: utf-8
"""
make_data_test.py を最小改変で拡張:
- 既存の根幹(3PL×スキル積、能力の成長/減衰、逐次CSV追記)は維持
- 追加のノブだけ用意（既定は元コードと同等の挙動）
    * 初期θの平均/分散 (--theta-mean/--theta-std)
    * a,b,cの分布パラメータ (--a-mu/--a-sigma/--b-mean/--b-std/--c-mean/--c-kappa)
    * 1〜5スキル数の重み (--k-dist "w1,w2,w3,w4,w5")  ※未指定なら従来通り一様1〜5
    * 目標正答率に合わせた θ平均の微調整 (--tune-theta --target-acc 0.60)
    * BKT用の補助CSV出力 (--out-bkt) と多スキル取扱 (--bkt-skill first|random|explode)
- 既定値は元コード相当（θ平均=-10 など）なので、明示的に引数を指定しない限り元の振る舞いを維持します。
"""
import os
import time
import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# ===== 既定（元コード準拠） =====
np.random.seed(0)
USER     = 200
SKILL    = 50
ITEM     = 500
RECORD   = 100000
INCREASE = 0.02
DECREASE = 0.002
SAVE_FILE_NAME = "../../csv/tuned_200_50_500_100000.csv"

# ===== 元コード関数（最小改変） =====

def rsa(theta: float, a: float, b: float, c: float) -> float:
    """3PL-IRT (1.7ロジット係数は元コードのまま)"""
    return c + (1 - c) / (1 + np.exp(-1.7 * a * (theta - b)))


def decline_trait(theta_val: float, elapsed_sec: float) -> float:
    """能力衰退（元コード準拠）"""
    return theta_val - np.random.beta(10, 10) * DECREASE * (elapsed_sec / 3600.0)


def init_trait(user_num: int, skill_num: int, theta_mean: float, theta_std: float) -> np.ndarray:
    """USER×SKILL の初期能力（既定は N(-10, 2)）"""
    return np.random.normal(theta_mean, theta_std, (user_num, skill_num))


def calc_result(traits_row: np.ndarray, items_rows: pd.DataFrame, skills: np.ndarray) -> int:
    """多スキル積: 各スキルの正答確率を乗算（元コード準拠）"""
    prob = 1.0
    for s in skills:
        row = items_rows.loc[items_rows['skill'] == s].iloc[0]
        prob *= rsa(traits_row[s], float(row['a']), float(row['b']), float(row['c']))
        if prob <= 0.0:
            return 0
    return 1 if prob > np.random.random() else 0

# ===== 拡張: パラメータ化した項目生成 =====

def _parse_kdist(s: Optional[str]) -> Optional[np.ndarray]:
    if not s:
        return None
    w = np.array([float(x) for x in s.split(',')], dtype=float)
    if w.size != 5:
        raise ValueError("--k-dist は 'w1,w2,w3,w4,w5' の5要素で指定してください（1〜5スキルの重み）")
    if (w < 0).any() or w.sum() == 0:
        raise ValueError("--k-dist の重みは非負で総和>0である必要があります")
    return w / w.sum()


def make_items(skill_num: int, item_num: int,
               a_mu: float, a_sigma: float,
               b_mean: float, b_std: float,
               c_mean: float, c_kappa: float,
               kdist: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    元の make_items を一般化：
    - a ~ LogNormal(a_mu, a_sigma)
    - b ~ Normal(b_mean, b_std)
    - c ~ Beta(c_kappa*c_mean+1, c_kappa*(1-c_mean)+1)
    - スキル数 K は 1〜5 のカテゴリ分布（未指定なら従来通り K~Uniform{1..5}）
    """
    rows = []
    skill_ids = np.arange(skill_num, dtype=int)

    # cのベータ分布パラメータ
    alpha = c_kappa * c_mean + 1.0
    beta  = c_kappa * (1.0 - c_mean) + 1.0

    for i in range(item_num):
        if kdist is None:
            k = np.random.randint(5) + 1  # 1〜5（元コード）
        else:
            k = np.random.choice(np.arange(1, 6), p=kdist)
        skills = np.random.permutation(skill_ids)[:k]
        for s in np.sort(skills):
            a = np.random.lognormal(a_mu, a_sigma)
            b = np.random.normal(b_mean, b_std)
            c = np.random.beta(alpha, beta)
            rows.append((i, int(s), a, b, c))
    return pd.DataFrame(rows, columns=['item_id', 'skill', 'a', 'b', 'c'])

# ===== 目標正答率に合わせて θ平均を簡易チューニング =====

def quick_estimate_acc(theta_mean: float, theta_std: float,
                       items: pd.DataFrame, skill_num: int,
                       trials: int = 5000) -> float:
    """
    簡易サンプルで全体正答率を概算（能力成長・減衰は無視して近似）。
    目標正答率に寄せるための θ平均探索に利用。
    """
    acc = 0
    for _ in range(trials):
        theta = np.random.normal(theta_mean, theta_std)
        # ランダムに item を1つ選び、その item の複数スキルを積結合
        row = items.iloc[np.random.randint(len(items))]
        item_id = int(row['item_id'])
        item_rows = items[items['item_id'] == item_id]
        p = 1.0
        for _, r in item_rows.iterrows():
            p *= rsa(theta, float(r['a']), float(r['b']), float(r['c']))
        acc += (np.random.rand() < p)
    return acc / trials

# ===== メイン =====

def main():
    ap = argparse.ArgumentParser()
    # 規模・出力
    ap.add_argument('--users', type=int, default=USER)
    ap.add_argument('--skills', type=int, default=SKILL)
    ap.add_argument('--items', type=int, default=ITEM)
    ap.add_argument('--records', type=int, default=RECORD)
    ap.add_argument('--save', default=SAVE_FILE_NAME)
    # 3PLパラメータ分布
    ap.add_argument('--a-mu', type=float, default=0.0, help='lognormal の mu（既定=0.0）')
    ap.add_argument('--a-sigma', type=float, default=0.5, help='lognormal の sigma（既定=0.5）')
    ap.add_argument('--b-mean', type=float, default=0.0)
    ap.add_argument('--b-std', type=float, default=2.0)
    ap.add_argument('--c-mean', type=float, default=0.10, help='当て推量の期待値（既定=0.10）')
    ap.add_argument('--c-kappa', type=float, default=20.0, help='Betaの濃度（既定=20 → 元コード相当）')
    ap.add_argument('--k-dist', type=str, default=None, help='例: "0.7,0.3,0,0,0" (K=1..5)')
    # θの初期分布
    ap.add_argument('--theta-mean', type=float, default=-10.0)
    ap.add_argument('--theta-std', type=float, default=2.0)
    # 目標正答率へチューニング
    ap.add_argument('--tune-theta', action='store_true')
    ap.add_argument('--target-acc', type=float, default=0.60)
    ap.add_argument('--tune-grid', type=int, default=21, help='探索点数（±1.5範囲）')
    # BKT用補助出力
    ap.add_argument('--out-bkt', type=str, default=None, help='BKT用CSV (order_id,user_id,skill_name,correct)')
    ap.add_argument('--bkt-skill', type=str, default='first', choices=['first','random','explode'])

    args = ap.parse_args()

    USERS  = int(args.users)
    SKILLS = int(args.skills)
    ITEMS  = int(args.items)
    RECORDS= int(args.records)

    # 出力先準備
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    header = ['result', 'user_id', 'item_id', 'skill', 'date']
    pd.DataFrame(columns=header).to_csv(args.save, index=False)

    out_bkt_fp = None
    if args.out_bkt:
        os.makedirs(os.path.dirname(args.out_bkt) or ".", exist_ok=True)
        pd.DataFrame(columns=['order_id','user_id','skill_name','correct']).to_csv(args.out_bkt, index=False)

    # 項目生成
    kdist = _parse_kdist(args.k_dist)
    items = make_items(SKILLS, ITEMS,
                       a_mu=args.a_mu, a_sigma=args.a_sigma,
                       b_mean=args.b_mean, b_std=args.b_std,
                       c_mean=args.c_mean, c_kappa=args.c_kappa,
                       kdist=kdist)

    # θ平均チューニング（任意）
    theta_mean = float(args.theta_mean)
    if args.tune_theta:
        best_gap, best_tm = 999.0, theta_mean
        grid = np.linspace(theta_mean - 1.5, theta_mean + 1.5, int(args.tune_grid))
        for tm in grid:
            est = quick_estimate_acc(tm, args.theta_std, items, SKILLS, trials=3000)
            gap = abs(est - args.target_acc)
            if gap < best_gap:
                best_gap, best_tm = gap, tm
        theta_mean = best_tm
        print(f"[tune] theta_mean -> {theta_mean:.3f} (target={args.target_acc:.2f})")

    # 能力初期化
    trait = init_trait(USERS, SKILLS, theta_mean, args.theta_std)

    # スタート時間（UNIX秒）
    date = time.mktime(time.strptime("2019-04-01 12:00:00", "%Y-%m-%d %H:%M:%S"))
    user_last_update = np.full((USERS, ITEMS), date, dtype=float)

    correct_cnt = 0
    order_id = 0

    for _ in range(RECORDS):
        user_id = np.random.randint(USERS)
        item_id = np.random.randint(ITEMS)

        items_match_item_id = items[items['item_id'] == item_id]
        skills = items_match_item_id['skill'].values.astype(np.int64)

        # 経過時間→衰退
        passed = date - user_last_update[user_id, item_id]
        if passed > 0:
            for s in skills:
                trait[user_id, s] = decline_trait(trait[user_id, s], passed)

        # 結果
        result = calc_result(trait[user_id], items_match_item_id, skills)
        correct_cnt += result

        # 元フォーマットで追記
        row = pd.DataFrame([[
            result,
            user_id,
            item_id,
            ",".join(map(str, skills)),
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(date))
        ]], columns=header)
        row.to_csv(args.save, mode='a', header=False, index=False)

        # BKT用（任意）
        if args.out_bkt:
            if args.bkt_skill == 'first':
                sk = int(skills[0])
                r = pd.DataFrame([[order_id, f"u{user_id}", f"s{sk}", int(result)]],
                                 columns=['order_id','user_id','skill_name','correct'])
                r.to_csv(args.out_bkt, mode='a', header=False, index=False)
            elif args.bkt_skill == 'random':
                sk = int(np.random.choice(skills))
                r = pd.DataFrame([[order_id, f"u{user_id}", f"s{sk}", int(result)]],
                                 columns=['order_id','user_id','skill_name','correct'])
                r.to_csv(args.out_bkt, mode='a', header=False, index=False)
            else:  # explode
                rows = [[order_id, f"u{user_id}", f"s{int(sk)}", int(result)] for sk in skills]
                r = pd.DataFrame(rows, columns=['order_id','user_id','skill_name','correct'])
                r.to_csv(args.out_bkt, mode='a', header=False, index=False)

        # 時刻更新・成長
        date += np.random.randint(1, 60)
        user_last_update[user_id, item_id] = date
        for s in skills:
            trait[user_id, s] = trait[user_id, s] + np.random.beta(10, 10) * INCREASE

        order_id += 1

    acc = correct_cnt / float(RECORDS)
    print(f"done. rows={RECORDS}  acc={acc:.4f}  save={args.save}")
    if args.out_bkt:
        print(f"bkt_out={args.out_bkt}  mode={args.bkt_skill}")


if __name__ == '__main__':
    main()
