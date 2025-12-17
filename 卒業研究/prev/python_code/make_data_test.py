#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import time
np.random.seed(0)

USER     = 200
SKILL    = 50
ITEM     = 500
RECORD   = 100000
INCREASE = 0.02
DECREASE = 0.002

save_file_name = "csv/200_50_500_100000.csv"

# 項目の作成（append廃止 → 一括生成）
def make_items():
    p = 0.1
    rows = []
    for i in range(ITEM):
        skills = np.random.permutation(np.arange(SKILL))[:np.random.randint(5) + 1]  # 1〜5個
        for s in np.sort(skills):
            a = np.random.lognormal(0, 0.5)
            b = np.random.normal(0, 2)
            c = np.random.beta(20 * p + 1, 20 * (1 - p) + 1)
            rows.append((i, int(s), a, b, c))
    return pd.DataFrame(rows, columns=['item_id', 'skill', 'a', 'b', 'c'])

# USER×SKILLの能力初期化
def init_trait():
    return np.random.normal(-10, 2, (USER, SKILL))

# 能力衰退
def decline_trait(trait, elapsed_sec):
    return trait - np.random.beta(10, 10) * DECREASE * (elapsed_sec / 3600.0)

# 3PL-IRT
def rsa(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-1.7 * a * (theta - b)))

# 解答成否
def calc_result(traits_row, items_rows, skills):
    prob = 1.0
    for skill in skills:
        row = items_rows.loc[items_rows['skill'] == skill].iloc[0]  # ← Seriesではなく1行
        prob *= rsa(traits_row[skill], float(row['a']), float(row['b']), float(row['c']))
        if prob <= 0.0:
            return 0
    return 1 if prob > np.random.random() else 0

# 生成開始
items = make_items()
trait = init_trait()

# スタート時間（UNIX秒）
date = time.mktime(time.strptime("2019-04-01 12:00:00", "%Y-%m-%d %H:%M:%S"))

# 最後に回答した日時（user×item）
user_last_update = np.full((USER, ITEM), date, dtype=float)

# 出力準備
os.makedirs(os.path.dirname(save_file_name) or ".", exist_ok=True)
header = ['result', 'user_id', 'item_id', 'skill', 'date']
pd.DataFrame(columns=header).to_csv(save_file_name, index=False)

# 生成ループ
for _ in range(RECORD):
    user_id = np.random.randint(USER)
    item_id = np.random.randint(ITEM)

    items_match_item_id = items[items['item_id'] == item_id]
    skills = items_match_item_id['skill'].values.astype(np.int64)

    # 経過時間→衰退
    passed = date - user_last_update[user_id, item_id]
    if passed > 0:
        for s in skills:
            trait[user_id, s] = decline_trait(trait[user_id, s], passed)

    # 結果
    result = calc_result(trait[user_id], items_match_item_id, skills)

    # 出力
    row = pd.DataFrame([[
        result,
        user_id,
        item_id,
        ",".join(map(str, skills)),
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(date))
    ]], columns=header)
    row.to_csv(save_file_name, mode='a', header=False, index=False)

    # 時刻更新・成長
    date += np.random.randint(1, 60)
    user_last_update[user_id, item_id] = date
    for s in skills:
        trait[user_id, s] = trait[user_id, s] + np.random.beta(10, 10) * INCREASE
