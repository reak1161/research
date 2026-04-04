#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graphviz import Digraph


def parse_ts(ts: str) -> datetime:
    # 例: "2026-01-26T01:06:15.151367+00:00"
    return datetime.fromisoformat(ts)


@dataclass
class Turn:
    t: int
    timestamp: datetime
    src_path: Path
    user_id: str
    payload: Dict[str, Any]
    choice_index: int


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_turns(json_dir: Path, user_id: Optional[str] = None) -> List[Turn]:
    turns_raw: List[Tuple[datetime, Path, Dict[str, Any]]] = []

    for p in sorted(json_dir.rglob("*.json")):
        obj = load_json(p)
        if not obj:
            continue
        if obj.get("event") != "selection":
            continue
        if "payload" not in obj or "response" not in obj:
            continue
        if user_id is not None and str(obj.get("user_id")) != str(user_id):
            continue

        ts = obj.get("timestamp")
        if not ts:
            continue
        try:
            dt = parse_ts(ts)
        except Exception:
            continue

        turns_raw.append((dt, p, obj))

    turns_raw.sort(key=lambda x: x[0])

    turns: List[Turn] = []
    for i, (dt, p, obj) in enumerate(turns_raw, start=1):
        resp = obj.get("response", {})
        payload = obj.get("payload", {})
        choice_index = resp.get("choice_index", None)
        if choice_index is None:
            continue
        turns.append(
            Turn(
                t=i,
                timestamp=dt,
                src_path=p,
                user_id=str(obj.get("user_id", "")),
                payload=payload,
                choice_index=int(choice_index),
            )
        )
    return turns


def shorten(s: str, n: int = 42) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def render_chain(
    turns: List[Turn],
    out_path: str = "selection_chain",
    fmt: str = "png",
    top_k: Optional[int] = None,
    show_question: bool = True,
) -> str:
    g = Digraph("chain", format=fmt)
    g.attr(rankdir="TB", nodesep="0.25", ranksep="0.55", splines="ortho")

    g.node(
        "root",
        "Session start",
        shape="box",
        style="rounded,filled",
        fillcolor="#f2f2f2",
    )

    prev_chosen = "root"

    for turn in turns:
        payload = turn.payload
        cands: List[Dict[str, Any]] = payload.get("candidates_indexed") or payload.get("candidates") or []
        if not cands:
            # candidates_indexed が無いログはスキップ
            continue

        # candidates_indexed 前提で index を持たせる（無い場合は並び順）
        normalized = []
        for j, c in enumerate(cands):
            idx = c.get("index", j)
            normalized.append((int(idx), c))
        normalized.sort(key=lambda x: x[0])

        # top_kが指定されていれば、p_finalで上位だけ表示（選択が落ちないように補正）
        chosen_idx = turn.choice_index
        if top_k is not None and top_k > 0:
            scored = []
            for idx, c in normalized:
                p = c.get("p_final")
                p = float(p) if p is not None else -1.0
                scored.append((p, idx, c))
            scored.sort(reverse=True, key=lambda x: x[0])
            keep = scored[:top_k]
            # 選択問題が含まれてないなら追加
            if all(idx != chosen_idx for _, idx, _ in keep):
                for p, idx, c in scored:
                    if idx == chosen_idx:
                        keep.append((p, idx, c))
                        break
            # 表示順は index 順に戻す
            keep.sort(key=lambda x: x[1])
            normalized = [(idx, c) for _, idx, c in keep]

        cluster_name = f"cluster_t{turn.t}"
        label_head = f"t={turn.t}  {turn.timestamp.strftime('%H:%M:%S')}"

        with g.subgraph(name=cluster_name) as sg:
            sg.attr(label=label_head, color="#cccccc")

            # 候補ノード
            for idx, c in normalized:
                node_id = f"t{turn.t}_c{idx}"

                domain = c.get("domain", "")
                tier = c.get("tier", "")
                p_final = c.get("p_final")
                q = c.get("question_text", "")

                lines = [f"[{idx}] item {c.get('item_id','')}", f"{domain} tier {tier}", f"p_final={float(p_final):.3f}" if p_final is not None else "p_final=?"]
                if show_question and q:
                    lines.append(shorten(q, 52))

                node_label = "\n".join(lines)

                is_chosen = (idx == chosen_idx)
                sg.node(
                    node_id,
                    node_label,
                    shape="box",
                    style="rounded,filled",
                    fillcolor="#b6f0c1" if is_chosen else "#ffffff",
                    penwidth="2.5" if is_chosen else "1",
                )

        # 前の選択ノード → 今ターン候補へ枝
        for idx, _c in normalized:
            node_id = f"t{turn.t}_c{idx}"
            if idx == chosen_idx:
                g.edge(prev_chosen, node_id, penwidth="3")
            else:
                g.edge(prev_chosen, node_id, style="dashed", penwidth="1")

        prev_chosen = f"t{turn.t}_c{chosen_idx}"

    g.render(out_path, cleanup=True)
    return f"{out_path}.{fmt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=str, required=True, help="json_history/<user名> のディレクトリ")
    ap.add_argument("--user_id", type=str, default=None, help="特定ユーザに絞る（任意）")
    ap.add_argument("--out", type=str, default="selection_chain", help="出力ファイルのベース名")
    ap.add_argument("--fmt", type=str, default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--top_k", type=int, default=0, help="各ターンで表示する候補数（0なら全表示）")
    ap.add_argument("--hide_question", action="store_true", help="question_textをノードに出さない")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise SystemExit(f"not found: {json_dir}")

    turns = collect_turns(json_dir, user_id=args.user_id)
    if not turns:
        raise SystemExit("selectionログが見つからない（event=selection のjsonが無い/壊れてる/フィルタ条件が違う）")

    out = render_chain(
        turns,
        out_path=args.out,
        fmt=args.fmt,
        top_k=(args.top_k if args.top_k > 0 else None),
        show_question=(not args.hide_question),
    )
    print(out)


if __name__ == "__main__":
    main()
