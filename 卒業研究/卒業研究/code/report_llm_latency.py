#!/usr/bin/env python
"""Aggregate LLM latency from json_history files.

機能:
    - json_history 配下の *.json を走査して latency_ms を集計
    - 全体平均 / event別 / model別 / user別の統計を出力
    - 必要なら TSV で保存

使い方例:
    python -m code.report_llm_latency \
        --json-dir runs/json_history
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Iterable


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Report average LLM response latency from json_history.")
    ap.add_argument("--json-dir", default="runs/json_history", help="Directory containing json_history files")
    ap.add_argument("--user-id", help="Filter by user_id")
    ap.add_argument("--event", choices=["selection", "feedback"], help="Filter by event")
    ap.add_argument("--model", help="Filter by model name")
    ap.add_argument("--out-tsv", help="Optional output TSV path")
    return ap.parse_args()


def iter_latency_rows(json_dir: Path) -> Iterable[dict]:
    for p in sorted(json_dir.rglob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        lat = obj.get("latency_ms")
        if lat is None:
            continue
        try:
            lat_f = float(lat)
        except (TypeError, ValueError):
            continue
        row = {
            "path": str(p),
            "user_id": str(obj.get("user_id", "")),
            "event": str(obj.get("event", "")),
            "model": str(obj.get("model", "")),
            "latency_ms": lat_f,
        }
        yield row


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "median_ms": 0.0}
    return {
        "count": len(values),
        "avg_ms": sum(values) / len(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "median_ms": median(values),
    }


def group_stats(rows: list[dict], key: str) -> list[tuple[str, dict]]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        k = r.get(key, "") or "(empty)"
        groups.setdefault(k, []).append(float(r["latency_ms"]))
    out: list[tuple[str, dict]] = []
    for k in sorted(groups):
        out.append((k, summarize(groups[k])))
    return out


def fmt_stats(name: str, s: dict) -> str:
    return (
        f"{name}: n={s['count']} avg={s['avg_ms']:.1f}ms "
        f"median={s['median_ms']:.1f}ms min={s['min_ms']:.1f}ms max={s['max_ms']:.1f}ms"
    )


def main() -> None:
    args = parse_args()
    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"json dir not found: {json_dir}")

    rows = list(iter_latency_rows(json_dir))
    if args.user_id:
        rows = [r for r in rows if r["user_id"] == args.user_id]
    if args.event:
        rows = [r for r in rows if r["event"] == args.event]
    if args.model:
        rows = [r for r in rows if r["model"] == args.model]

    if not rows:
        print("[info] no latency rows matched filters.")
        return

    overall = summarize([float(r["latency_ms"]) for r in rows])
    print(fmt_stats("overall", overall))
    print("--- by event ---")
    for k, s in group_stats(rows, "event"):
        print(fmt_stats(k, s))
    print("--- by model ---")
    for k, s in group_stats(rows, "model"):
        print(fmt_stats(k, s))
    print("--- by user ---")
    for k, s in group_stats(rows, "user_id"):
        print(fmt_stats(k, s))

    if args.out_tsv:
        out = Path(args.out_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "group_type\tgroup_key\tcount\tavg_ms\tmedian_ms\tmin_ms\tmax_ms",
            (
                "overall\toverall\t{count}\t{avg_ms:.6f}\t{median_ms:.6f}\t{min_ms:.6f}\t{max_ms:.6f}".format(
                    **overall
                )
            ),
        ]
        for group_type, key in [("event", "event"), ("model", "model"), ("user", "user_id")]:
            for name, s in group_stats(rows, key):
                lines.append(
                    f"{group_type}\t{name}\t{s['count']}\t{s['avg_ms']:.6f}\t{s['median_ms']:.6f}\t{s['min_ms']:.6f}\t{s['max_ms']:.6f}"
                )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[info] wrote tsv -> {out}")


if __name__ == "__main__":
    main()

