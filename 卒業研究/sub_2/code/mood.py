"""Utilities for mood-aware candidate selection based on recent performance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class MoodThresholds:
    window: int = 5
    thr_low: float = 0.3
    thr_high: float = 0.8
    max_consec_wrong: int = 3


@dataclass
class MoodPolicy:
    target: float
    low: float
    high: float
    comment_ja: str


def get_mood_state(
    history: Iterable[Tuple[int, float | None]],
    *,
    thresholds: MoodThresholds = MoodThresholds(),
) -> str:
    """
    Determine mood_state from recent history.

    Args:
        history: iterable of (correct, p_final or None) in chronological order.
        thresholds: configuration for window size and thresholds.

    Returns:
        "FRUSTRATED" / "NORMAL" / "CONFIDENT"
    """
    hist = list(history)[-thresholds.window :]
    if not hist:
        return "NORMAL"
    correct_flags = [int(bool(c)) for c, _ in hist]
    recent_acc = sum(correct_flags) / len(correct_flags)

    # consecutive wrong count from the end
    consec_wrong = 0
    for c, _ in reversed(hist):
        if int(bool(c)) == 0:
            consec_wrong += 1
        else:
            break

    if recent_acc < thresholds.thr_low or consec_wrong >= thresholds.max_consec_wrong:
        return "FRUSTRATED"
    if recent_acc > thresholds.thr_high:
        return "CONFIDENT"
    return "NORMAL"


def mood_policy(mood: str) -> MoodPolicy:
    """Return target/low/high and a short Japanese comment for the given mood."""
    if mood == "FRUSTRATED":
        return MoodPolicy(
            target=0.75,
            low=0.60,
            high=0.90,
            comment_ja="最近の正答率が低いため、成功体験を増やす難易度を優先してください。",
        )
    if mood == "CONFIDENT":
        return MoodPolicy(
            target=0.65,
            low=0.50,
            high=0.85,
            comment_ja="余裕があるので、ややチャレンジ寄りの問題を含めてステップアップを狙ってください。",
        )
    return MoodPolicy(
        target=0.70,
        low=0.60,
        high=0.85,
        comment_ja="適正難易度（6〜7割正答見込み）の問題を中心に選んでください。",
    )


def select_candidates_by_mood(
    candidates: List[dict],
    mood: str,
    *,
    k: int = 10,
) -> List[dict]:
    """
    Filter and sort candidates by mood-aware P_final ranges.

    Args:
        candidates: list of dicts with at least {"p_final": float}
        mood: FRUSTRATED / NORMAL / CONFIDENT
        k: number of items to return

    Returns:
        Top-k candidates filtered by mood range and sorted by closeness to target.
    """
    policy = mood_policy(mood)
    filtered = [c for c in candidates if "p_final" in c and c["p_final"] is not None and policy.low <= c["p_final"] <= policy.high]
    scored = sorted(filtered, key=lambda c: abs(float(c["p_final"]) - policy.target))
    return scored[:k]
