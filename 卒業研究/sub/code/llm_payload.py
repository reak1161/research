"""Common JSON payload builder for LLM-based problem selection.

機能:
    - 候補問題を `CandidateItem` dataclass で統一管理
    - ユーザー状態 + 候補上位K件をまとめた JSON ペイロードを生成
    - offline/online 両モードで利用できる共通インターフェースを提供
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CandidateItem:
    item_id: str
    domain: str
    p_final: float
    difficulty: str | None = None
    tags: List[str] | None = None
    question_text: str | None = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "item_id": self.item_id,
            "domain": self.domain,
            "p_final": round(float(self.p_final), 4),
        }
        if self.difficulty:
            data["difficulty"] = self.difficulty
        if self.tags:
            data["tags"] = self.tags
        if self.question_text:
            data["question_text"] = self.question_text
        return data


def build_llm_payload(
    *,
    user_id: str | int,
    mode: str,
    user_state: Dict[str, float],
    candidates: List[CandidateItem],
    k: int = 5,
    mood_state: str | None = None,
    policy_hint: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Assemble a JSON-serializable payload for the LLM."""
    if mode not in {"offline", "online"}:
        raise ValueError("mode must be 'offline' or 'online'")
    selected = candidates[:k]
    return {
        "schema_version": "1.0",
        "mode": mode,
        "user_id": str(user_id),
        "user_state": {str(k): round(float(v), 4) for k, v in user_state.items()},
        "mood_state": mood_state,
        "policy_hint": policy_hint,
        "candidates": [cand.to_dict() for cand in selected],
    }
