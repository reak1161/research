"""Common JSON payload builder for LLM-based problem selection."""
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
        "candidates": [cand.to_dict() for cand in selected],
    }
