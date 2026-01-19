from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_LOGS = BASE_DIR / "csv" / "sim_logs.csv"
CSV_ITEMS = BASE_DIR / "csv" / "items_sample_lpic.csv"
CSV_SCORES = BASE_DIR / "runs" / "sim_user_item_scores.csv"
CSV_BKT_PARAMS = BASE_DIR / "csv" / "bkt_params_sim_multi.csv"
CSV_IRT_ITEMS = BASE_DIR / "csv" / "irt_items_estimated.csv"
W_BKT = 0.6  # demo用の BKT vs IRT 重み
ANTHROPIC_MODEL_FAST = os.getenv("ANTHROPIC_MODEL_FAST", "claude-3-haiku-20240307")
ANTHROPIC_MODEL_ACCURATE = os.getenv("ANTHROPIC_MODEL_ACCURATE", "claude-sonnet-4-20250514")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

sys.path.append(str(BASE_DIR / "code"))
try:
    import bkt_core  # type: ignore
except Exception:
    bkt_core = None  # type: ignore



def _safe_logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _clean_float(x: float | None, default: float | None = None) -> float | None:
    if x is None:
        return default
    if isinstance(x, (int, float)) and math.isfinite(x):
        return float(x)
    return default


def _clean_history(records: List[Dict]) -> List[Dict]:
    cleaned = []
    for rec in records:
        rec = {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in rec.items()}
        cleaned.append(rec)
    return cleaned


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _extract_json(text: str) -> Optional[Dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _normalize_text_cell(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _select_anthropic_model(mode: str, override: Optional[str] = None) -> str:
    if override:
        return override
    if mode == "accurate":
        return ANTHROPIC_MODEL_ACCURATE
    return ANTHROPIC_MODEL_FAST


def _call_anthropic_llm(
    candidates: List[Dict],
    user_id: str,
    mode: str = "fast",
    model_override: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    if not ANTHROPIC_API_KEY:
        return None
    if not candidates:
        return None
    model = _select_anthropic_model(mode, model_override)
    prompt = {
        "user_id": user_id,
        "candidates": [
            {
                "item_id": c.get("item_id"),
                "domain": c.get("domain"),
                "p_final": c.get("p_final"),
                "question_text": c.get("question_text"),
            }
            for c in candidates
        ],
    }
    system = (
        "You are a tutor selecting the best next question. "
        "Return JSON only with keys: item_id, reason_ja, motivation_ja. "
        "reason_ja should explain why this item suits the learner. "
        "motivation_ja should be a short motivational sentence in Japanese."
    )
    payload = {
        "model": model,
        "max_tokens": 200,
        "temperature": 0.4,
        "system": system,
        "messages": [
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data.get("content", [])
        text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        text = "".join(text_parts).strip()
        result = json.loads(text)
        if not isinstance(result, dict):
            return None
        item_id = str(result.get("item_id", "")).strip()
        reason = str(result.get("reason_ja", "")).strip()
        motivation = str(result.get("motivation_ja", "")).strip()
        if not item_id:
            return None
        return {"item_id": item_id, "reason_ja": reason, "motivation_ja": motivation}
    except json.JSONDecodeError:
        parsed = _extract_json(text)
        if isinstance(parsed, dict):
            try:
                item_id = str(parsed.get("item_id", "")).strip()
                reason = str(parsed.get("reason_ja", "")).strip()
                motivation = str(parsed.get("motivation_ja", "")).strip()
                if not item_id:
                    return None
                return {"item_id": item_id, "reason_ja": reason, "motivation_ja": motivation}
            except Exception:
                return None
        return None
    except (urllib.error.URLError, KeyError, ValueError):
        return None


def load_items() -> pd.DataFrame:
    df = pd.read_csv(CSV_ITEMS)
    df["item_id"] = df["item_id"].astype(str)
    # domain 列が無い場合は L2 を domain として使う
    if "domain" not in df.columns:
        if "L2" in df.columns:
            df = df.rename(columns={"L2": "domain"})
        else:
            df["domain"] = "default"
    if "tier" in df.columns:
        df["tier"] = pd.to_numeric(df["tier"], errors="coerce").fillna(1).astype(int)
    else:
        df["tier"] = 1
    # choices がなければ choice_* から組み立てる
    if "choices" not in df.columns:
        choice_cols = [c for c in df.columns if c.lower().startswith("choice_")]
        if choice_cols:
            def _choices(row: pd.Series) -> List[str]:
                vals: List[str] = []
                for c in sorted(choice_cols):
                    val = row.get(c)
                    if isinstance(val, str) and val.strip():
                        vals.append(val.strip())
                return vals
            df["choices"] = df.apply(_choices, axis=1)
    return df


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(CSV_SCORES)
    df["item_id"] = df["item_id"].astype(str)
    if "P_final" in df.columns:
        df["P_final"] = df["P_final"].apply(lambda x: _clean_float(x, 0.5))
    if "P_bkt" in df.columns:
        df["P_bkt"] = df["P_bkt"].apply(lambda x: _clean_float(x, 0.5))
    return df


def load_logs() -> pd.DataFrame:
    df = pd.read_csv(CSV_LOGS)
    df["item_id"] = df["item_id"].astype(str)
    if "domain" not in df.columns:
        if "L2" in df.columns:
            df = df.rename(columns={"L2": "domain"})
        else:
            df["domain"] = "default"
    if "p_L_after" not in df.columns:
        df["p_L_after"] = None
    return df


def load_irt_items() -> Dict[str, Dict[str, float]]:
    p = CSV_IRT_ITEMS
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if not {"item_id", "a", "b", "c"}.issubset(df.columns):
        return {}
    df["item_id"] = df["item_id"].astype(str)
    return {row["item_id"]: {"a": float(row["a"]), "b": float(row["b"]), "c": float(row["c"])} for _, row in df.iterrows()}


def load_bkt_params() -> Dict[str, Dict[str, float]]:
    if not bkt_core:
        return {}
    df = pd.read_csv(CSV_BKT_PARAMS)
    if "skill_name" not in df.columns:
        return {}
    # L2 がメインなので skill_field==L2 を優先
    if "skill_field" in df.columns:
        df = df[df["skill_field"] == "L2"]
    params = {}
    for _, row in df.iterrows():
        params[str(row["skill_name"])] = {
            "L0": float(row["L0"]),
            "T": float(row["T"]),
            "S": float(row["S"]),
            "G": float(row["G"]),
        }
    return params


items_df = load_items()
scores_df = load_scores()
logs_df = load_logs()
bkt_params = load_bkt_params()
state_cache: Dict[str, Dict[str, float]] = {}
irt_items = load_irt_items()
current_question: Dict[str, Dict] = {}

app = FastAPI(title="BKT×IRT Demo API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # "*" で確実に ACAO を付ける
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cors_header(request, call_next):
    try:
        response = await call_next(request)
    except Exception as exc:
        # ensure CORS headers even on error
        response = JSONResponse({"detail": str(exc)}, status_code=500)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response


class LoginRequest(BaseModel):
    user_id: str
    role: str  # "learner" or "admin"


class QuestionResponse(BaseModel):
    item_id: str
    domain: str
    p_final: float
    question_text: str
    choices: Optional[List[str]] = None


def get_user_state(user_id: str) -> Dict[str, float]:
    user_scores = scores_df[scores_df["user_id"].astype(str) == str(user_id)]
    if user_scores.empty:
        raise HTTPException(status_code=404, detail="user not found in scores")
    grouped = user_scores.groupby("domain")["P_final"].mean().to_dict()
    # NaN/inf を None に
    return {k: _clean_float(v) for k, v in grouped.items()}


def get_user_history(user_id: str) -> List[Dict]:
    hist = logs_df[logs_df["user_id"].astype(str) == str(user_id)].copy()
    if hist.empty:
        return []
    hist = hist.sort_values("order_id")
    # P_final を結合
    if "P_final" in scores_df.columns:
        merged = hist.merge(
            scores_df[["user_id", "item_id", "P_final"]],
            on=["user_id", "item_id"],
            how="left",
        )
    else:
        merged = hist
    # sanitize all NaN to None and float columns to safe values
    merged = merged.where(pd.notna(merged), None)
    for col in ("P_final", "p_correct_sim"):
        if col in merged.columns:
            merged[col] = merged[col].apply(lambda x: _clean_float(x))
    merged = merged.where(pd.notna(merged), None)
    for col in ("P_final", "p_correct_sim", "p_L_after"):
        if col in merged.columns:
            merged[col] = merged[col].apply(lambda x: _clean_float(x))
    hist_list = _clean_history(jsonable_encoder(merged.to_dict(orient="records")))
    return _sanitize(hist_list)


def _get_user_tier_mastery(user_id: str) -> Dict[int, float]:
    user_logs = logs_df[logs_df["user_id"].astype(str) == str(user_id)]
    if user_logs.empty:
        return {}
    merged = user_logs.merge(items_df[["item_id", "tier"]], on="item_id", how="left")
    if "p_L_after" in merged.columns and merged["p_L_after"].notna().any():
        series = merged.groupby("tier")["p_L_after"].mean()
    else:
        series = merged.groupby("tier")["correct"].mean()
    return {int(k): float(v) for k, v in series.items() if pd.notna(v)}


def _allowed_tiers(mastery: Dict[int, float], threshold: float) -> List[int]:
    max_tier = max([1] + list(mastery.keys()))
    allowed = {1}
    for t in range(2, max_tier + 1):
        if mastery.get(t - 1, 0.0) >= threshold:
            allowed.add(t)
    return sorted(allowed)


def _filter_candidates_by_tier(df: pd.DataFrame, user_id: str, threshold: float) -> pd.DataFrame:
    if "tier" not in df.columns:
        return df
    mastery = _get_user_tier_mastery(user_id)
    allowed = _allowed_tiers(mastery, threshold)
    return df[df["tier"].isin(allowed)]


def select_next_question(
    user_id: str,
    top_k: int = 5,
    tier_mode: str = "flat",
    tier_threshold: float = 0.6,
) -> List[QuestionResponse]:
    user_scores = scores_df[scores_df["user_id"].astype(str) == str(user_id)].copy()
    if user_scores.empty:
        raise HTTPException(status_code=404, detail="user not found in scores")
    user_scores = user_scores.merge(items_df[["item_id", "tier"]], on="item_id", how="left")
    # 直近未出題を優先: logs で出ていないものを先頭から
    asked = set(logs_df[logs_df["user_id"].astype(str) == str(user_id)]["item_id"].astype(str).tolist())
    candidates = user_scores[~user_scores["item_id"].isin(asked)]
    if candidates.empty:
        candidates = user_scores  # 全問出たら全体から
    if tier_mode == "prereq":
        filtered = _filter_candidates_by_tier(candidates, user_id, tier_threshold)
        if not filtered.empty:
            candidates = filtered
    candidates = candidates.sort_values("P_final", ascending=False).head(top_k)
    result = []
    for _, row in candidates.iterrows():
        qtext = items_df.loc[items_df["item_id"] == row["item_id"], "question_text"]
        question_text = qtext.iloc[0] if not qtext.empty else ""
        ch = items_df.loc[items_df["item_id"] == row["item_id"], "choices"]
        choices = ch.iloc[0] if not ch.empty else None
        p_final = _clean_float(row["P_final"], 0.5)
        result.append(
            QuestionResponse(
                item_id=str(row["item_id"]),
                domain=str(row["domain"]),
                p_final=p_final if p_final is not None else 0.5,
                question_text=question_text,
                choices=choices if isinstance(choices, list) else None,
            )
        )
    return result


def ensure_state(user_id: str, domain: str) -> float:
    if not bkt_core or domain not in bkt_params:
        return 0.5
    user_state = state_cache.setdefault(user_id, {})
    if domain not in user_state:
        user_state[domain] = bkt_params[domain]["L0"]
    return user_state[domain]


def update_state(user_id: str, domain: str, correct: int) -> Dict[str, float]:
    if not bkt_core or domain not in bkt_params:
        return {"p_L_after": 0.5, "p_next": 0.5}
    user_state = state_cache.setdefault(user_id, {})
    params = bkt_core.BKTParams.from_row(
        {"L0": bkt_params[domain]["L0"], "T": bkt_params[domain]["T"], "S": bkt_params[domain]["S"], "G": bkt_params[domain]["G"]}
    )
    prior = user_state.get(domain, params.L0)
    info = bkt_core.update_state(prior, params, correct)
    user_state[domain] = info["p_L_after"]
    return info


class AnswerRequest(BaseModel):
    user_id: str
    item_id: str
    answer: str


@app.post("/login")
def login(req: LoginRequest):
    if req.role not in {"learner", "admin"}:
        raise HTTPException(status_code=400, detail="invalid role")
    return {"token": f"demo-{req.role}-{req.user_id}", "role": req.role}


@app.get("/learner/{user_id}/state")
def learner_state(user_id: str):
    return {"user_id": user_id, "state": get_user_state(user_id)}


@app.get("/learner/{user_id}/history")
def learner_history(user_id: str):
    return {"user_id": user_id, "history": get_user_history(user_id)}


@app.post("/next-question")
def next_question(user_id: str, top_k: int = 5):
    return {"user_id": user_id, "candidates": select_next_question(user_id, top_k=top_k)}


@app.get("/admin/learners")
def admin_learners():
    # 簡易集計: 直近の正答率・最新回答時刻
    users = []
    for uid, group in logs_df.groupby("user_id"):
        acc = group["correct"].mean() if "correct" in group else None
        last_ts = group["timestamp"].iloc[-1] if "timestamp" in group else None
        users.append({"user_id": uid, "recent_acc": acc, "last_ts": last_ts})
    return {"learners": users}


@app.get("/admin/learner/{user_id}")
def admin_learner(user_id: str):
    return {
        "user_id": user_id,
        "state": get_user_state(user_id),
        "history": get_user_history(user_id),
        "current_question": current_question.get(user_id),
    }


def evaluate_answer(raw_answer: str, item_row: pd.Series) -> bool:
    correct_key = _normalize_text_cell(item_row.get("correct_key", ""))
    correct_text = _normalize_text_cell(item_row.get("correct_text", ""))
    ans_type = str(item_row.get("answer_type", "text")).lower()
    user_ans = raw_answer.strip()
    if ans_type == "mcq":
        # map 1/2/3/4 to A/B/C/D
        def to_letter(token: str) -> str:
            if token.isdigit():
                val = int(token)
                if 1 <= val <= 26:
                    return chr(ord("A") + val - 1)
            return token.upper()
        return to_letter(user_ans) == to_letter(correct_key)
    # text: correct_key が空なら correct_text を使う
    target = correct_key or correct_text
    return user_ans.lower() == target.lower()


@app.post("/session/answer")
def session_answer(req: AnswerRequest):
    # find item
    row = items_df[items_df["item_id"] == req.item_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="item not found")
    item_row = row.iloc[0]
    is_correct = int(evaluate_answer(req.answer, item_row))
    correct_key = _normalize_text_cell(item_row.get("correct_key", ""))
    correct_text = _normalize_text_cell(item_row.get("correct_text", ""))
    if not correct_key and correct_text:
        correct_key = correct_text
    choices = item_row.get("choices", None)
    domain = str(item_row["domain"])
    state_info = update_state(req.user_id, domain, is_correct)
    # append to in-memory logs
    global logs_df
    new_log = {
        "order_id": (logs_df["order_id"].max() + 1) if not logs_df.empty else 1,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "user_id": req.user_id,
        "item_id": req.item_id,
        "domain": domain,
        "correct": is_correct,
        "p_L_after": _clean_float(state_info.get("p_L_after")),
    }
    if logs_df.empty:
        logs_df = pd.DataFrame([new_log])
    else:
        logs_df = pd.concat([logs_df, pd.DataFrame([new_log])], ignore_index=True)
    # 最新履歴も返す
    history = get_user_history(req.user_id)
    resp = {
        "user_id": req.user_id,
        "item_id": req.item_id,
        "domain": domain,
        "correct": int(is_correct),
        "p_L_after": _clean_float(state_info.get("p_L_after")),
        "p_next": _clean_float(state_info.get("p_next")),
        "correct_key": correct_key,
        "correct_text": correct_text,
        "choices": choices if isinstance(choices, list) else None,
        "history": _clean_history(history),
    }
    return jsonable_encoder(_sanitize(resp))


@app.post("/session/next")
def session_next(
    user_id: str,
    top_k: int = 5,
    use_llm: bool = False,
    set_current: bool = True,
    tier_mode: str = "flat",
    tier_threshold: float = 0.6,
    llm_mode: str = "fast",
    llm_model: Optional[str] = None,
):
    user_state = state_cache.get(user_id, {})
    asked = set(logs_df[logs_df["user_id"].astype(str) == str(user_id)]["item_id"].astype(str).tolist())
    candidates: List[QuestionResponse] = []
    pool = items_df[~items_df["item_id"].isin(asked)].head(20)
    if tier_mode == "prereq":
        pool = _filter_candidates_by_tier(pool, user_id, tier_threshold)
        if pool.empty:
            pool = items_df[~items_df["item_id"].isin(asked)].head(20)
    for _, row in pool.iterrows():
        item_id = str(row["item_id"])
        dom = str(row["domain"])
        p_bkt = float(user_state.get(dom, 0.5))
        p_irt = None
        if item_id in irt_items:
            theta = _safe_logit(p_bkt)
            itm = irt_items[item_id]
            p_irt = itm["c"] + (1.0 - itm["c"]) / (1.0 + math.exp(-itm["a"] * (theta - itm["b"])))
        p_final = p_bkt if p_irt is None else (W_BKT * p_bkt + (1.0 - W_BKT) * p_irt)
        p_final = _clean_float(p_final, 0.5)
        choices = row.get("choices", None)
        candidates.append(
            QuestionResponse(
                item_id=item_id,
                domain=dom,
                p_final=p_final if p_final is not None else 0.5,
                question_text=str(row.get("question_text", "")),
                choices=choices if isinstance(choices, list) else None,
            )
        )
    # fallback: if no state (新規ユーザー) はオフラインスコアで補う
    if not user_state:
        base_candidates = select_next_question(
            user_id,
            top_k=top_k,
            tier_mode=tier_mode,
            tier_threshold=tier_threshold,
        )
        base = {"user_id": user_id, "candidates": base_candidates}
        if use_llm:
            cand_list = jsonable_encoder(base["candidates"])
            llm_choice = _call_anthropic_llm(cand_list, user_id, llm_mode, llm_model)
            if llm_choice:
                hit = next((c for c in cand_list if str(c.get("item_id")) == str(llm_choice["item_id"])), None)
                if hit:
                    llm_choice["question_text"] = hit.get("question_text")
                    llm_choice["domain"] = hit.get("domain")
                    llm_choice["p_final"] = hit.get("p_final")
                base["llm_choice"] = llm_choice
                if set_current:
                    current_question[user_id] = hit or cand_list[0]
            elif cand_list:
                if set_current:
                    current_question[user_id] = cand_list[0]
        elif base_candidates:
            if set_current:
                current_question[user_id] = jsonable_encoder(base_candidates[0])
        return base
    candidates = sorted(candidates, key=lambda c: c.p_final, reverse=True)[: top_k]
    resp = {"user_id": user_id, "candidates": candidates}
    if use_llm:
        cand_list = jsonable_encoder(candidates)
        llm_choice = _call_anthropic_llm(cand_list, user_id, llm_mode, llm_model)
        if llm_choice:
            hit = next((c for c in cand_list if str(c.get("item_id")) == str(llm_choice["item_id"])), None)
            if hit:
                llm_choice["question_text"] = hit.get("question_text")
                llm_choice["domain"] = hit.get("domain")
                llm_choice["p_final"] = hit.get("p_final")
            resp["llm_choice"] = llm_choice
            if set_current:
                current_question[user_id] = hit or cand_list[0]
        elif cand_list:
            if set_current:
                current_question[user_id] = cand_list[0]
    return resp
