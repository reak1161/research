"""OpenAI API ping.

使い方:
  python -m code.openai_ping --prompt "テストです。1行返答をください。"

前提:
  export OPENAI_API_KEY="..."
  export OPENAI_MODEL="gpt-4o-mini"  # 省略可
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY is not set.")
        return 2

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Reply briefly."},
            {"role": "user", "content": args.prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 120,
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        print(text)
        return 0
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[error] HTTP {e.code}: {body}")
        return 1
    except Exception as e:
        print(f"[error] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
