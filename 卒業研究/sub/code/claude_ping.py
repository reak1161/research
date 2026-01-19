"""Claude API ping.

使い方:
  python -m code.claude_ping --prompt "テストです。1行返答をください。"

前提:
  export ANTHROPIC_API_KEY="..."
  export ANTHROPIC_MODEL="claude-3-5-sonnet-20240620"  # 省略可
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
    ap.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"))
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[error] ANTHROPIC_API_KEY is not set.")
        return 2

    payload = {
        "model": args.model,
        "max_tokens": 200,
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": args.prompt}
        ],
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data.get("content", [])
        text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        print("".join(text_parts).strip())
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
