#!/usr/bin/env python
"""Quick sanity check for Gemini API connectivity.

機能:
    - 環境変数 GEMINI_API_KEY を読んで Google Gemini API を初期化
    - 任意のプロンプトを投げて、LLM の応答テキストを標準出力に表示

使い方例:
    python -m code.gemini_ping --prompt "テストです。1行メッセージをください。"
"""
from __future__ import annotations

import argparse
import os
import sys

from google import genai


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Send a simple prompt to Gemini API and print the response.")
    ap.add_argument(
        "--prompt",
        default="こんにちは、これは Gemini API の接続テストです。1行で短い返事をください。",
        help="LLM に送るテキスト",
    )
    ap.add_argument("--model", default="gemini-2.5-flash", help="使用する Gemini モデル名")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("error: GEMINI_API_KEY が設定されていません。`export GEMINI_API_KEY=...` を実行してください。", file=sys.stderr)
        sys.exit(2)

    client = genai.Client(api_key=api_key)

    print(f"[info] sending prompt to {args.model}")
    response = client.models.generate_content(
        model=args.model,
        contents=[args.prompt],
    )
    text = getattr(response, "text", "")
    if not text and getattr(response, "output", None):
        # fallback: extract first content part if text property is empty
        try:
            text = response.output[0].content[0].text
        except Exception:
            text = str(response)
    print("=== Gemini response ===")
    print(text.strip())


if __name__ == "__main__":
    main()
