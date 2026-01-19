from openai import OpenAI
import os

def main():
    # 環境変数からキー取得（設定してなかったらエラーにする）
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が環境変数に設定されていません")

    client = OpenAI(api_key=api_key)

    # 一番軽い動作確認：短いチャットを投げてみる
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # 課金的にも軽いモデル
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "こんにちは、一言だけ自己紹介して。"},
        ],
        max_tokens=50,
    )

    print("=== API 応答 ===")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
