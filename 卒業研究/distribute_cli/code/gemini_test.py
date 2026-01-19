from google import genai
import os

# 環境変数からキーを読む（AI Studio 推奨スタイル）
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

resp = client.models.generate_content(
    model="gemini-2.5-flash",  # お試しならこれでOK
    contents="Linuxのファイル操作について簡単なクイズを2問作って。"
)

print(resp.text)
