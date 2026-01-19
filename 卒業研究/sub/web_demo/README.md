# Web Demo (Learner/Admin)

This is a tiny FastAPI demo that exposes mock APIs for:
- Learner: state, history, next-question (LLM候補はダミー)
- Admin: learners list, per-learner state/history

## 事前準備
- カレント: `cd ~/repos/research/research/卒業研究/main`
- Python 3.10, 仮想環境 `.venv` を使用
- 依存インストール（まだ入っていない場合）:
  ```bash
  pip install fastapi uvicorn[standard] pandas pydantic
  ```

## 起動
```bash
uvicorn web_demo.api:app --reload --port 8000
```

別ターミナルでフロント（静的HTML）を表示する場合:
```bash
cd web_demo
python -m http.server 5173 --directory frontend
# ブラウザで http://localhost:5173/index.html を開く
# （ディレクトリ一覧が出た場合は /index.html を指定してください）
```

## 主なエンドポイント
- `POST /login` （ボディ: `{"user_id": "...", "role": "learner|admin"}`）…簡易セッション（トークンは固定ダミー）
- `GET /learner/{user_id}/state` … P_bkt_total / P_final の概況
- `GET /learner/{user_id}/history` … 正誤履歴＋P_final 時系列
- `POST /next-question` … mood 判定＋候補 Top-K（LLM はダミー）を返す
- `GET /admin/learners` … 学習者一覧（直近正答率、最新 mood）
- `GET /admin/learner/{user_id}` … 上記 state/history の統合

## データソース
- `csv/sim_logs.csv` … 学習ログ
- `csv/items_sample_lpic.csv` … 問題マスタ
- `runs/sim_user_item_scores.csv` … オフライン計算済み P_final

これらを起動時に読み込み、メモリ上で提供します。簡易デモなので永続化や認証は最小限です。***
