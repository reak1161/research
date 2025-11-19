# docs/setup_wsl_python310.md

> **ゴール**: WSL2(Ubuntu 24.04) 上に Python 3.10.13（pyenv）＋ プロジェクト venv(`~/repos/research/.venv`) を構築し、`numpy<2`, `pybind11>=2.12`, `pyBKT==1.4.1` をピン留め。VS Code(Remote‑WSL) で venv を自動アクティブにし、pyBKT の C++ 拡張が有効なことを確認する。

---

## 0) 事前確認：VS Code を WSL と同期

WSL のシェルでプロジェクトを開くのがコツ。

```bash
cd ~/repos/research/research/卒業研究/code/python_3_10
code .
```

左下が **`WSL: Ubuntu-24.04`** になっていれば OK。ならない場合は `F1 → WSL: Reopen Folder in WSL`。

> 混在注意：Windows 側の `C:\...` と WSL 側の `/home/...` を同時に編集しない。

---

## 1) OS 依存の準備（ビルド系パッケージ）

```bash
sudo apt update
sudo apt install -y build-essential git curl zip unzip ca-certificates \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libffi-dev liblzma-dev tk-dev uuid-dev
```

---

## 2) pyenv で Python 3.10.13 を用意

```bash
# pyenv を導入
curl https://pyenv.run | bash

# シェル初期化（~/.bashrc に追記）
cat >> ~/.bashrc <<'EOS'
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOS
source ~/.bashrc

# Python 3.10.13 をインストールし、このリポで採用
pyenv install 3.10.13
cd ~/repos/research
pyenv local 3.10.13
python -V   # -> Python 3.10.13
```

---

## 3) プロジェクト venv を作成 & 依存ピン留め

```bash
python -m venv ~/repos/research/.venv
source ~/repos/research/.venv/bin/activate

# pip を先に最新化
python -m pip install -U pip

# 依存（再現性のためピン留め）
pip install "numpy<2" "pybind11>=2.12" "pyBKT==1.4.1" pandas matplotlib ipykernel

# VS Code/Jupyter 用カーネル（任意）
python -m ipykernel install --user --name research310 --display-name "Python 3.10 (.venv)"

# requirements を保存（NumPy の条件を保持）
pip freeze | sed 's/^numpy==2\..*/numpy<2/' > requirements.txt
```

---

## 4) pyBKT の C++ 拡張が効いているか確認

```bash
# 実行ファイルと拡張モジュールのパスを確認
python - <<'PY'
import importlib, sys
print(sys.executable)
m = importlib.import_module("pyBKT.fit.E_step")
print(getattr(m, "__file__", m))
PY

# 学習～評価の最小テスト
python - <<'PY'
import pandas as pd
from pyBKT.models import Model

df = pd.DataFrame({
  'order_id': range(10),
  'user_id': ['u1']*10,
  'skill_name': ['add']*10,
  'correct': [0,0,1,0,1,1,1,0,1,1]
})

m = Model(seed=42, num_fits=1).fit(data=df, defaults={
  'order_id':'order_id','user_id':'user_id',
  'skill_name':'skill_name','correct':'correct'
})
print("AUC:", m.evaluate(data=df, metric='auc'))
PY
```

---

## 5) VS Code の設定（venv 自動アクティブ）

* `F1 → Python: Select Interpreter` で `/home/reak1161/repos/research/.venv/bin/python` を選択
* `.vscode/settings.json` を以下の内容に（次章参照）

> 以後、VS Code で新しいターミナルを開くと自動で venv が有効化されます。

---

## 6) 毎回の起動手順（チートシート）

```bash
# CLI 作業時：
source ~/repos/research/.venv/bin/activate

# VS Code を WSL で開く：
cd ~/repos/research/research/卒業研究/code/python_3_10
code .

# VS Code のターミナルが venv を自動有効化（されない場合）
# source ~/repos/research/.venv/bin/activate
```

---

## 7) よくあるつまずき & 即解決

* **`correct default column not specified`**: `--correct-col` の列名ミス。例：CSV が `result` なら `--correct-col result`。
* **`invalid literal for int()`（order_id が日付）**: 自動でユーザー内連番に置換される（最新スクリプト）。手動なら `groupby('user_id').cumcount()`。
* **図保存の `FileNotFoundError`**: 保存先ディレクトリを先に作成（`mkdir -p runs`）。最新は自動作成済み。
* **NumPy 2.x に上がって壊れる**: 必ず `pip install "numpy<2"`。`requirements.txt` に条件を残す。
* **VS Code が WSL と同期しない**: 必ず WSL 側で `code .`。左下が `WSL: Ubuntu-24.04` かチェック。大規模リポは inotify 増量：

  ```bash
  echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
  sudo sysctl -p
  ```

---

## 8) あなたの CSV での動作確認（例）

**忘却オン/オフ比較（AUC/Logloss）**

```bash
python pybkt_user_skill_report.py \
  --csv ../../csv/one_200_50_500_50000.csv \
  --order-col date --user-col user_id --skill-col skill --correct-col result \
  --compare-forgets --auc-by overall
```

**学習状態の可視化（複数“分野”サブプロット）**

```bash
python pybkt_user_skill_report.py \
  --csv ../../csv/one_200_50_500_50000.csv \
  --user 68 --skills 33 7 3 \
  --order-col date --user-col user_id --skill-col skill --correct-col result \
  --display-skill-label "分野" \
  --plot-out runs/u68_domains.png
```

---

# .vscode/settings.json

> リポ直下に `.vscode/settings.json` を作成してください（無ければフォルダごと）。

```json
{
  "python.defaultInterpreterPath": "/home/reak1161/repos/research/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "terminal.integrated.defaultProfile.linux": "bash",
  "files.eol": "\n",
  "files.watcherExclude": {
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/*/**": true
  },
  "editor.formatOnSave": true
}
```

---

## 付録：セットアップ自動化スクリプト（任意）

`bootstrap.sh` として保存→ `bash bootstrap.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1) deps
sudo apt update
sudo apt install -y build-essential git curl zip unzip ca-certificates \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libffi-dev liblzma-dev tk-dev uuid-dev

# 2) pyenv
if ! command -v pyenv >/dev/null 2>&1; then
  curl https://pyenv.run | bash
  if ! grep -q 'pyenv init' ~/.bashrc; then
    cat >> ~/.bashrc <<'EOS'
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOS
  fi
  source ~/.bashrc
fi

# 3) python 3.10.13
pyenv install -s 3.10.13
(
  cd ~/repos/research && pyenv local 3.10.13
)

# 4) venv & pip deps
python -m venv ~/repos/research/.venv
source ~/repos/research/.venv/bin/activate
python -m pip install -U pip
pip install "numpy<2" "pybind11>=2.12" "pyBKT==1.4.1" pandas matplotlib ipykernel
pip freeze | sed 's/^numpy==2\..*/numpy<2/' > ~/repos/research/requirements.txt

# 5) vscode settings
mkdir -p ~/repos/research/.vscode
cat > ~/repos/research/.vscode/settings.json <<'JSON'
{
  "python.defaultInterpreterPath": "/home/reak1161/repos/research/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "terminal.integrated.defaultProfile.linux": "bash",
  "files.eol": "\n"
}
JSON

echo "Done. Open project with: cd ~/repos/research/research/卒業研究/code/python_3_10 && code ."
```
