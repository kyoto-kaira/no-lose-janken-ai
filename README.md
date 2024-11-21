# no-lose-janken-ai

## 動かし方
1. uvを入れる（入っている人はこの手順は飛ばしてください）

```shell
# macOS または Linuxの人
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windowsの人
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. リポジトリをクローンする

```shell
# 任意のディレクトリに移動して以下を実行すると、このリポジトリがその下に作成されます
git clone https://github.com/kyoto-kaira/no-lose-janken-ai.git

# 作成されたリポジトリに移動する
cd no-lose-janken-ai
```

3. Python3.11の仮想環境を作成する

```shell
# 以下を実行すると、現在のディレクトリに.venv/という名前の仮想環境が作成されます
uv venv -p 3.11.10
```

4. 仮想環境を有効化する

```shell
# macOS または Linuxの人
source .venv/bin/activate

# Windowsの人 (バックスラッシュです!!)
.venv\Scripts\activate

# ディレクトリ名の左側に (no-lose-janken-ai) と表示されれば成功です
```

5. 依存パッケージをインストールする

```shell
uv pip install -r requirements.txt
```

6. アプリを起動する

```shell
streamlit run app.py
```
