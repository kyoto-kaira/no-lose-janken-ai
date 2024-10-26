# ベースイメージ
FROM python:3.9

# 作業ディレクトリの設定
WORKDIR /app

# 必要なシステムライブラリをインストール
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get clean

# 仮想環境のディレクトリを指定
ENV VIRTUAL_ENV=/app/venv

# 仮想環境の作成
RUN python3 -m venv $VIRTUAL_ENV

# 仮想環境のPythonとpipをデフォルトに設定
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 必要なパッケージリストをコンテナ内にコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# コードをコンテナにコピー
COPY backend/inference.py .
COPY backend/data/lstm3.pth data/
# コンテナ起動時にapp.pyを実行
CMD ["python", "inference.py"]
