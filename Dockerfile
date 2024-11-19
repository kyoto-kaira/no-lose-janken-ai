# ベースイメージ
FROM python:3.10-slim

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libopencv-dev \
    ffmpeg \
    v4l-utils \
    gcc \
    g++ \
    make \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 必要なPythonライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# コードをコンテナにコピー
COPY backend/* .

CMD ["python", "main.py"]