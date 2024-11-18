# ベースイメージ
FROM ubuntu:22.04
# タイムゾーン設定を非対話モードにするための環境変数を設定
ENV DEBIAN_FRONTEND=noninteractive
# パッケージの更新と必要なツールのインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libopencv-dev \
    v4l-utils \
    ffmpeg \
    x11-apps
# タイムゾーンを設定
RUN ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata
# 必要なパッケージリストをコンテナ内にコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# コードをコンテナにコピー
COPY backend/* .

CMD ["python", "main.py"]