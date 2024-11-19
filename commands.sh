#!/bin/bash

# イメージ名
IMAGE_NAME="opencv_camera_app"

# コンテナ名
CONTAINER_NAME="opencv_camera_container"

# Dockerイメージのビルド
docker build -t $IMAGE_NAME .

# コンテナを削除（もし既に存在している場合）
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# コンテナの起動（カメラデバイスをマウント）
docker run --rm \
    --name $CONTAINER_NAME \
    --device /dev/video0:/dev/video0 \
    $IMAGE_NAME