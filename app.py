import streamlit as st
import cv2
import numpy as np
from threading import Event
from backend.game import JankenGame  # JankenGame をインポート
import torch
from backend.model import LSTMNet

# ジャンケンゲームのモデルのパス
MODEL_PATH = "backend/data/lstm.pth"

# モデルとデバイスの初期化（グローバルに配置）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(63, 50, 100, 3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# JankenGame 初期化
janken_game = JankenGame(model, device)

# Streamlit UI
st.title("リアルタイムじゃんけんゲーム")

start_event = Event()
stop_event = Event()

# ゲーム開始ボタン
if st.button("ゲーム開始"):
    start_event.set()
    stop_event.clear()
    st.write("ゲームを開始しました。カメラに手を見せてください！")

# カメラ表示用のプレースホルダー
frame_placeholder = st.empty()

# ジャンケンの結果表示
result_placeholder = st.empty()

# ゲームループ
cap = cv2.VideoCapture(0)
if start_event.is_set():
    # ストップボタンで終了
    if st.button("停止"):
        start_event.clear()
        stop_event.set()
while start_event.is_set() and cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.write("カメラからフレームを取得できませんでした。")
        break

    # 手のランドマークを検出して予測
    multi_hand_landmarks = janken_game.tracker.process_frame(frame)
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            gesture = janken_game.predict(landmarks)
            result_placeholder.write(f"予測された手: {gesture}")

        # ランドマークをフレームに描画
        janken_game.tracker.draw_landmarks(frame, multi_hand_landmarks)

    # フレームを表示
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

cap.release()

# ゲーム終了メッセージ
if stop_event.is_set():
    st.write("ゲームを終了しました。")
