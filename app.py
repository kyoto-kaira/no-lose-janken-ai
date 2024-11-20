import streamlit as st
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import asyncio

class LSTM_Net1_pred(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, tagset_size):
        super(LSTM_Net1_pred, self).__init__()
        self.linear = nn.Linear(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x, hc=None):
        x = self.linear(x)
        if hc == None:
            x, hc = self.lstm(x)
        else:
            x, hc = self.lstm(x, hc)
        x = self.hidden2tag(x)
        return x, hc

janken = {0: "グー",
          1: "チョキ",
          2: "パー"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_Net1_pred(63, 50, 100, 3)
model.to(device)
model.load_state_dict(torch.load("backend/data/lstm.pth", map_location=device,weights_only=True), strict=False)
model.eval()

# Streamlit UI
st.title("リアルタイムじゃんけんゲーム")

stop_event = asyncio.Event()
start_event= asyncio.Event()
# Mediapipe の Hands モジュールの初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

async def janken_game():
    hc = None
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                st.write("カメラのフレームを取得できませんでした。")
                break

            # BGR から RGB に変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # 画像への書き込みを禁止して高速化

            # 手の骨格を推定
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # ランドマーク座標をリストに変換
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                        landmarks.append(lm.z)

                    # データを予測
                    data = torch.tensor(landmarks).reshape(1, 1, 63).to(device)
                    with torch.no_grad():
                        y_pred, hc = model(data, hc)
                        y_pred = int(y_pred.argmax(2).item())
                        if not start_event.is_set() and not st.session_state["wrote_janken_result"]:
                            st.write(f"予測された手: {janken[y_pred]}")
                            st.session_state["wrote_janken_result"]=True

            # 画像を再度書き込み可能にして BGR に戻す
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 手のランドマークが検出された場合、描画する
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 結果の画像を表示
            frame_placeholder.image(image, channels="BGR")

            if stop_event.is_set():
                if not st.session_state["set_finish_button"]:
                    st.session_state["set_finish_button"]=True
                    if st.button("ゲーム終了"):
                        start_event.clear()
                        frame_placeholder.empty()
                        st.write("ゲームを終了します。")
                        break
            #whileを円滑に処理するためのawait
            await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()
    start_event.clear()

async def timer():
    await asyncio.sleep(0.5)
    st.write("-" * 20)
    st.write("最初はグー")
    st.write("-" * 20)
    await asyncio.sleep(1)
    st.write("-" * 20)
    st.write("じゃんけん")
    st.write("-" * 20)
    await asyncio.sleep(1.5)
    st.write("-" * 20)
    st.write("ポン")
    st.write("-" * 20)
    start_event.clear()
    stop_event.set()

async def main():
    janken=asyncio.create_task(janken_game())
    time=asyncio.create_task(timer())
    st.session_state.wrote_janken_result=False
    await asyncio.gather(janken,time)
    
if st.button("ゲーム開始"):
    start_event.set()
    stop_event.clear()
    frame_placeholder = st.empty()
    if "wrote_janken_result" not in st.session_state:
        st.session_state.wrote_janken_result =False
    if "set_finish_button" not in st.session_state:
        st.session_state.set_finish_button=False
    asyncio.run(main())

