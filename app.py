import asyncio

import cv2
import pandas as pd
import streamlit as st
import torch
from streamlit.delta_generator import DeltaGenerator

from backend.const import JANKEN_LABELS, MODEL_PATH
from backend.hand_tracker import HandTracker
from backend.model import LSTMNet

# モデルの初期設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(63, 50, 100, 3)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
model.eval()

# Streamlit UI
st.title("リアルタイムじゃんけんゲーム")

stop_event = asyncio.Event()
start_event = asyncio.Event()
start_event.clear()
# Mediapipe の Hands モジュールの初期化
hand_tracker = HandTracker()
softmax = torch.nn.Softmax(dim=2)


async def janken_game(frame_placeholder: DeltaGenerator) -> None:
    hc = None
    cap = cv2.VideoCapture(0)
    result_placeholder = st.empty()
    graph_placeholder = st.empty()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            st.write("カメラのフレームを取得できませんでした。")
            break

        # BGR から RGB に変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # 画像への書き込みを禁止して高速化

        # 手の骨格を推定
        multi_hand_landmarks = hand_tracker.process_frame(image)

        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                # ランドマーク座標をリストに変換
                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

            # データを予測
            data = torch.tensor(landmarks).reshape(1, 1, 63).to(device)
            with torch.no_grad():
                y_preds, hc = model(data, hc)
            # if not start_event.is_set() and not st.session_state["wrote_janken_result"]:
                y_probs = softmax(y_preds).tolist()[0][0]
                print(y_probs)
                col1, col2, col3 = result_placeholder.columns(3)
                col1.metric("グー", f"{y_probs[0] * 100:.2f}%")
                col2.metric("チョキ", f"{y_probs[1] * 100:.2f}%")
                col3.metric("パー", f"{y_probs[2] * 100:.2f}%")

                df = pd.DataFrame(y_probs, index=JANKEN_LABELS.values(), columns=["確率"])
                graph_placeholder.bar_chart(df, height=300)

                st.session_state["wrote_janken_result"] = True

        # 画像を再度書き込み可能にして BGR に戻す
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 手のランドマークが検出された場合、描画する
        hand_tracker.draw_landmarks(image, multi_hand_landmarks)

        # 結果の画像を表示
        frame_placeholder.image(image, channels="BGR")

        if stop_event.is_set() and not st.session_state["set_finish_button"]:
            st.session_state["set_finish_button"] = True
            if st.button("ゲーム終了"):
                stop_event.clear()
                st.session_state["wrote_janken_result"] = False
                st.session_state["set_finish_button"] = False
                break
        # whileを円滑に処理するためのawait
        await asyncio.sleep(0.005)

    cap.release()
    cv2.destroyAllWindows()
    start_event.clear()


async def timer(placeholder: DeltaGenerator) -> None:

    # 最初の状態
    await asyncio.sleep(3)
    placeholder.write("最初はグー")
    await asyncio.sleep(2)

    # 次の状態
    placeholder.write("最初はグー・じゃんけん")
    await asyncio.sleep(2)

    # 最終状態
    placeholder.write("最初はグー・じゃんけん・ポン")
    await asyncio.sleep(0.5)

    start_event.clear()
    stop_event.set()


async def main_game(frame_placeholder: DeltaGenerator, timing_placeholder: DeltaGenerator) -> None:
    janken = asyncio.create_task(janken_game(frame_placeholder))
    time = asyncio.create_task(timer(timing_placeholder))
    st.session_state.wrote_janken_result = False
    st.session_state.set_finish_button = False
    await asyncio.gather(janken, time)


async def main() -> None:
    if st.session_state.pushet_start_button:
        start_event.set()
        stop_event.clear()
        frame_placeholder = st.empty()
        timing_placeholder = st.empty()
        st.session_state.pushet_start_button = False
        await asyncio.gather(main_game(frame_placeholder, timing_placeholder))
        st.rerun()
    else:
        if st.button("ゲーム開始"):
            st.session_state.pushet_start_button = True
            st.rerun()


if __name__ == "__main__":
    if "pushet_start_button" not in st.session_state:
        st.session_state.pushet_start_button = False
    if "wrote_janken_result" not in st.session_state:
        st.session_state.wrote_janken_result = False
    if "set_finish_button" not in st.session_state:
        st.session_state.set_finish_button = False
    asyncio.run(main())
