import asyncio

from streamlit_webrtc.webrtc import WebRtcMode
import streamlit as st
import torch
from backend.const import JANKEN_LABELS, MODEL_PATH
from backend.hand_tracker import HandTracker
from backend.model import LSTMNet
import queue
from streamlit_webrtc import webrtc_streamer
from av import VideoFrame
# モデルの初期設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(63, 50, 100, 3)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
model.eval()
hc = None

# Streamlit UI
st.title("リアルタイムじゃんけんゲーム")

stop_event = asyncio.Event()
start_event = asyncio.Event()
start_event.clear()
# Mediapipe の Hands モジュールの初期化
hand_tracker = HandTracker()
softmax = torch.nn.Softmax(dim=2)
def callback(frame:VideoFrame) -> None:
    """
    WebRTC のビデオフレームを受け取り、手のランドマークを検出して描画します。
    """
    global hc
    #ジャンケンが始まっていないなら、フレームをそのまま返す
    if not start_event.is_set():
        return frame
    image=frame.to_ndarray(format="rgb24")
    image.flags.writeable = False  # 画像への書き込みを禁止して高速化
    multi_hand_landmarks = hand_tracker.process_frame(image)
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
        data = torch.tensor(landmarks).reshape(1, 1, 63).to(device)
        with torch.no_grad():
            y_preds, hc = model(data, hc)
            st.session_state.y_probs.put(softmax(y_preds).tolist()[0][0])
            print(f"=========（画像処理）============={st.session_state.y_probs}====================")
        # 画像を再度書き込み可能にして戻す
        image.flags.writeable = True

        # 手のランドマークが検出された場合、描画する
        hand_tracker.draw_landmarks(image, multi_hand_landmarks)

    return VideoFrame.from_ndarray(image, format="rgb24")

async def janken_game() -> None:
    while True:
        if not start_event.is_set() and not st.session_state["wrote_janken_result"]:
            y_probs=st.session_state.y_probs.get()
            for i, y_prob in enumerate(y_probs):
                st.write(f"{JANKEN_LABELS[i]}:{y_prob}")
            st.session_state["wrote_janken_result"] = True
        if stop_event.is_set() and not st.session_state["set_finish_button"]:
            st.session_state["set_finish_button"] = True
            if st.button("ゲーム終了"):
                stop_event.clear()
                st.session_state["wrote_janken_result"] = False
                st.session_state["set_finish_button"] = False
                break
        # whileを円滑に処理するためのawait
        await asyncio.sleep(0.01)
    start_event.clear()


async def timer() -> None:
    placeholder = st.empty()  # 更新可能なプレースホルダーを作成

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


async def main_game() -> None:
    janken = asyncio.create_task(janken_game())
    time = asyncio.create_task(timer())
    st.session_state.wrote_janken_result = False
    st.session_state.set_finish_button = False
    await asyncio.gather(janken, time)


async def main() -> None:
    if st.session_state.pushet_start_button:
        start_event.set()
        stop_event.clear()
        st.session_state.pushet_start_button = False
        st.session_state.y_probs=queue.Queue()
        asyncio.run(main_game())
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
    if "y_probs" not in st.session_state:
        st.session_state.y_probs = queue.Queue()
    camera=webrtc_streamer(key="example",mode=WebRtcMode.SENDRECV,video_frame_callback=callback,media_stream_constraints={"video": True, "audio": False},async_processing=True)
    asyncio.run(main())