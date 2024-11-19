import cv2
import torch
from threading import Event
from .camera import HandTracker, find_camera
from .model import LSTMModel
from typing import List
JANKEN_LABELS = {0: "グー", 1: "チョキ", 2: "パー"}


def predict_hand_gesture(landmarks: List[float], model: LSTMModel, device: torch.device, hc=None) -> str:
    """
    手のランドマークを使用してジャンケンの手を予測する。
    """
    data = torch.tensor(landmarks, dtype=torch.float32).reshape(1, 1, 63).to(device)
    with torch.no_grad():
        y_pred, hc = model(data, hc)
        gesture = JANKEN_LABELS[int(y_pred.argmax(2).item())]
        return gesture


def janken_game(event_start: Event, event_end: Event, model: LSTMModel, device: torch.device) -> None:
    """
    カメラ上での手の動きからジャンケンの手を予測する関数
    """
    hc = None
    tracker = HandTracker()
    camera_id = find_camera()

    if camera_id is None:
        print("カメラが見つかりませんでした。")
        return

    cap = cv2.VideoCapture(camera_id)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("カメラのフレームを取得できませんでした。")
            break

        if event_start.is_set():
            landmarks_list = tracker.process_frame(frame)
            if landmarks_list:
                landmarks = [
                    lm.x for lm in landmarks_list[0].landmark
                ] + [
                    lm.y for lm in landmarks_list[0].landmark
                ] + [
                    lm.z for lm in landmarks_list[0].landmark
                ]
                gesture = predict_hand_gesture(landmarks, model, device, hc)
                print(f"予測: {gesture}")

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or event_end.is_set():
            break

    cap.release()
    cv2.destroyAllWindows()