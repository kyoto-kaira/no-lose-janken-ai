import torch
import cv2
from lstm_net import LSTMNet
from hand_tracker import HandTracker
from typing import Optional

class JankenGame:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.tracker = HandTracker()
        self.janken_labels = {0: "グー", 1: "チョキ", 2: "パー"}
        self.hc = None

    def predict(self, landmarks: list) -> str:
        data = torch.tensor(landmarks, dtype=torch.float32).reshape(1, 1, 63).to(self.device)
        with torch.no_grad():
            y_pred, self.hc = self.model(data, self.hc)
            return self.janken_labels[int(y_pred.argmax(2).item())]

    def play(self, event_start, event_end) -> None:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("カメラのフレームを取得できませんでした。")
                break

            hand_landmarks = self.tracker.process_frame(frame)
            if event_start.is_set() and hand_landmarks:
                for hand in hand_landmarks:
                    landmarks = [lm.x for lm in hand.landmark] + [lm.y for lm in hand.landmark] + [lm.z for lm in hand.landmark]
                    gesture = self.predict(landmarks)
                    print(f"予測: {gesture}")

            self.tracker.draw_landmarks(frame, hand_landmarks)
            cv2.imshow('Hand Tracking', frame)

            if event_end.is_set() or cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()