import cv2
import mediapipe as mp
from typing import Optional


def find_camera() -> Optional[int]:
    """
    使用可能なカメラデバイスのIDを検索する。
    """
    for i in range(0, 20):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    return None


class HandTracker:
    """
    Mediapipe を使用した手の検出クラス。
    """

    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing_utils = mp.solutions.drawing_utils

    def process_frame(self, image) -> Optional[list]:
        """
        フレームから手のランドマークを検出する。
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks
        return None