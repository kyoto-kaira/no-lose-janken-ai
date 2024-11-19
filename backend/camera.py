from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np


def find_camera() -> Optional[int]:
    """
    使用可能なカメラデバイスのIDを検索する。
    """
    for i in range(0, 20):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None


class HandTracker:
    """
    Mediapipe を使用した手の検出クラス。
    """

    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing_utils = mp.solutions.drawing_utils

    def process_frame(self, image: np.ndarray) -> Optional[List | None]:
        """
        フレームから手のランドマークを検出する。
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks  # type: ignore
        return None
