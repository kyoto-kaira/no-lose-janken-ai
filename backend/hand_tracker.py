from typing import Optional

import cv2
import numpy as np
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.hands import HandLandmark, Hands
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS


class HandTracker:
    """
    Mediapipeを使用して手のランドマークを検出および描画するクラス。

    このクラスは、画像を入力として手のランドマークを検出し、
    検出結果をもとにランドマークを画像に描画する機能を提供します。
    """

    def __init__(self) -> None:
        """
        HandTrackerクラスの初期化メソッド。

        MediapipeのHandsモジュールを初期化し、最大検出する手の数や
        検出・追跡信頼度のパラメータを設定します。
        """
        self.hands = Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_frame(self, image: np.ndarray) -> Optional[HandLandmark | None]:
        """
        入力画像から手のランドマークを検出します。

        MediapipeのHandsモジュールを使用して画像から手のランドマークを抽出します。

        Args:
            image (np.ndarray): 入力画像（BGR形式）

        Returns:
            Optional[HandLandmark | None]: 検出された手のランドマーク。
            検出できなかった場合はNoneを返します。
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks
        return None

    def draw_landmarks(self, image: np.ndarray, multi_hand_landmarks: Optional[HandLandmark | None]) -> None:
        """
        入力画像に検出された手のランドマークを描画します。

        Mediapipeの描画ユーティリティを使用して、手のランドマークを画像上に描画します。

        Args:
            image (np.ndarray): 入力画像（BGR形式）
            multi_hand_landmarks (Optional[HandLandmark | None]): 検出された手のランドマーク

        Returns:
            None
        """
        if multi_hand_landmarks:
            for landmarks in multi_hand_landmarks:
                draw_landmarks(image, landmarks, HAND_CONNECTIONS)
