import cv2
import mediapipe as mp
import torch
from typing import Optional

class HandTracker:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_frame(self, image: torch.Tensor) -> Optional[mp.solutions.hands.HandLandmark]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        return results.multi_hand_landmarks

    def draw_landmarks(self, image: torch.Tensor, hand_landmarks) -> None:
        if hand_landmarks:
            for landmarks in hand_landmarks:
                self.mp_drawing.draw_landmarks(image, landmarks, self.mp_hands.HAND_CONNECTIONS)