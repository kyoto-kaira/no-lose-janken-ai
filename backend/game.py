from threading import Event

import cv2
import torch
from torch import nn

from .hand_tracker import HandTracker

JANKEN_LABELS = {0: "グー", 1: "チョキ", 2: "パー"}


class JankenGame:
    """
    ジャンケンゲームを管理するクラス

    手のランドマークを検出し、モデルを用いてジャンケンの手（グー、チョキ、パー）を予測します。
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """
        JankenGameクラスの初期化メソッド

        Args:
            model (nn.Module): ジャンケンの予測に使用するPyTorchモデル
            device (torch.device): モデルの実行デバイス（CPUまたはGPU）
        """
        self.model = model  # 予測モデル
        self.device = device  # 実行デバイス
        self.tracker = HandTracker()  # 手のランドマーク検出器
        self.hc = None  # LSTMの隠れ状態を保持

    def predict(self, landmarks: list[float]) -> str:
        """
        手のランドマーク情報からジャンケンの手を予測します。

        Args:
            landmarks (list): 検出された手のランドマーク情報（x, y, z座標）

        Returns:
            str: 予測されたジャンケンの手（"グー", "チョキ", "パー"）
        """
        # 入力データをテンソルに変換し、モデルで予測
        data = torch.tensor(landmarks, dtype=torch.float32).reshape(1, 1, 63).to(self.device)
        with torch.no_grad():  # 勾配計算を無効化
            y_pred, self.hc = self.model(data, self.hc)  # モデルによる予測とLSTMの隠れ状態の更新
            return JANKEN_LABELS[int(y_pred.argmax(2).item())]  # 最大値のインデックスをラベルに変換

    def play(self, event_start: Event, event_end: Event) -> None:
        """
        カメラを使用してジャンケンをリアルタイムでプレイします。

        手のランドマークを検出し、モデルでジャンケンの手を予測します。
        ゲームは、イベントフラグによって開始および終了が制御されます。

        Args:
            event_start (Event): ゲーム開始フラグ
            event_end (Event): ゲーム終了フラグ
        """
        cap = cv2.VideoCapture(0)  # カメラを起動
        while cap.isOpened():  # カメラが正常に動作している間ループ
            cv2.waitKey(1)
            success, frame = cap.read()  # フレームをキャプチャ
            if not success:  # フレームが取得できない場合
                print("カメラのフレームを取得できませんでした。")
                break

            # 手のランドマークを検出
            multi_hand_landmarks = self.tracker.process_frame(frame)
            if event_start.is_set() and multi_hand_landmarks:  # ゲームが開始されており、ランドマークが検出された場合
                for hand_landmarks in multi_hand_landmarks:
                    landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                gesture = self.predict(landmarks)
                print(f"予測: {gesture}")
            # 手のランドマークをフレームに描画
            if multi_hand_landmarks:
                self.tracker.draw_landmarks(frame, multi_hand_landmarks)
                cv2.imshow("Hand Tracking", frame)  # フレームを表示

            # 終了フラグまたは「q」キーが押された場合、ループを終了
            if event_end.is_set():
                break

        cap.release()  # カメラリソースを解放
        cv2.destroyAllWindows()  # ウィンドウを閉じる
