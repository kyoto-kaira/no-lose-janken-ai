import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import time
import threading
import keyboard

DATA_PATH = "data/lstm3.pth"


class LSTM_Net1_pred(nn.Module):
    """
    LSTMモデル(ジャンケンの手を予測)
    """

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
print(device)

model = LSTM_Net1_pred(63, 50, 100, 3)
model.to(device)
model.load_state_dict(torch.load(DATA_PATH, map_location=device, weights_only=True))
model.eval()


def janken_game():
    """
    カメラで手を追跡し、ジャンケンを行う
    """
    hc = None
    landmarks = [0.0] * 63
    # MediapipeのHandsモジュールを初期化
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # 動画ファイルの読み込み（カメラに切り替える場合は、0を指定）
    cap = cv2.VideoCapture(0)  # または 0

    a = (torch.tensor(range(63), dtype=torch.float32) / 62).reshape(1, 1, 63).to(device)
    model(a, None)

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            cv2.waitKey(1)
            success, image = cap.read()
            if not success:
                print("カメラのフレームを取得できませんでした。")
                break

            # BGRからRGBに変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # 画像への書き込みを禁止して高速化

            # 手の骨格を推定

            results = hands.process(image)
            if event1.is_set():
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # ランドマーク座標をリストに変換
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append(lm.x)
                            landmarks.append(lm.y)
                            landmarks.append(lm.z)

                    # data = torch.tensor(landmarks).reshape(-1, 3).T.reshape(1,1,63).to(device)
                    data = torch.tensor(landmarks).reshape(1, 1, 63).to(device)
                    # print(data)
                    with torch.no_grad():  # 勾配計算しなくて良い
                        y_pred, hc = model(data, hc)
                        y_pred = int(y_pred.argmax(2).item())
                        print(janken[y_pred])

            # 画像を再度書き込み可能にしてBGRに戻す
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 手のランドマークが検出された場合、描画する
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # print(hand_landmarks)
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 結果の画像を表示

            cv2.imshow('Hand Tracking', image)

            if event2.is_set():
                break

    cap.release()
    cv2.destroyAllWindows()


def timer():
    """
    ジャンケンの合図を出す
    """
    time.sleep(5)
    print("-" * 20)
    print("最初はグー")
    print("-" * 20)
    time.sleep(1)
    event1.set()
    time.sleep(1)
    print("-" * 20)
    print("じゃんけん")
    print("-" * 20)
    time.sleep(1.5)
    print("-" * 20)
    print("ポン")
    print("-" * 20)
    event2.set()


while True:
    user_input = input("Press 's' to start the game or 'q' to quit: ")
    if user_input == "s":
        event1 = threading.Event()
        event2 = threading.Event()

        woker = threading.Thread(target=janken_game)
        trigger = threading.Thread(target=timer)

        woker.start()
        trigger.start()

        woker.join()
        trigger.join()

    elif user_input == "q":
        break
