import cv2
import mediapipe as mp
import torch
import time
import threading
import keyboard #推論スタートの入力処理のため
from models import *

#推論結果の値と手の対応
janken_hands = {0 : "グー",
                1 : "チョキ",
                2 : "パー"}
#推論結果に勝てる手
janken_hands_win = {0 : "パー",
                    1 : "グー",
                    2 : "チョキ"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#modelのロード
weight_path = "./model_weights/lstm.pth"
model = LSTM_Net1_pred(63, 50, 100, 3)
#model = GRU_Net1_pred(63, 50, 100, 3)
model.to(device)
model.load_state_dict(torch.load(weight_path, map_location=device))
model.eval()

def janken_game():
    hc = None #隠れ状態の初期化
    landmarks = [0.0]*63 #手が検出されるまでの値
    # MediapipeのHandsモジュールを初期化
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    #カメラの取得
    cap = cv2.VideoCapture(0)

    #初回の計算だけ遅いので、先に計算しておく
    a = (torch.tensor(range(63), dtype=torch.float32)/62).reshape(1, 1, 63).to(device)
    model(a, None)

    with mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            cv2.waitKey(1) #これが無いと描画が上手くいかないが、無くせるならなくていい
            success, image = cap.read()
            if not success:
                print("カメラのフレームを取得できませんでした。")
                break

            #BGRからRGBに変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # 画像への書き込みを禁止して高速化

            # 手の骨格を推定
            results = hands.process(image)
            #推論開始
            if event1.is_set():
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # ランドマーク座標をリストに変換
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append(lm.x)
                            landmarks.append(lm.y)
                            landmarks.append(lm.z)
                    data = torch.tensor(landmarks).reshape(1,1,63).to(device)
                    with torch.no_grad():
                        y_pred, hc = model(data, hc)
                        y_pred = int(y_pred.argmax(2).item())
                        print(janken_hands_win[y_pred]) #出された手に勝つ手を表示
                        #print(janken_hands[y_pred]) #出された手を表示
            
            # 画像を再度書き込み可能にしてBGRに戻す
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 手のランドマークが検出された場合、描画する
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #print(hand_landmarks)
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 結果の画像を表示            
            cv2.imshow('Hand Tracking', image)
            
            #じゃんけんが終わったら推論・描画終了
            if event2.is_set():
                break

    cap.release()
    cv2.destroyAllWindows()

#じゃんけんのタイミング処理
def timer():
    time.sleep(5) #初期化関係を待つ時間、パソコンの性能・状態によっては5秒で足りないこともある
    print("-"*20)
    print("最初はグー")
    print("-"*20)
    time.sleep(1)
    event1.set() #推論開始
    time.sleep(1)
    print("-"*20)
    print("じゃんけん")
    print("-"*20)
    time.sleep(1.5)
    print("-"*20)
    print("ポン")
    print("-"*20)
    event2.set()


while True:
    #じゃんけんを始める
    if keyboard.is_pressed("s"):
        event1 = threading.Event()
        event2 = threading.Event()

        woker = threading.Thread(target=janken_game)
        trigger = threading.Thread(target=timer)

        woker.start()
        trigger.start()

        woker.join()
        trigger.join()
    
    if keyboard.is_pressed("q"):
        break