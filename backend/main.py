import threading

import torch

from .game import JankenGame
from .model import LSTMNet
from .timer import GameTimer

MODEL_PATH = "backend/data/lstm.pth"
# モデルとデバイスの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(63, 50, 100, 3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# ジャンケンゲームの初期化
game = JankenGame(model, device)

# メインループ
while True:
    pressed = input("Press 's' to start the game or 'q' to quit: ")
    if pressed == "s":
        event_start = threading.Event()
        event_end = threading.Event()

        worker = threading.Thread(target=game.play, args=(event_start, event_end))
        trigger = threading.Thread(target=GameTimer.start, args=(event_start, event_end))

        worker.start()
        trigger.start()

        worker.join()
        trigger.join()

    if pressed == "q":
        break
