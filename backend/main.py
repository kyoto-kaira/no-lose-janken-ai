import torch
import keyboard
import threading
from .model import LSTMNet
from .game import JankenGame
from .timer import GameTimer

# モデルとデバイスの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(63, 50, 100, 3).to(device)
model.load_state_dict(torch.load("lstm3.pth", map_location=device))
model.eval()

# ジャンケンゲームの初期化
game = JankenGame(model, device)

# メインループ
while True:
    if keyboard.is_pressed("s"):
        event_start = threading.Event()
        event_end = threading.Event()

        worker = threading.Thread(target=game.play, args=(event_start, event_end))
        trigger = threading.Thread(target=GameTimer.start, args=(event_start, event_end))

        worker.start()
        trigger.start()

        worker.join()
        trigger.join()

    if keyboard.is_pressed("q"):
        break