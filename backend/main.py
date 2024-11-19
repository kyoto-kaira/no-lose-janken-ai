import threading

import torch

from .game import JankenGame
from .model import LSTMNet
from .timer import GameTimer

# 定数
MODEL_PATH = "backend/data/lstm.pth"

# モデルとデバイスの初期化（グローバルに配置）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(63, 50, 100, 3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# ジャンケンゲームの初期化（グローバルに配置）
game = JankenGame(model, device)


def main() -> None:
    """
    ジャンケンゲームを開始・管理する関数。

    ゲームの開始や終了、スレッド管理を行い、ユーザーの入力に基づいて
    ゲームの実行フローを制御します。

    Args:
        None

    Returns:
        None
    """
    while True:
        # ゲーム操作の入力受付
        pressed = input("Press 's' to start the game or 'q' to quit: ")
        if pressed == "s":
            # イベントフラグの初期化
            event_start = threading.Event()
            event_end = threading.Event()

            # ジャンケンゲームスレッドの起動
            worker = threading.Thread(target=game.play, args=(event_start, event_end))
            trigger = threading.Thread(target=GameTimer.start, args=(event_start, event_end))

            worker.start()
            trigger.start()

            # スレッドの終了を待機
            worker.join()
            trigger.join()

        elif pressed == "q":
            print("ゲームを終了します。")
            break

        else:
            print("無効な入力です。's' または 'q' を入力してください。")


if __name__ == "__main__":
    main()
