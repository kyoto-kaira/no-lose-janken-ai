import threading

import keyboard
import torch

from .const import MODEL_PATH
from .game import JankenGame
from .model import LSTMNet
from .timer import GameTimer

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
        print("Press 's' to start the game or 'q' to quit: ")
        pressed = keyboard.read_event()
        if pressed.name == "s":
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

        elif pressed.name == "q":
            print("ゲームを終了します。")
            break


if __name__ == "__main__":
    main()
