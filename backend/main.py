import torch
import threading
import time
from .game import janken_game
from .model import load_model


def start_timer(event_start: threading.Event, event_end: threading.Event) -> None:
    """
    ジャンケンの合図を出すためのタイマー関数。
    """
    time.sleep(5)
    print("最初はグー")
    time.sleep(1)
    event_start.set()
    time.sleep(1)
    print("じゃんけん")
    time.sleep(1.5)
    print("ポン")
    event_end.set()


def main():
    """
    ゲームを始めるための関数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    while True:
        user_input = input("Press 's' to start the game or 'q' to quit: ")
        if user_input.lower() == 's':
            event_start = threading.Event()
            event_end = threading.Event()

            game_thread = threading.Thread(target=janken_game, args=(event_start, event_end, model, device))
            timer_thread = threading.Thread(target=start_timer, args=(event_start, event_end))

            game_thread.start()
            timer_thread.start()

            game_thread.join()
            timer_thread.join()

        elif user_input.lower() == 'q':
            print("終了します。")
            break


if __name__ == "__main__":
    main()
