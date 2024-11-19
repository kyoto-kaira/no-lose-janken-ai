import time
import threading

class GameTimer:
    @staticmethod
    def start(event_start: threading.Event, event_end: threading.Event) -> None:
        time.sleep(5)
        print("-" * 20)
        print("最初はグー")
        print("-" * 20)
        time.sleep(1)
        event_start.set()
        time.sleep(1)
        print("-" * 20)
        print("じゃんけん")
        print("-" * 20)
        time.sleep(1.5)
        print("-" * 20)
        print("ポン")
        print("-" * 20)
        event_end.set()