import threading
import time


class GameTimer:
    """
    ゲームのタイマーを管理するクラス

    ゲームの開始と終了のタイミングを制御します。
    """

    @staticmethod
    def start(event_start: threading.Event, event_end: threading.Event) -> None:
        """
        ゲームのタイミングを管理し、開始および終了のイベントを通知します。

        指定したタイミングで「最初はグー」、「じゃんけん」、「ポン」のメッセージを表示し、
        各タイミングに対応したイベントをセットします。

        Args:
            event_start (threading.Event): ゲームの開始タイミングを通知するイベントフラグ
            event_end (threading.Event): ゲームの終了タイミングを通知するイベントフラグ

        Returns:
            None
        """
        # ゲーム開始前の猶予時間
        time.sleep(5)
        print("-" * 20)
        print("最初はグー")  # ジャンケンの掛け声その1
        print("-" * 20)

        # 少し待機してゲーム開始の通知
        time.sleep(1)
        event_start.set()  # ゲーム開始フラグをセット

        # さらに少し待機して「じゃんけん」を表示
        time.sleep(1)
        print("-" * 20)
        print("じゃんけん")  # ジャンケンの掛け声その2
        print("-" * 20)

        # 「ポン」を表示してゲーム終了の通知
        time.sleep(1.5)
        print("-" * 20)
        print("ポン")  # ジャンケンの掛け声その3
        print("-" * 20)
        event_end.set()  # ゲーム終了フラグをセット
