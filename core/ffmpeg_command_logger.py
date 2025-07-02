import logging

class FFMPEGCommandLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.command = None

    def info(self, msg, *args, **kwargs):
        if "MoviePy: running " in msg:
            self.command = msg.replace("MoviePy: running ", "").strip()
            print(f"▶️ \033[94mFFMPEG COMMAND: {self.command}\033[0m")
        super().info(msg, *args, **kwargs)
