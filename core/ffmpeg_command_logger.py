import logging

class FFMPEGCommandLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.command = None

    def info(self, msg, *args, **kwargs):
        if "MoviePy: running " in msg:
            self.command = msg.replace("MoviePy: running ", "").strip()
            print(f"FFMPEG COMMAND: {self.command}")
        super().info(msg, *args, **kwargs)
