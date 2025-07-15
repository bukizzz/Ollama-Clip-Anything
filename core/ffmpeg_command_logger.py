import logging
import subprocess

class FFMPEGCommandLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.command = None

    def info(self, msg, *args, **kwargs):
        if "MoviePy: running " in msg:
            self.command = msg.replace("MoviePy: running ", "").strip()
            print(f"▶️ \033[94mFFMPEG COMMAND: {self.command}\033[0m")
        super().info(msg, *args, **kwargs)

    @staticmethod
    def log_command(command: list, command_name: str):
        """
        Executes an FFmpeg command and logs its stdout and stderr.
        Returns stdout, stderr, and return code.
        """
        logger = logging.getLogger('ffmpeg_commands')
        logger.setLevel(logging.INFO)

        # Ensure a handler is attached for this logger
        if not logger.handlers:
            handler = logging.FileHandler('logs/ffmpeg_commands.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        full_command_str = " ".join(command)
        logger.info(f"Executing {command_name} command: {full_command_str}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate()
        returncode = process.returncode

        if stdout:
            logger.info(f"{command_name} STDOUT:\n{stdout}")
        if stderr:
            logger.error(f"{command_name} STDERR:\n{stderr}")
        
        logger.info(f"{command_name} exited with code: {returncode}")
        
        return stdout, stderr, returncode
