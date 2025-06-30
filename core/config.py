# config.py
"""
Central configuration file for the video processing application.
"""

# --- Directory Settings ---
TEMP_DIR = "temp_processing"
STATE_FILE = ".temp_clips.json"
OUTPUT_DIR = "videos"
CLIP_PREFIX = "clip"

# --- Model Settings ---
WHISPER_MODEL = "base"
LLM_MODEL = "qwen2.5-coder:7b"

# --- Subtitle Settings ---
SUBTITLE_FONT_FAMILIES = ['Arial-Bold', 'Helvetica-Bold', 'Georgia-Bold']

# --- Video Processing Settings ---
CLIP_DURATION_RANGE = (45, 75) # (min, max) seconds for LLM selection
CLIP_VALIDATION_RANGE = (30, 90) # (min, max) seconds for sanitizing clips
SMOOTHING_FACTOR = 0.05 # Lower values mean more smoothing (0.0 to 1.0)

# --- LLM Interaction Settings ---
LLM_MAX_RETRIES = 100
LLM_MIN_CLIPS_NEEDED = 1

# --- Encoding Settings ---
VIDEO_ENCODER = "h264_nvenc"  # Options: "h264_nvenc" (NVIDIA GPU), "libx264" (CPU), "hevc_nvenc", "av1_nvenc" etc.

# Path to FFmpeg executable. Set to None to use system's PATH.
# Example: FFMPEG_PATH = "/usr/local/bin/ffmpeg" or "C:/ffmpeg/bin/ffmpeg.exe"
FFMPEG_PATH = "/usr/bin/ffmpeg"

FFMPEG_GLOBAL_PARAMS = ['-pix_fmt', 'yuv420p', '-movflags', '+faststart']
FFMPEG_ENCODER_PARAMS = {
    "h264_nvenc": [],
    "hevc_nvenc": ['-preset', 'p5', '-tune', 'hq'],
    "av1_nvenc": ['-preset', 'p5', '-tune', 'hq'],
    # Add other encoders if needed, e.g., "libx264": ['-preset', 'medium', '-crf', '23']
}

