# config.py
"""
Central configuration file for the video processing application.
"""

# --- Directory Settings ---
TEMP_DIR = ".temp"
OUTPUT_DIR = "videos"
CLIP_PREFIX = "clip"

# --- Model Settings ---
WHISPER_MODEL = "large-v3"
LLM_MODEL = "qwen2.5-coder:7b"

# --- Subtitle Settings ---
SUBTITLE_MAX_CHARS_PER_LINE = 35
SUBTITLE_MAX_LINES = 2

# --- Video Processing Settings ---
CLIP_DURATION_RANGE = (45, 75) # (min, max) seconds for LLM selection
CLIP_VALIDATION_RANGE = (30, 90) # (min, max) seconds for sanitizing clips
FACE_DETECTION_SAMPLES = 3 # Number of frames to sample for face detection
# --- Object detection settings ---
OBJECT_DETECTION_CONFIDENCE = 0.7  #
