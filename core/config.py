# config.py
"""
Central configuration file for the video processing application.
"""
import yaml
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yaml')

def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)

_config = load_config()

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
SUBTITLE_FONT_SIZE = _config.get('subtitle_font_size', 28)
SUBTITLE_FONT_COLOR = _config.get('subtitle_font_color', 'FFFFFF')
SUBTITLE_OUTLINE_COLOR = _config.get('subtitle_outline_color', '000000')
SUBTITLE_SHADOW_COLOR = _config.get('subtitle_shadow_color', '000000')

# --- Video Processing Settings ---
CLIP_DURATION_RANGE = (_config.get('clip_duration_min', 45), _config.get('clip_duration_max', 75))
CLIP_VALIDATION_RANGE = (30, 90) # (min, max) seconds for sanitizing clips
SMOOTHING_FACTOR = _config.get('smoothing_factor', 0.05)

# --- LLM Interaction Settings ---
LLM_MAX_RETRIES = _config.get('llm_max_retries', 100)
LLM_MIN_CLIPS_NEEDED = _config.get('llm_min_clips_needed', 1)

# --- Personalization Settings ---
CUSTOM_CLIP_THEMES = _config.get('custom_clip_themes', [])

# --- Encoding Settings ---
VIDEO_ENCODER = "h264_nvenc"  # Options: "h264_nvenc" (NVIDIA GPU), "libx264" (CPU), "hevc_nvenc", "av1_nvenc" etc.

# Path to FFmpeg executable. Set to None to use system's PATH.
# Example: FFMPEG_PATH = "/usr/local/bin/ffmpeg" or "C:/ffmpeg/bin/ffmpeg.exe"
FFMPEG_PATH = "/usr/local/bin/ffmpeg"

FFMPEG_GLOBAL_PARAMS = ['-pix_fmt', 'yuv420p', '-movflags', '+faststart']
FFMPEG_ENCODER_PARAMS = {
    "h264_nvenc": [],
    "hevc_nvenc": ['-preset', 'p5', '-tune', 'hq'],
    "av1_nvenc": ['-preset', 'p5', '-tune', 'hq'],
    # Add other encoders if needed, e.g., "libx264": ['-preset', 'medium', '-crf', '23']
}

