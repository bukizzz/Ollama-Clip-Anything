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
TEMP_DIR = _config.get('temp_dir', "temp_processing")
STATE_FILE = _config.get('state_file', ".temp_clips.json")
OUTPUT_DIR = _config.get('output_dir', "videos")
CLIP_PREFIX = _config.get('clip_prefix', "clip")
B_ROLL_ASSETS_DIR = _config.get('b_roll_assets_dir', "b_roll_assets")

# --- Model Settings ---
WHISPER_MODEL = _config.get('whisper_model', "base")
LLM_MODEL = _config.get('llm_model', "deepseek-coder:6.7b")
IMAGE_RECOGNITION_MODEL = _config.get('image_recognition_model', "qwen2.5vl:7b")

# --- Subtitle Settings ---
SUBTITLE_FONT_FAMILIES = _config.get('subtitle_font_families', ['Impact', 'Arial Black', 'Bebas Neue'])
SUBTITLE_BACKGROUND_COLOR = _config.get('subtitle_background_color', 'FFA500')
SUBTITLE_BORDER_RADIUS = _config.get('subtitle_border_radius', 10)
SUBTITLE_FONT_SIZE = _config.get('subtitle_font_size', 28)
SUBTITLE_FONT_COLOR = _config.get('subtitle_font_color', 'FFFFFF')
SUBTITLE_OUTLINE_COLOR = _config.get('subtitle_outline_color', '000000')
SUBTITLE_SHADOW_COLOR = _config.get('subtitle_shadow_color', '000000')
HIGHLIGHT_FONT_COLOR = _config.get('highlight_font_color', '00FFFF')
HIGHLIGHT_OUTLINE_COLOR = _config.get('highlight_outline_color', '000000')

# --- Video Processing Settings ---
CLIP_DURATION_RANGE = (_config.get('clip_duration_min', 45), _config.get('clip_duration_max', 75))
CLIP_VALIDATION_RANGE = (_config.get('clip_validation_min', 30), _config.get('clip_validation_max', 90))
SMOOTHING_FACTOR = _config.get('smoothing_factor', 0.1)

# --- LLM Interaction Settings ---
LLM_MAX_RETRIES = _config.get('llm_max_retries', 100)
LLM_MIN_CLIPS_NEEDED = _config.get('llm_min_clips_needed', 1)

# --- LLM Configuration from YAML ---
LLM_CONFIG = _config.get('llm', {})

# --- Personalization Settings ---
CUSTOM_CLIP_THEMES = _config.get('custom_clip_themes', [])

# --- Agent Configuration ---
AGENT_CONFIG = _config.get('agents', {})

# --- Encoding Settings ---
VIDEO_ENCODER = _config.get('video_encoder', "h264_nvenc")

# Path to FFmpeg executable. Set to None to use system's PATH.
FFMPEG_PATH = _config.get('ffmpeg_path', "/usr/local/bin/ffmpeg")

FFMPEG_GLOBAL_PARAMS = _config.get('ffmpeg_global_params', ['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
FFMPEG_ENCODER_PARAMS = _config.get('ffmpeg_encoder_params', {
    "h264_nvenc": ['-preset', 'p5', '-tune', 'hq', '-cq', '20'],
    "hevc_nvenc": ['-preset', 'p5', '-tune', 'hq'],
    "av1_nvenc": ['-preset', 'p5', '-tune', 'hq'],
})
