# utils.py
"""
Utility functions for system checks and video analysis.
"""
import os
import json
import subprocess
import shutil
import platform

def get_video_info(video_path: str) -> dict:
    """Get video information using ffprobe."""
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'duration': float(data['format']['duration']),
            'fps': eval(video_stream.get('r_frame_rate', '0/1')),
            'codec': video_stream['codec_name'],
        }
    except (subprocess.CalledProcessError, StopIteration, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to probe video info for {video_path}: {e}")

def system_checks():
    """Performs and prints results of system checks like disk space and GPU."""
    # Check disk space
    try:
        free_space = shutil.disk_usage('.').free / (1024**3)
        print(f"💾 Available disk space: {free_space:.1f}GB")
        if free_space < 5:
            print("⚠️  Warning: Low disk space (< 5GB).")
    except Exception as e:
        print(f"Could not check disk space: {e}")

    # Check for FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, text=True)
        print("✅ FFmpeg is installed and accessible.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ CRITICAL: FFmpeg not found. Please install it and ensure it's in your system's PATH.")
        
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("🎮 NVIDIA GPU detected. NVENC encoding will be attempted.")
        else:
            print("ℹ️  No NVIDIA GPU detected - will use CPU encoding.")
    except FileNotFoundError:
        print("ℹ️  `nvidia-smi` not found. Assuming no NVIDIA GPU. Will use CPU encoding.")

def print_system_info():
    """Prints system information for debugging purposes."""
    print(f"\n🖥️  System info: {platform.system()} {platform.release()}")
    print(f"   Python version: {platform.python_version()}")
