import os
import json
import subprocess
import shutil
import platform
import ollama
import psutil
from core.temp_manager import get_temp_path
from core.config import LLM_MODEL, FFMPEG_PATH

def convert_av1_to_hevc(video_path: str) -> str:
    """Converts an AV1 video to H.265 (HEVC) using FFmpeg."""
    output_path = get_temp_path(f"converted_{os.path.basename(video_path)}")
    
    # Check for NVIDIA GPU to decide encoder
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        if result.returncode == 0:
            encoder = "hevc_nvenc"
            print("  Using hevc_nvenc (NVIDIA GPU) for AV1 conversion.")
        else:
            encoder = "libx265"
            print("Using libx265 (CPU) for AV1 conversion.")
    except FileNotFoundError:
        encoder = "libx265"
        print("nvidia-smi not found. Using libx265 (CPU) for AV1 conversion.")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-c:v", encoder, "-preset", "medium", "-crf", "23",
        "-c:a", "copy", output_path
    ]
    print(f"Running FFmpeg to convert AV1 to HEVC: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully converted {video_path} to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed. Stderr: {e.stderr}")
        raise RuntimeError(f"Failed to convert AV1 video: {e}")

def terminate_existing_processes():
    """Terminates any other running instances of the current script."""
    current_pid = os.getpid()
    
    # Find the actual main.py script name
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline'] and 'main.py' in ' '.join(proc.info['cmdline']):
                if proc.info['pid'] != current_pid:
                    print(f"Found existing main.py process (PID: {proc.info['pid']}). Terminating...")
                    proc.terminate()
                    proc.wait(timeout=5) # Wait for process to terminate
                    if proc.is_running():
                        print(f"Process {proc.info['pid']} did not terminate gracefully, killing...")
                        proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def get_video_info(video_path: str) -> dict:
    """Get video information using ffprobe."""
    probe_cmd = [
        "/usr/local/bin/ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        if video_stream['codec_name'] == 'av1':
            print("⚠️  AV1 video codec detected. Attempting to convert to H.265 (HEVC)...")
            converted_video_path = convert_av1_to_hevc(video_path)
            # Update video_path to the converted video for further processing
            video_path = converted_video_path
            # Re-probe the converted video to get its info
            probe_cmd = [
                "/snap/ffmpeg/current/usr/bin/ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        video_info = {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'duration': float(data['format']['duration']),
            'fps': eval(video_stream.get('r_frame_rate', '0/1')),
            'codec': video_stream['codec_name'],
        }
        return video_info, video_path
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
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is installed and accessible.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ CRITICAL: FFmpeg not found. Please install it and ensure it's in your system's PATH.")
        
    # Check for Ollama service
    try:
        client = ollama.Client(host='http://localhost:11434')
        client.list() # This will raise an exception if Ollama is not running
        print("✅ Ollama service is running.")
    except Exception as e:
        print(f"❌ CRITICAL: Ollama service not found or not reachable at http://localhost:11434. Please ensure Ollama is installed and running. Error: {e}")
        return # Exit if Ollama service is not running

    # Check for LLM model availability
    try:
        client = ollama.Client(host='http://localhost:11434')
        models = client.list()['models']
        if any(model['name'] == LLM_MODEL for model in models):
            print(f"✅ LLM model '{LLM_MODEL}' is available.")
        else:
            print(f"❌ CRITICAL: LLM model '{LLM_MODEL}' not found. Please download it using 'ollama pull {LLM_MODEL}'.")
    except Exception as e:
        print(f"❌ CRITICAL: Could not check LLM model availability. Error: {e}")

    # Check for PyTorch
    try:
        import torch
        print(f"✅ PyTorch is installed (version: {torch.__version__}).")
        if torch.cuda.is_available():
            print(f"  CUDA is available (version: {torch.version.cuda}).")
        else:
            print("  CUDA is NOT available. Processing will use CPU.")
    except ImportError:
        print("❌ CRITICAL: PyTorch is not installed. Please install it (e.g., pip install torch torchvision torchaudio).")
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")

    # Check for OpenCV
    try:
        import cv2
        print(f"✅ OpenCV is installed (version: {cv2.__version__}).")
    except ImportError:
        print("❌ CRITICAL: OpenCV (opencv-python) is not installed. Please install it (e.g., pip install opencv-python).")
    except Exception as e:
        print(f"❌ Error checking OpenCV: {e}")

    # Check for MediaPipe
    try:
        import mediapipe
        print(f"✅ MediaPipe is installed (version: {mediapipe.__version__}).")
    except ImportError:
        print("❌ CRITICAL: MediaPipe is not installed. Please install it (e.g., pip install mediapipe).")
    except Exception as e:
        print(f"❌ Error checking MediaPipe: {e}")

    # Check for spaCy model
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✅ spaCy 'en_core_web_sm' model is loaded.")
    except OSError:
        print("❌ CRITICAL: spaCy 'en_core_web_sm' model not found. Please download it (python -m spacy download en_core_web_sm).")
    except ImportError:
        print("❌ CRITICAL: spaCy is not installed. Please install it (e.g., pip install spacy).")
    except Exception as e:
        print(f"❌ Error checking spaCy: {e}")

def print_system_info():
    """Prints detailed system information for debugging purposes."""
    print("\n🖥️  System Information:")
    print(f"   Operating System: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python Version: {platform.python_version()} ({platform.python_compiler()})")
    print(f"   Processor: {platform.processor()}")
    try:
        import psutil
        print(f"   Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    except ImportError:
        print("   (Install 'psutil' for more RAM info)")
    except Exception as e:
        print(f"   Error getting RAM info: {e}")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   GPU (CUDA): {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("   GPU: None (CUDA not available)")
    except ImportError:
        pass # Already handled in system_checks
    except Exception as e:
        print(f"   Error checking GPU: {e}")
