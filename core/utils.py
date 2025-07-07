import os
import json
import subprocess
import shutil
import platform
import ollama
import psutil
import requests # Import requests for Gemini API check
from core.temp_manager import get_temp_path
from core.config import config

def convert_av1_to_hevc(video_path: str) -> str:
    """Converts an AV1 video to H.265 (HEVC) using FFmpeg."""
    output_path = get_temp_path(f"converted_{os.path.basename(video_path)}")
    
    # Check for NVIDIA GPU to decide encoder
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        if result.returncode == 0:
            encoder = "hevc_nvenc"
            print("⚙️ Using NVENC (NVIDIA GPU) for AV1 conversion.")
        else:
            encoder = "libx265"
            print("⚙️ Using CPU for AV1 conversion.")
    except FileNotFoundError:
        encoder = "libx265"
        print("⚠️ NVENC not found. Using CPU for AV1 conversion.")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-c:v", encoder, "-preset", "medium", "-crf", "23",
        "-c:a", "copy", output_path
    ]
    print(f"⚙️ Running FFmpeg to convert AV1 to HEVC: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Successfully converted {video_path} to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed. Stderr: {e.stderr}")
        raise RuntimeError(f"\nFailed to convert AV1 video: {e}")

def terminate_existing_processes():
    """Terminates any other running instances of the current script."""
    current_pid = os.getpid()
    
    # Find the actual main.py script name
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline'] and 'main.py' in ' '.join(proc.info['cmdline']):
                if proc.info['pid'] != current_pid:
                    print(f"\n🧹 Found existing main.py process (PID: {proc.info['pid']}). Terminating...")
                    proc.terminate()
                    proc.wait(timeout=5) # Wait for process to terminate
                    if proc.is_running():
                        print(f"\n💀 Process {proc.info['pid']} did not terminate gracefully, killing...")
                        proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def get_video_info(video_path: str) -> tuple[dict, str]: # Corrected return type hint
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
        raise RuntimeError(f"\nFailed to probe video info for {video_path}: {e}")

def system_checks():
    """Performs and prints results of system checks like disk space and GPU."""
    # Check disk space
    try:
        free_space = shutil.disk_usage('.').free / (1024**3)
        print(f"💾 Available disk space: {free_space:.1f}GB")
        if free_space < 20:
            print("⚠️  Warning: Low disk space (< 20GB).")
    except Exception as e:
        print(f"❌ Could not check disk space: {e}")

    # Check for FFmpeg
    try:
        subprocess.run([config.get('ffmpeg_path'), "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is installed and accessible.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ CRITICAL: FFmpeg not found. Please install it and ensure it's in your system's PATH.")
        
    # Determine active LLM model and its provider
    active_llm_model_name = config.get('llm.current_active_llm_model')
    active_image_model_name = config.get('llm.current_active_image_model')

    # Get model details from config
    llm_models_config = config.get('llm.models', {})
    
    active_llm_provider = llm_models_config.get(active_llm_model_name, {}).get('provider')
    active_image_provider = llm_models_config.get(active_image_model_name, {}).get('provider')

    # Check for LLM service based on active provider
    if active_llm_provider == 'ollama' or active_image_provider == 'ollama':
        try:
            client = ollama.Client(host=config.get('llm.api_keys.ollama'))
            client.list() # This will raise an exception if Ollama is not running
            print("✅ Ollama service is running.")
        except Exception as e:
            print(f"❌ CRITICAL: Ollama service not found or not reachable at {config.get('llm.api_keys.ollama')}. Please ensure Ollama is installed and running. Error: \n{e}")
            return # Exit if Ollama service is not running
    elif active_llm_provider == 'gemini' or active_image_provider == 'gemini':
        try:
            gemini_api_key = config.get('llm.api_keys.gemini')
            if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
                print("❌ CRITICAL: Gemini API key not configured. Please set 'llm.api_keys.gemini' in config.yaml.")
                return
            # A simple check to see if the API is reachable
            # This is a placeholder, a more robust check would involve a small API call
            response = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_api_key}", timeout=5)
            response.raise_for_status()
            print("✅ Gemini API is reachable.")
        except requests.exceptions.RequestException as e:
            print(f"❌ CRITICAL: Gemini API not reachable. Please check your API key and network connection. Error: \n{e}")
            return
        except Exception as e:
            print(f"❌ CRITICAL: An unexpected error occurred while checking Gemini API: {e}")
            return
    else:
        print("⚠️ No active LLM or Image model provider configured for system checks.")

    # Check for LLM model availability (only if a model is active)
    if active_llm_model_name:
        if active_llm_provider == 'ollama':
            try:
                client = ollama.Client(host=config.get('llm.api_keys.ollama'))
                models = client.list()['models']
                if any(model['name'] == active_llm_model_name for model in models): # Ollama client returns 'name'
                    print(f"✅ '{active_llm_model_name}' is available.")
                else:
                    print(f"❌ CRITICAL: '{active_llm_model_name}' not found. Please download it using 'ollama pull {active_llm_model_name}'.")
            except Exception as e:
                print(f"❌ CRITICAL: Could not check Ollama LLM model availability. Error: {e}")
        elif active_llm_provider == 'gemini':
            # For Gemini, we assume if the API is reachable, the model is available
            # A more thorough check would involve listing models via the API
            print(f"✅ Gemini model '{active_llm_model_name}' is assumed available via API.")
        else:
            print(f"⚠️ Skipping availability check for LLM model '{active_llm_model_name}' (unknown provider).")

    if active_image_model_name and active_image_model_name != active_llm_model_name: # Avoid double checking if same model
        if active_image_provider == 'ollama':
            try:
                client = ollama.Client(host=config.get('llm.api_keys.ollama'))
                models = client.list()['models']
                if any(model['name'] == active_image_model_name for model in models):
                    print(f"✅ '{active_image_model_name}' is available.")
                else:
                    print(f"❌ CRITICAL: '{active_image_model_name}' not found. Please download it using 'ollama pull {active_image_model_name}'.")
            except Exception as e:
                print(f"❌ CRITICAL: Could not check Ollama Image model availability. Error: {e}")
        elif active_image_provider == 'gemini':
            print(f"✅ Gemini image model '{active_image_model_name}' is assumed available via API.")
        else:
            print(f"⚠️ Skipping availability check for Image model '{active_image_model_name}' (unknown provider).")


    # Check for PyTorch
    try:
        import torch
        print(f"✅ PyTorch is installed (version: {torch.__version__}).")
        if torch.cuda.is_available():
            # The 'version' attribute is correct for torch.version.cuda
            print(f"✅ CUDA is available (version: {torch.version.cuda}).")
        else:
            print("🐢 CUDA is NOT available. Processing will use CPU.")
    except ImportError:
        print("❌CRITICAL: PyTorch is not installed. Please install it (e.g., pip install torch torchvision torchaudio).")
    except Exception as e:
        print(f"❌Error checking PyTorch: {e}")

    # Check for OpenCV
    try:
        import cv2
        print(f"✅ OpenCV is installed (version: {cv2.__version__}).")
    except ImportError:
        print("❌ CRITICAL: OpenCV is not installed.")
    except Exception as e:
        print(f"❌ Error checking OpenCV:\n{e}")

    # Check for MediaPipe
    try:
        import mediapipe
        print(f"✅ MediaPipe is installed (version: {mediapipe.__version__}).")
    except ImportError:
        print("❌ CRITICAL: MediaPipe is not installed.")
    except Exception as e:
        print(f"❌ Error checking MediaPipe:\n{e}")

    # Check for spaCy model
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✅ spaCy model is loaded.")
    except OSError:
        print("❌ CRITICAL: spaCy model not found.")
    except ImportError:
        print("❌ CRITICAL: spaCy is not installed.")
    except Exception as e:
        print(f"❌ Error checking spaCy:\n{e}")

def print_system_info():
    """Prints detailed system information for debugging purposes."""
    print("\n\n🖥️ \u001b[94mSystem Information:\u001b[0m")
    print(f"   💻 Operating System: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"   🏗️ Architecture: {platform.machine()}")
    print(f"   🐍 Python Version: {platform.python_version()} ({platform.python_compiler()})")
    print(f"   🧠 Processor: {platform.processor()}\n")
    try:
        import psutil
        print(f"\n   💾 Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    except ImportError:
        print("\n   ℹ️ (Install 'psutil' for more RAM info)")
    except Exception as e:
        print(f"\n   ❌ Error getting RAM info:\n{e}")
    
    try:
        import torch
        if torch.cuda.is_available():
            # The 'version' attribute is correct for torch.version.cuda
            print(f"   ⚡ GPU (CUDA): {torch.cuda.get_device_name(0)}")
            print(f"   ⚡ CUDA Version: {torch.version.cuda}")
        else:
            print("   🐢 GPU: None (CUDA not available)")
    except ImportError:
        pass # Already handled in system_checks
    except Exception as e:
        print(f"   ❌ Error checking GPU: {e}")
