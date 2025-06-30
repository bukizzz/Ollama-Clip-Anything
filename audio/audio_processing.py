# audio_processing.py
"""
Handles audio extraction and transcription of the video file using faster-whisper.
"""
import subprocess
import gc
import torch
from faster_whisper import WhisperModel
from core.temp_manager import get_temp_path
from core.config import WHISPER_MODEL

def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video using FFmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", # Use WAV for Whisper
        "-avoid_negative_ts", "make_zero", audio_path
    ]
    print("Running FFmpeg audio extraction...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg stderr:\n", result.stderr)
        raise RuntimeError("FFmpeg audio extraction failed.")

def transcribe_video(video_path: str) -> list[dict]:
    """Transcribe video using faster-whisper with precise timing."""
    model = None
    try:
        # Try to initialize faster-whisper model with fallback options
        device = "cpu"  # Start with CPU as default
        compute_type = "int8"
        
        # Only try CUDA if torch reports it's available and we can test it
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality first
                torch.cuda.current_device()
                device = "cuda"
                compute_type = "float16"
                print("CUDA detected and working, using GPU...")
            except Exception as cuda_error:
                print(f"CUDA available but not working properly: {cuda_error}")
                print("Falling back to CPU...")
                device = "cpu"
                compute_type = "int8"
        
        print(f"Loading faster-whisper model '{WHISPER_MODEL}' on {device}...")
        
        try:
            model = WhisperModel(
                WHISPER_MODEL, 
                device=device, 
                compute_type=compute_type,
                cpu_threads=4 if device == "cpu" else 0
            )
        except Exception as model_error:
            print(f"Failed to load model on {device}: {model_error}")
            if device == "cuda":
                print("Trying CPU fallback...")
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(
                    WHISPER_MODEL, 
                    device=device, 
                    compute_type=compute_type,
                    cpu_threads=4
                )
            else:
                raise
        
        audio_path = get_temp_path("temp_audio.wav")
        extract_audio(video_path, audio_path)

        print(f"Running faster-whisper transcription on {device}...")
        try:
            segments, info = model.transcribe(
                audio_path,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,  # Voice activity detection for better accuracy
                vad_parameters=dict(min_silence_duration_ms=500)  # Optimize for speech detection
            )
        except Exception as transcribe_error:
            print(f"Transcription failed with VAD enabled: {transcribe_error}")
            print("Retrying without VAD...")
            segments, info = model.transcribe(
                audio_path,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False  # Disable VAD as fallback
            )
        
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Convert generator to list with proper formatting
        transcription = []
        for segment in segments:
            words = []
            if segment.words:
                for word in segment.words:
                    words.append({
                        'start': float(word.start),
                        'end': float(word.end),
                        'text': word.word.strip()
                    })
            transcription.append({
                'start': float(segment.start),
                'end': float(segment.end),
                'text': segment.text.strip(),
                'words': words
            })
            
        return transcription
        
    finally:
        if model:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("faster-whisper model unloaded and GPU memory released.")
