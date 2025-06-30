import subprocess
import gc
import torch
from faster_whisper import WhisperModel
from core.config import WHISPER_MODEL
from core.temp_manager import get_temp_path
from ffmpeg_normalize import FFmpegNormalize

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

def normalize_audio_loudness(input_audio_path: str, output_audio_path: str) -> None:
    """Normalize audio loudness using ffmpeg-normalize."""
    print(f"Normalizing audio loudness for {input_audio_path}...")
    norm = FFmpegNormalize()
    norm.loudness_target = -23.0  # EBU R128 standard
    norm.true_peak_target = -1.0
    norm.print_stats = False
    norm.add_media_file(input_audio_path, output_audio_path)
    try:
        norm.run_normalization()
        print("Audio loudness normalization complete.")
    except Exception as e:
        print(f"Audio loudness normalization failed: {e}")
        # Fallback to just copying the file if normalization fails
        subprocess.run(["cp", input_audio_path, output_audio_path], check=True)
        print("Falling back to copying original audio due to normalization failure.")

def voice_separation(input_audio_path: str, output_vocals_path: str, output_music_path: str) -> None:
    """Placeholder for voice separation. This would typically use a model like Demucs or Spleeter."""
    print("Voice separation is not yet implemented. Skipping this step.")
    # TODO: Integrate a voice separation library here (e.g., Demucs, Spleeter)
    # For now, we'll just copy the original audio to the vocals path.
    subprocess.run(["cp", input_audio_path, output_vocals_path], check=True)
    subprocess.run(["cp", input_audio_path, output_music_path], check=True) # For demonstration, copy to music as well

def transcribe_video(video_path: str) -> list[dict]:
    """Transcribe video using faster-whisper with precise timing."""
    model = None
    try:
        device = "cpu"
        compute_type = "int8"
        
        if torch.cuda.is_available():
            try:
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
        
        raw_audio_path = get_temp_path("temp_audio_raw.wav")
        extracted_audio_path = get_temp_path("temp_audio_extracted.wav")
        normalized_audio_path = get_temp_path("temp_audio_normalized.wav")
        vocals_audio_path = get_temp_path("temp_audio_vocals.wav")
        music_audio_path = get_temp_path("temp_audio_music.wav")

        extract_audio(video_path, raw_audio_path)
        normalize_audio_loudness(raw_audio_path, normalized_audio_path)
        voice_separation(normalized_audio_path, vocals_audio_path, music_audio_path) # Use normalized audio for separation

        print(f"Running faster-whisper transcription on {device}...")
        try:
            segments, info = model.transcribe(
                vocals_audio_path, # Transcribe only vocals
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
        except Exception as transcribe_error:
            print(f"Transcription failed with VAD enabled: {transcribe_error}")
            print("Retrying without VAD...")
            segments, info = model.transcribe(
                vocals_audio_path, # Transcribe only vocals
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False
            )
        
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
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
