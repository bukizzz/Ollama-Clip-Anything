import subprocess
import torch
from faster_whisper import WhisperModel
from core.config import config
from core.temp_manager import get_temp_path
from ffmpeg_normalize import FFmpegNormalize
import logging
from llm import llm_interaction
from core.gpu_manager import gpu_manager
import librosa
import numpy as np
import soundfile as sf


def separate_vocals(input_path: str, output_dir: str) -> str:
    """Separates vocals from an audio file using Demucs.
    Note: Demucs requires a separate installation and models.
    You might need to run 'demucs -d cpu -n htdemucs_ft {input_path} -o {output_dir}' to use it.
    """
    # This is a placeholder. Actual Demucs integration would involve calling its API or CLI.
    # For now, we'll just return the input path as if no separation occurred.
    separated_vocals_path = f"{output_dir}/htdemucs_ft/{input_path.split('/')[-1].replace('.wav', '')}/vocals.wav"
    # In a real scenario, you'd run demucs here:
    # subprocess.run(['demucs', '-d', 'cpu', '-n', 'htdemucs_ft', input_path, '-o', output_dir], check=True)
    # For demonstration, we'll just copy the original file to simulate output
    import shutil
    import os
    os.makedirs(os.path.dirname(separated_vocals_path), exist_ok=True)
    shutil.copy(input_path, separated_vocals_path)
    print(f"Demucs voice separation simulated. Output at: {separated_vocals_path}")
    return separated_vocals_path

def enhance_audio(input_path: str, output_path: str):
    """Enhances audio by reducing noise.
    """
    # This is a placeholder for a more advanced noise reduction model
    y, sr = librosa.load(input_path, sr=None)
    # Simple noise reduction could be implemented with libraries like 'noisereduce'
    sf.write(output_path, y, sr)
    return output_path

def mix_audio(main_audio_path: str, background_audio_path: str, output_path: str, bg_volume: float = -20.0):
    """Mixes main audio with background audio using ffmpeg.
    """
    cmd = [
        'ffmpeg', '-y', '-i', main_audio_path, '-i', background_audio_path,
        '-filter_complex', f'[1:a]volume={bg_volume}dB[bg];[0:a][bg]amix=inputs=2:duration=first', 
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

logger = logging.getLogger(__name__)

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
        logger.error("FFmpeg stderr:\n", result.stderr)
        raise RuntimeError("FFmpeg audio extraction failed.")

def normalize_audio_loudness(input_audio_path: str, output_audio_path: str) -> None:
    """Normalize audio loudness using ffmpeg-normalize."""
    print(f"Normalizing audio loudness for {input_audio_path}...")
    norm = FFmpegNormalize()
    norm.loudness_target = -23.0  # EBU R128 standard
    norm.true_peak_target = -1.0
    norm.loudness_range_target = 10.0 # Added to address the warning
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
        
        print(f"Loading faster-whisper model '{config.get('whisper_model')}' on {device}...")
        
        try:
            model = WhisperModel(
                config.get('whisper_model'), 
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
                    config.get('whisper_model'), 
                    device=device, 
                    compute_type=compute_type,
                    cpu_threads=4
                )
            else:
                raise
        
        raw_audio_path = get_temp_path("temp_audio_raw.wav")
        normalized_audio_path = get_temp_path("temp_audio_normalized.wav")
        # vocals_audio_path = get_temp_path("temp_audio_vocals.wav")

        extract_audio(video_path, raw_audio_path)
        normalize_audio_loudness(raw_audio_path, normalized_audio_path)
        
        # Voice separation
        separated_vocals_path = separate_vocals(normalized_audio_path, get_temp_path('demucs_output'))
        
        # Audio enhancement
        enhanced_vocals_path = enhance_audio(separated_vocals_path, get_temp_path('enhanced_vocals.wav'))

        print(f"Running faster-whisper transcription on {device}...")
        try:
            segments, info = model.transcribe(
                enhanced_vocals_path, # Transcribe only vocals
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
        except Exception as transcribe_error:
            print(f"Transcription failed with VAD enabled: {transcribe_error}")
            print("Retrying without VAD...")
            segments, info = model.transcribe(
                enhanced_vocals_path, # Transcribe only vocals
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
            gpu_manager.release_gpu_memory()
            print("faster-whisper model unloaded and GPU memory released.")

def analyze_transcript_with_llm(transcript: list[dict]) -> dict:
    """Analyzes the transcript using an LLM to identify themes, sentiment, and speaker changes."""
    print("Analyzing transcript with LLM for themes, sentiment, and speaker changes...")
    
    # Prepare a simplified version of the transcript for the LLM
    simplified_transcript = [{
        "start": round(seg['start'], 1),
        "end": round(seg['end'], 1),
        "text": seg['text']
    } for seg in transcript]

    llm_prompt = f"""
    Analyze the following video transcript and provide:
    1. Main themes or topics discussed.
    2. Overall sentiment (e.g., positive, negative, neutral, mixed).
    3. Any noticeable speaker changes or distinct voices (if detectable from the text).
    4. Key takeaways or summary points.

    Transcript:
    {simplified_transcript}

    Provide the output in a structured JSON format with keys: "themes", "sentiment", "speaker_changes", "key_takeaways".
    """
    
    try:
        response = llm_interaction.llm_pass(config.get('llm_model'), [
            {"role": "system", "content": "You are an expert in transcript analysis."},
            {"role": "user", "content": llm_prompt.strip()}
        ])
        
        analysis_results = llm_interaction.extract_json_from_text(response)
        print("Transcript analysis by LLM complete.")
        return analysis_results
    except Exception as e:
        print(f"❌ \033[91mFailed to analyze transcript with LLM: {e}\033[0m")
        return {"themes": [], "sentiment": "unknown", "speaker_changes": "not detected", "key_takeaways": []}

def extract_audio_rhythm(audio_path: str, config: dict) -> dict:
    """Extracts tempo, beats, and speech rhythm patterns from an audio file."""
    try:
        y, sr = librosa.load(audio_path)

        # Extract audio tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Analyze speech rhythm patterns (emphasis detection)
        rms = librosa.feature.rms(y=y)[0]
        emphasis_threshold_std = config.get('audio_rhythm', {}).get('emphasis_threshold_std', 1.5)
        emphasis_threshold = np.mean(rms) + emphasis_threshold_std * np.std(rms)
        emphasized_segments = np.where(rms > emphasis_threshold)[0]
        emphasized_times = librosa.frames_to_time(emphasized_segments, sr=sr)

        # Generate rhythm map
        rhythm_map = {
            'tempo': tempo,
            'beat_times': beat_times.tolist(),
            'emphasized_times': emphasized_times.tolist(),
            'emphasis_threshold': emphasis_threshold
        }
        
        return rhythm_map
    except Exception as e:
        print(f"Error extracting audio rhythm: {e}")
        return None
