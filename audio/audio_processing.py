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
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import shutil
import os

logger = logging.getLogger(__name__)

def separate_vocals(input_path: str, output_dir: str) -> str:
    """Separates vocals from an audio file using Demucs.
    Note: Demucs requires a separate installation and models.
    You might need to run 'demucs -d cpu -n htdemucs_ft {input_path} -o {output_dir}' to use it.
    
    This is currently a placeholder implementation. In a real scenario, you would
    integrate with the Demucs library or call its CLI tool.
    For demonstration, it copies the original file to simulate the output.
    """
    separated_vocals_path = f"{output_dir}/htdemucs_ft/{os.path.basename(input_path).replace('.wav', '')}/vocals.wav"
    # In a real scenario, you'd run demucs here, e.g.:
    # try:
    #     subprocess.run(['demucs', '-d', 'cpu', '-n', 'htdemucs_ft', input_path, '-o', output_dir], check=True)
    # except FileNotFoundError:
    #     logger.error("Demucs command not found. Please ensure Demucs is installed and in your PATH.")
    #     raise
    # except subprocess.CalledProcessError as e:
    #     logger.error(f"Demucs failed with error: {e.stderr}")
    #     raise
    
    os.makedirs(os.path.dirname(separated_vocals_path), exist_ok=True)
    shutil.copy(input_path, separated_vocals_path)
    logger.info(f"Demucs voice separation simulated. Output at: {separated_vocals_path}")
    return separated_vocals_path

def enhance_audio(input_path: str, output_path: str):
    """Enhances audio by reducing noise.
    
    This is currently a placeholder implementation. For actual noise reduction,
    consider using libraries like 'noisereduce' or more advanced audio processing models.
    """
    try:
        y, sr = librosa.load(input_path, sr=None)
        # Example of where noise reduction logic would go:
        # from noisereduce import reduce_noise
        # reduced_noise_audio = reduce_noise(y=y, sr=sr, ...)
        # sf.write(output_path, reduced_noise_audio, sr)
        sf.write(output_path, y, sr) # Currently just copies the audio
        logger.info(f"Audio enhancement simulated. Output at: {output_path}")
    except Exception as e:
        logger.error(f"Error enhancing audio: {e}")
        # Fallback to copying if enhancement fails
        shutil.copy(input_path, output_path)
        logger.info("Falling back to copying original audio due to enhancement failure.")
    return output_path

def mix_audio(main_audio_path: str, background_audio_path: str, output_path: str, bg_volume: float = -20.0):
    """Mixes main audio with background audio using ffmpeg.
    """
    cmd = [
        'ffmpeg', '-y', '-i', main_audio_path, '-i', background_audio_path,
        '-filter_complex', f'[1:a]volume={bg_volume}dB[bg];[0:a][bg]amix=inputs=2:duration=first', 
        output_path
    ]
    logger.info(f"Mixing audio: {main_audio_path} with {background_audio_path}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Audio mixing complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio mixing failed. Stderr: {e.stderr}")
        raise
    return output_path


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video using FFmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", # Use WAV for Whisper
        "-avoid_negative_ts", "make_zero", audio_path
    ]
    logger.info("Running FFmpeg audio extraction...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg stderr:\n{result.stderr}")
        raise RuntimeError("FFmpeg audio extraction failed.")
    logger.info("FFmpeg audio extraction complete.")

def normalize_audio_loudness(input_audio_path: str, output_audio_path: str) -> None:
    """Normalize audio loudness using ffmpeg-normalize."""
    logger.info(f"Normalizing audio loudness for {input_audio_path}...")
    try:
        norm = FFmpegNormalize(
            print_stats=False
        )
        norm.add_media_file(input_audio_path, output_audio_path)
        norm.run_normalization()
        logger.info("Audio loudness normalization complete.")
    except Exception as e:
        logger.error(f"Audio loudness normalization failed: {e}")
        # Fallback to just copying the file if normalization fails
        shutil.copy(input_audio_path, output_audio_path)
        logger.info("Falling back to copying original audio due to normalization failure.")

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
                logger.info("CUDA detected and working, using GPU...")
            except Exception as cuda_error:
                logger.warning(f"CUDA available but not working properly: {cuda_error}")
                logger.info("Falling back to CPU...")
                device = "cpu"
                compute_type = "int8"
        
        logger.info(f"Loading faster-whisper model '{config.get('whisper_model')}' on {device}...")
        
        try:
            model = WhisperModel(
                config.get('whisper_model'), 
                device=device, 
                compute_type=compute_type,
                cpu_threads=4 if device == "cpu" else 0
            )
        except Exception as model_error:
            logger.error(f"Failed to load model on {device}: {model_error}")
            if device == "cuda":
                logger.info("Trying CPU fallback...")
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

        extract_audio(video_path, raw_audio_path)
        normalize_audio_loudness(raw_audio_path, normalized_audio_path)
        
        # Voice separation
        separated_vocals_path = separate_vocals(normalized_audio_path, get_temp_path('demucs_output'))
        
        # Audio enhancement
        enhanced_vocals_path = enhance_audio(separated_vocals_path, get_temp_path('enhanced_vocals.wav'))

        logger.info(f"Running faster-whisper transcription on {device}...")
        # Disable VAD filter to ensure timestamps directly correspond to the original audio's timeline
        # This is to test the hypothesis that VAD is causing A/V desynchronization.
        segments, info = model.transcribe(
            enhanced_vocals_path, # Transcribe only vocals
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False # VAD filter explicitly set to False
        )
        
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
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
            logger.info("faster-whisper model unloaded and GPU memory released.")

class TranscriptAnalysis(BaseModel):
    themes: List[str] = Field(description="Main themes or topics discussed.")
    sentiment: str = Field(description="Overall sentiment (e.g., positive, negative, neutral, mixed).")
    speaker_changes: str = Field(description="Any noticeable speaker changes or distinct voices (if detectable from the text).")
    key_takeaways: List[str] = Field(description="Key takeaways or summary points.")

def analyze_transcript_with_llm(transcript: list[dict]) -> dict:
    """Analyzes the transcript using an LLM to identify themes, sentiment, and speaker changes."""
    logger.info("Analyzing transcript with LLM for themes, sentiment, and speaker changes...")
    
    # Prepare a simplified version of the transcript for the LLM
    simplified_transcript = [{
        "start": round(seg['start'], 1),
        "end": round(seg['end'], 1),
        "text": seg['text']
    } for seg in transcript]

    # Generate the JSON schema for TranscriptAnalysis
    transcript_analysis_schema = TranscriptAnalysis.model_json_schema()

    system_prompt_for_analysis = f"""
You are an expert in transcript analysis. Your task is to analyze the provided transcript and extract key information.
You MUST output a JSON object that strictly adheres to the following Pydantic schema:
{transcript_analysis_schema}
"""

    llm_prompt = f"""
Analyze the following video transcript and provide your response in the exact JSON format specified in the system prompt.

Transcript:
{simplified_transcript}

"""
    
    try:
        analysis_results_obj = llm_interaction.robust_llm_json_extraction(
            system_prompt=system_prompt_for_analysis.strip(),
            user_prompt=llm_prompt.strip(),
            output_schema=TranscriptAnalysis
        )
        
        logger.info("Transcript analysis by LLM complete.")
        return analysis_results_obj.model_dump()
    except Exception as e:
        logger.error(f"Failed to analyze transcript with LLM: {e}")
        return {"themes": [], "sentiment": "unknown", "speaker_changes": "not detected", "key_takeaways": []}

def extract_audio_rhythm(audio_path: str, config: dict) -> Dict[str, Any]:
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
        logger.error(f"Error extracting audio rhythm: {e}")
        return {}
