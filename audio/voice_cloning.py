from typing import Any
from TTS.api import TTS
import torch
from core.temp_manager import get_temp_path

def generate_voice_from_text(text: str, output_path: str, speaker_wav_path: str = None) -> bool:
    """
    Generates speech from text, optionally cloning a voice from an input audio file.
    Returns True on success, False on failure.
    """
    print(f"Generating voice from text: '{text[:50]}...'")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Initialize TTS model. Using xtts_v2 for voice cloning capabilities.
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
        
        if speaker_wav_path and speaker_wav_path != "":
            print(f"Cloning voice from {speaker_wav_path}...")
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav_path,
                language="en" # Assuming English for now
            )
        else:
            print("Generating voice with default speaker...")
            # If no speaker_wav is provided, use a default speaker (e.g., from the model's default speakers)
            # This might require specifying a speaker_idx or speaker_id depending on the model.
            # For xtts_v2, if speaker_wav is not provided, it uses a default voice.
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                language="en"
            )
        
        print(f"✅ Voice generation successful. Output saved to {output_path}")
        return True
    except Exception as e:
        print(f"⚠️ Voice generation failed: {e}")
        return False

def clone_voice(input_audio: str, text: str, voice_profile: Any = None) -> str:
    """
    Clones a voice from an audio file and uses it to generate speech for the given text.
    This function is kept for backward compatibility but generate_voice_from_text is preferred.
    """
    output_path = get_temp_path("cloned_voice.wav")
    success = generate_voice_from_text(text, output_path, input_audio)
    return output_path if success else ""