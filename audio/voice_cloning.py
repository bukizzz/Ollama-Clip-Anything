from typing import Any
from TTS.api import TTS
import torch
from core.temp_manager import get_temp_path

def clone_voice(input_audio: str, text: str, voice_profile: Any = None) -> str:
    """
    Clones a voice from an audio file and uses it to generate speech for the given text.
    """
    print("Cloning voice...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
        
        output_path = get_temp_path("cloned_voice.wav")
        
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=input_audio,
            language="en"
        )
        
        print(f"✅ Voice cloned successfully. Output saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"⚠️ Voice cloning failed: {e}")
        return ""
