from typing import Any, Optional
from TTS.api import TTS
import torch
from core.temp_manager import get_temp_path
from core.gpu_manager import gpu_manager

class VoiceCloning:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        self.model_name = model_name
        self.tts = None

    def _load_model(self):
        if self.tts is None:
            print(f"Loading TTS model: {self.model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tts = TTS(self.model_name, progress_bar=True).to(device)
            gpu_manager.load_model("tts_model", self.tts, priority=3)

    def generate_voice(self, text: str, output_path: str, speaker_wav: Optional[str] = None, language="en") -> bool:
        """Generates speech from text, optionally cloning a voice."""
        try:
            self._load_model()
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=language
            )
            print(f"✅ Voice generation successful: {output_path}")
            return True
        except Exception as e:
            print(f"⚠️ Voice generation failed: {e}")
            return False
        finally:
            gpu_manager.unload_model("tts_model")

# Functions for backward compatibility
def generate_voice_from_text(text: str, output_path: str, speaker_wav_path: str = None) -> bool:
    cloner = VoiceCloning()
    return cloner.generate_voice(text, output_path, speaker_wav=speaker_wav_path)

def clone_voice(input_audio: str, text: str, voice_profile: Any = None) -> str:
    output_path = get_temp_path("cloned_voice.wav")
    success = generate_voice_from_text(text, output_path, input_audio)
    return output_path if success else ""
