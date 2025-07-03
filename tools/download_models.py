import subprocess
from core.config import config

def download_models():
    """Downloads external models and assets."""
    print("⬇️ \033[94mExecuting model download script...\033[0m")

    # Download Ollama models
    ollama_models = [config.get('llm_model'), config.get('llm.image_model')]
    for model in ollama_models:
        if model:
            print(f"Attempting to download Ollama model: {model}")
            try:
                subprocess.run(["ollama", "pull", model], check=True)
                print(f"✅ Successfully downloaded {model}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download Ollama model {model}: {e}")
                print("Please ensure Ollama is installed and running, and the model name is correct.")

    # Provide instructions for Whisper model if not found
    if config.get('whisper_model'):
        print(f"ℹ️ Whisper model '{config.get('whisper_model')}' is used for transcription.")
        print("   If you encounter issues, ensure it's correctly installed for faster-whisper.")
        print("   You might need to download it manually using: ")
        print(f"   `pip install faster-whisper` then `faster-whisper --model {config.get('whisper_model')}` (or similar, check faster-whisper documentation)")

    print("\n⬇️ \033[94mModel download script finished.\033[0m")
