# Technical Context: 60-Second Clips Generator

## Technologies Used:
- **Python:** Primary programming language.
- **FFmpeg:** Command-line tool for video and audio processing (extraction, cutting, merging, effects). Integrated via `os.environ` and direct command execution. Configured with `ffmpeg_path: /usr/local/bin/ffmpeg` and specific encoder parameters (`av1_nvenc`, `h264_nvenc`, `hevc_nvenc`).
- **Ollama:** Local LLM inference server. Used for various AI tasks, including multimodal analysis (Qwen-VL), clip selection, storyboarding, and content direction. Specific Ollama models configured include `codegemma:7b`, `codegemma:7b-instruct`, `deepseek-coder:6.7b`, `dolphin3:latest`, `gemma3:4b`, `llama3.1:8b`, `mistral:latest`, `phi3:3.8b`, `phi3:3.8b-instruct`, `qwen2.5-coder:latest`, `qwen3:8b`. `ollama_keep_alive` is set to `-1` (indefinite).
- **LangChain:** Framework for developing applications powered by language models. Used for interacting with Ollama and Google Gemini models.
- **Pydantic:** Data validation and settings management library. Crucial for defining and validating structured outputs from LLMs.
- **PyTorch:** Deep learning framework. Used by various models for audio and visual analysis.
- **Hugging Face Transformers:** Provides pre-trained models for tasks like sentiment analysis (`distilbert-base-uncased-finetuned-sst-2-english`).
- **MediaPipe:** Google's framework for building multimodal applied ML pipelines. Used for holistic pose, hand, and face landmarks.
- **DeepFace:** Facial analysis library for emotion detection.
- **Librosa:** Python library for audio analysis (tempo, rhythm, pitch).
- **Scikit-learn:** Machine learning library. Used for clustering (e.g., KMeans for audio themes).
- **OpenCV (`cv2`):** Computer vision library for frame processing, image manipulation, and basic video operations.
- **Tqdm:** Progress bar library for visualizing pipeline execution.
- **Tiktoken:** OpenAI's tokenizer for estimating token counts in prompts.

## Development Setup:
- **Python Environment:** Recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.
- **Dependencies:** Managed via `requirements.txt`. Key packages include `opencv-python`, `torch`, `torchvision`, `mediapipe`, `spacy`, `scikit-learn`, `librosa`, `webcolors`, `Pillow`, `TTS`, `demucs`, `langchain-ollama`, `langchain-google-genai`, `pydantic`, `tiktoken`, `deepface`, `pyannote.audio`.
- **FFmpeg Installation:** Must be installed and accessible in the system's PATH, or its path configured in `core/config.yaml` (`ffmpeg_path`).
- **Ollama Installation:** Ollama server must be running locally, and required models (e.g., `llava`, `llama3.1`, `gemma-3-12b-it`, `qwen2.5vl:3b`) must be pulled.
- **Hugging Face Token:** A Hugging Face authentication token is required for `pyannote.audio` models and should be configured in `core/config.yaml` (`huggingface_tokens.pyannote_audio`).
- **Google Gemini API Key:** If using Gemini models, an API key must be configured in `core/config.yaml` (`llm.api_keys.gemini`).

## Technical Constraints:
- **LLM Token Limits:** Prompts are adaptively built using `prompt_utils.py` to stay within model-specific token limits.
- **GPU Memory:** Large models (LLMs, PyTorch models) can consume significant GPU memory. `ResourceManager` and `llm_interaction.cleanup()` are used to manage this, but high-end GPUs are beneficial.
- **Processing Time:** Video analysis and LLM inference are computationally intensive, leading to potentially long processing times for longer videos. Caching is used to mitigate this for repeated runs.
- **Model Availability:** Reliance on external models (Ollama, Hugging Face, DeepFace) means their availability and performance directly impact the pipeline.
- **FFmpeg Compatibility:** Assumes a compatible FFmpeg installation.
- **Pydantic Strictness:** While beneficial for validation, strict Pydantic schemas require LLM outputs to be precise, necessitating robust retry and correction mechanisms.

## Dependencies:
- `requirements.txt`: Lists all Python package dependencies.
- Pre-trained models:
    - Ollama models (e.g., `llava`, `llama3.1`, `gemma-3-12b-it`, `qwen2.5vl:3b`, `gemma-3-4b-it`, `gemma-3-1b-it`, `gemma-3-27b-it`, `qwen2.5-coder:7b`)
    - Hugging Face models (e.g., `distilbert-base-uncased-finetuned-sst-2-english`, `pyannote/speaker-diarization-3.1`, `all-MiniLM-L6-v2`)
    - DeepFace models (internal to DeepFace, e.g., `mediapipe` backend)
    - YuNet face detection ONNX model (`weights/face_detection_yunet_2023mar.onnx`)

## Tool Usage Patterns:
- **`execute_command`:** Used for system-level operations, potentially for FFmpeg commands if not abstracted by Python libraries.
- **`read_file` / `write_to_file`:** For managing configuration, state, and logging.
- **`list_files` / `search_files`:** For exploring the codebase and asset directories.
- **`use_mcp_tool` / `access_mcp_resource`:** Not currently used, but the architecture is designed to be compatible with MCP for external tool/resource integration.
- **`browser_action` / `web_fetch`:** Not directly used in the core video processing pipeline, but could be used for fetching video URLs or external data if needed.

## Command-line Arguments:
The `main.py` script accepts the following command-line arguments for controlling application behavior:
- `--video_path <path>`: Specifies the path to a local MP4 video file for processing.
- `--youtube_url <url>`: Provides the URL of a YouTube video to download and process.
- `--youtube_quality <int>`: Sets the desired quality for YouTube video downloads (e.g., 0, 1, 2...).
- `--user_prompt "<prompt>"`: An optional prompt for the LLM to guide clip selection.
- `--retry`: A flag to automatically resume from a previous failed session.
- `--nocache`: A flag to force a fresh start, deleting any existing state and temporary files.
- `--image_analysis_fps <float>`: Sets the frames per second for image analysis.
