# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1] - 2025-07-02

### Added

-   **Unit Testing Framework:**
    -   Introduced a `tests/` directory with `pytest` configuration (`pytest.ini`) and an example unit test (`test_utils.py`) for `core/utils.py`. This marks a significant step towards improving code quality and maintainability.
-   **Centralized Tracking Management:**
    -   New module `video/tracking_manager.py` introduced to centralize the instantiation and management of `FaceTracker` and `ObjectTracker` instances. This ensures that a single instance of each tracker is used across the application, improving efficiency and consistency.
    -   `VideoAnalysisAgent` and `VideoEditingAgent` now obtain tracker instances from `TrackingManager` instead of creating them directly or receiving them as arguments.
    -   `analysis/analysis_and_reporting.py` also uses `TrackingManager` for obtaining trackers during video content analysis.
-   **Refactored Clip Enhancement Logic:**
    -   The core logic for creating individual enhanced clips has been extracted from `video/video_editing.py` into a new dedicated module `video/clip_enhancer.py`. This improves modularity and separation of concerns.
-   **Enhanced State Management:**
    -   A new function `handle_pipeline_completion` has been added to `core/state_manager.py` to centralize the logic for managing state files and temporary directories based on the pipeline's success or failure.
-   **Automated Model Downloads:**
    -   `tools/download_models.py` now includes actual `ollama pull` commands for `LLM_MODEL` and `IMAGE_RECOGNITION_MODEL`, making model setup more automated.


### Changed

-   **LLM Interaction:**
    -   The `llm/llm_interaction.py` module has been refactored to use a more streamlined, single-pass JSON extraction approach (`robust_llm_json_extraction`) instead of the previous three-pass method, aiming for improved efficiency while maintaining robustness.
    -   `get_clips_with_retry` has been renamed to `get_clips_from_llm` to reflect the updated extraction strategy.
    -   The `cleanup()` function in `llm/llm_interaction.py` no longer attempts to directly `ollama stop` models, but instead focuses on clearing PyTorch GPU memory and provides clear guidance to the user for manual Ollama model unloading or server restart if needed.
-   **Agent and Pipeline Logic:**
    -   `main.py` no longer manually orchestrates `VideoAnalysisAgent` and `VideoEditingAgent`; the `AgentManager` now handles the full pipeline execution.
    -   `main.py` now utilizes `state_manager.handle_pipeline_completion` in its `finally` block, simplifying the main application flow.
    -   `VideoAnalysisAgent` and `VideoEditingAgent` no longer accept `face_tracker_instance` and `object_tracker_instance` as arguments, as they now obtain them from `TrackingManager`.
    -   `analysis/analysis_and_reporting.py` no longer takes `face_tracker` and `object_tracker` as arguments for `analyze_video_content`.
-   **Configuration:**
    -   The default `LLM_MODEL` in `core/config.py` and `core/config.yaml` has been updated from `qwen2.5-coder:7b` to `deepseek-coder:6.7b`.
    -   The `ffmpeg_path` entry in `core/config.yaml` is now commented out by default, with a note to uncomment and set if FFmpeg is not in the system PATH.
    -   `main.py` now uses `os.environ["FFMPEG_BINARY"] = FFMPEG_PATH` for setting the FFmpeg binary path, ensuring consistency with the configured path.
-   **Dependencies:**
    -   `pyproject.toml` and `requirements.txt` updated to reflect the new dependencies and changes in existing ones.
    -   `librosa` version is unpinned in `requirements.txt`.
    -   `mediapipe` version is updated in `requirements.txt`.
    -   `typer` and `psutil` are added to `pyproject.toml` and `requirements.txt`.
-   **Video Editing:**
    -   `video/video_editing.py` now imports `create_enhanced_individual_clip` from `video/clip_enhancer.py`.
    -   `batch_create_enhanced_clips` in `video/video_editing.py` now takes `tracking_manager` and `output_dir` as arguments.
    -   `batch_process_with_analysis` in `video/video_editing.py` now instantiates `TrackingManager`.

### Removed

-   **Three-Pass LLM Extraction:**
    -   The `three_pass_llm_extraction` function has been removed from `llm/llm_interaction.py` in favor of the new single-pass approach.

## [4.0.9.2] - 2025-07-02

### Added

-   **B-Roll Integration:**
    -   Introduced `agents/broll_analysis_agent.py` to scan and analyze B-roll assets from the `b_roll_assets` directory.
    -   The `LLMSelectionAgent` now incorporates B-roll suggestions into the clip selection process.
    -   `video/frame_processor.py` now supports rendering B-roll images in a split-screen format.
-   **Split-Screen Mode:**
    -   `video/frame_processor.py` can now create split-screen videos when two faces are detected.
    -   `audio/subtitle_generation.py` adjusts subtitle positioning for split-screen layouts.
-   **CLI Interface:**
    -   Added `cli.py` with `typer` to provide a more robust and user-friendly command-line interface.
-   **GPU Management:**
    -   Introduced `core/gpu_manager.py` to handle GPU memory cleanup.

### Changed

-   **Configuration:**
    -   All configuration is now centralized in `core/config.yaml`, including agent enablement. `core/config.py` loads all values from the YAML file.
-   **LLM Interaction:**
    -   Refactored `llm/llm_interaction.py` for more robust JSON extraction and to handle B-roll and split-screen suggestions.
    -   Added `llm/image_analysis.py` to centralize image description logic.
-   **Agent and Pipeline Logic:**
    -   `main.py` is refactored to work with the new `cli.py` and a dictionary of arguments.
    -   The main pipeline now manually orchestrates the `VideoAnalysisAgent` and `VideoEditingAgent` to pass tracker instances.
    -   `core/agent_manager.py` now checks the `AGENT_CONFIG` in `core/config.yaml` to determine which agents to run.
-   **Dependencies:**
    -   `typer` added to `requirements.txt`.
    -   `spacy` is no longer pinned to a specific version.
-   **Voice Cloning:**
    -   The placeholder in `audio/voice_cloning.py` has been replaced with a full implementation using the `TTS` library.
-   **Video Editing:**
    -   `video/video_editing.py` now uses `SceneDetector` to improve dynamic framing by resetting smoothing at scene changes.

### Fixed

-   **Resource Cleanup:** Improved GPU memory management by explicitly unloading Ollama models and clearing the PyTorch cache.
-   **LLM Robustness:** Implemented a `clean_numerical_value` function to prevent errors from hallucinated units in LLM output.

## [4.0.9.1] - 2025-07-01

### Added

-   **Enhanced Subtitle Styling:**
    -   Introduced `HIGHLIGHT_FONT_COLOR`, `HIGHLIGHT_OUTLINE_COLOR`, `SUBTITLE_BACKGROUND_COLOR`, and `SUBTITLE_BORDER_RADIUS` in `core/config.py` for more customizable subtitle appearance.
    -   Updated `audio/subtitle_generation.py` to utilize these new configuration options, including a new `rgb_to_bgr_ass` helper for color conversion and `{\an5}` for text centering.
-   **Persistent Tracking:**
    -   Implemented persistent tracking for faces in `video/face_tracking.py` using `cv2.legacy.TrackerCSRT_create()` and `active_trackers` to maintain face IDs across frames.
    -   Implemented persistent tracking for objects in `video/object_tracking.py` using `cv2.legacy.TrackerCSRT_create()` and `active_trackers` to maintain object IDs across frames.

### Changed

-   **Configuration Updates:**
    -   Adjusted `SUBTITLE_FONT_FAMILIES` in `core/config.py` to include 'Impact', 'Arial Black', and 'Bebas Neue' for bolder subtitle options.
    -   Modified `CLIP_DURATION_RANGE` in `core/config.py` and `core/config.yaml` from (60, 90) to (45, 75) seconds.
    -   Updated `SMOOTHING_FACTOR` in `core/config.py` and `core/config.yaml` from 0.1 to 0.05 for smoother dynamic framing.
    -   Increased default `subtitle_font_size` in `core/config.py` and `core/config.yaml` from 24 to 28.
-   **Dependency Management:**
    -   Updated `mediapipe` version from `0.10.13` to `0.10.21` in `pyproject.toml` and `requirements.txt`.
    -   Pinned `pytubefix` to `1.0.0` in `pyproject.toml`.
-   **Improved Console Output:**
    -   Added color codes to various print statements across `main.py`, `agents/audio_transcription_agent.py`, `agents/base_agent.py`, `agents/content_alignment_agent.py`, `agents/llm_selection_agent.py`, `agents/results_summary_agent.py`, `agents/storyboarding_agent.py`, `agents/video_analysis_agent.py`, `agents/video_editing_agent.py`, `agents/video_input_agent.py`, `analysis/analysis_and_reporting.py`, `audio/audio_processing.py`, `core/agent_manager.py`, `core/ffmpeg_command_logger.py`, `core/llm_models.py`, `core/prompt_parser.py`, `core/temp_manager.py`, `core/utils.py`, `llm/llm_interaction.py`, `tools/download_models.py`, `video/face_tracking.py`, `video/object_tracking.py`, `video/scene_detection.py`, `video/video_effects.py`, and `video/video_input.py` for better readability and visual feedback.
-   **Logging & Debugging:**
    -   Removed debug print statements for `transcript` and `ffmpeg_params` from `video/video_editing.py`.

### Fixed

-   **FFprobe Path:** Corrected the hardcoded `ffprobe` path in `core/utils.py` from `/snap/ffmpeg/current/usr/bin/ffprobe` to `/usr/local/bin/ffprobe` for re-probing converted videos.

## [4.0.9] - 2025-07-01

### Added

-   **Modern CLI:**
    -   Introduced `typer` library for a modern command-line interface.
    -   Created `cli.py` to define new CLI commands (`run`, `config`, `tools`, `state`).

### Changed

-   **Image Analysis:**
    -   Updated `llm/image_analysis.py` to use `ollama.chat` for real-time image description.
-   **Configuration Management:**
    -   Adjusted `smoothing_factor` in `core/config.yaml` from `0.05` to `0.1` for smoother frame transitions.
    -   Refactored `core/config.py` to load all configuration values dynamically from `core/config.yaml`.
    -   Migrated all previously hardcoded configuration values from `core/config.py` into `core/config.yaml`.
    -   Added agent-specific configuration to `core/config.yaml` to enable/disable agents.
-   **Main Application Flow:**
    -   Refactored `main.py` to accept arguments as a dictionary, decoupling it from `argparse`.
    -   Updated `cli.py` to correctly invoke `main.py` with the new argument structure and adjusted `sys.path.append`.
    -   Updated `README.md` to reflect the new CLI usage instructions.
-   **Agent Management:**
    -   Modified `core/agent_manager.py` to dynamically execute agents based on the configuration settings in `core/config.yaml`.
-   **Logging:**
    -   Added basic logging configuration to `main.py`.

### Removed

-   **ImageBind, Demucs, and TTS Integration:**
    -   Removed `git+https://github.com/facebookresearch/ImageBind.git#egg=imagebind_client`, `git+https://github.com/facebookresearch/demucs#egg=demucs`, and `git+https://github.com/coqui-ai/TTS.git` from `requirements.txt` due to dependency conflicts with `torch==2.3.0`.
    -   Reverted changes in `core/llm_models.py`, `video/object_tracking.py`, `video/face_tracking.py`, `audio/voice_cloning.py`, and `audio/audio_processing.py` that integrated these libraries.

### Fixed

-   **Placeholder Implementations:** Completed the implementation of previously placeholder functions for voice cloning, voice separation, and image embedding, enhancing the system's core capabilities.

## [4.0.8] - 2025-07-01

### Added

-   **Modular Agent System:**
    -   Introduced `core/agent_manager.py` to orchestrate agent execution, replacing `agents/multi_agent.py`.
    -   Added new agents: `agents/storyboarding_agent.py` (for frame analysis and storyboard generation) and `agents/content_alignment_agent.py` (for audio/video synchronization).
    -   Updated `main.py` to integrate the new `AgentManager` and the new agents into the pipeline.
-   **Enhanced LLM Integration:**
    -   Expanded `core/llm_models.py` with more detailed placeholder functions for multi-modal LLMs (MiniCPM, ImageBind) and utility functions for base64 image conversion.
    -   Updated `core/config.yaml` to include a dedicated `llm` section for LLM configuration settings.
    -   Integrated LLM-based transcript analysis into `audio/audio_processing.py`.
    -   Implemented LLM-based user prompt parsing in `core/prompt_parser.py`.
-   **Rhythm Detection & Sync:**
    -   Implemented `detect_rhythm_and_beats` and `sync_cuts_with_beats` in `video/video_editing.py` using `librosa` for audio beat detection and clip synchronization.
-   **Visual Effects & Overlays:**
    -   Implemented `add_text_overlay` and `apply_simple_animation` in `video/video_effects.py` for text overlays and basic animations.
-   **Face Tracking Database:**
    -   Added basic face database functionality in `video/face_tracking.py` with methods for loading, saving, and managing face embeddings.
-   **Voice Cloning Placeholder:**
    -   Introduced `audio/voice_cloning.py` as a placeholder for future voice cloning/generation features.
-   **Project Structure & Dependencies:**
    -   Created `tools/download_models.py` as a placeholder for external model download scripts.
    -   Added `pyproject.toml` for modern project setup and dependency management.
    -   Updated `.gitignore` to include `video/face_db.json` and `pyproject.toml`.

### Changed

-   **Dependency Management:**
    -   Adjusted `mediapipe` version to `0.10.13` and `librosa` to an unpinned version in `requirements.txt` for compatibility.
    -   Added `Pillow` and `soundfile==0.12.1` to `requirements.txt`.
-   **Main Application Flow:**
    -   Modified `main.py` to automatically resume from a previous session state in non-interactive environments, avoiding `EOFError`. (Note: This change was reverted as per user's request to maintain interactive prompt behavior).
-   **Logging & Debugging:**
    -   Adjusted logging levels and removed some verbose print statements across `audio/audio_processing.py`, `video/object_tracking.py`, `video/scene_detection.py`, `video/video_editing.py`, and `video/video_input.py` for cleaner console output.
    -   Disabled `tqdm` progress bar in `video/video_editing.py`.

### Fixed

-   **`scipy.signal.hann` Error:** Resolved the `module 'scipy.signal' has no attribute 'hann'` error by updating `librosa` and `mediapipe` dependencies.
-   **FFprobe Path:** Corrected the hardcoded `ffprobe` path in `core/utils.py` for re-probing converted videos.

## [4.0.7] - 2025-06-30

### Added

-   **Voice Separation Placeholder:** Added a placeholder function `voice_separation` in `audio/audio_processing.py` to prepare for future voice isolation features.
-   **Rhythm Detection Placeholder:** Added a placeholder function `detect_rhythm_and_beats` in `video/video_editing.py` for future rhythm and beat detection.
-   **Visual Retrieval & Tracking Enhancements (Placeholders):**
    -   Added `face_db` and `image_embedding_model` placeholders in `video/face_tracking.py` for improved face recognition.
    -   Added `image_embedding_model` placeholder in `video/object_tracking.py` for improved object recognition.
-   **Visual Effects Enhancements (Placeholders):**
    -   Added `add_text_overlay` and `apply_simple_animation` placeholders in `video/video_effects.py` for future visual effects.
-   **Configurable Personalization Options:**
    -   Introduced `custom_clip_themes` in `core/config.yaml` for user-defined clip formatting options (e.g., vertical/horizontal split, two-face detection).
    -   `core/config.py` was updated to load these new personalization settings.
-   **Modular Tools Directory (Placeholder):** Added a `TODO` comment in `main.py` to consider creating a dedicated `tools/` directory for managing external models and libraries.

### Fixed

-   **Audio Normalization:** Corrected `FFmpegNormalize` usage in `audio/audio_processing.py` from `execute()` to `run()`.

## [4.0.6] - 2025-06-30

### Added

-   **Configurable Inputs:**
    -   Introduced `core/config.yaml` for basic configurable inputs, including subtitle font size, colors, clip duration range, smoothing factor, LLM max retries, and minimum clips needed.
    -   `core/config.py` was updated to load these settings from `config.yaml`.
    -   `pyyaml` added to `requirements.txt` for YAML parsing.
-   **Audio Processing Enhancements:**
    -   Added `ffmpeg-normalize` to `requirements.txt`.
    -   Implemented `normalize_audio_loudness` in `audio/audio_processing.py` to ensure consistent audio levels.
    -   Added a placeholder for `voice_separation` in `audio/audio_processing.py` to prepare for future voice isolation features.
    -   `transcribe_video` in `audio/audio_processing.py` now utilizes the normalized audio for transcription.
-   **Basic Visual Effects:**
    -   Added `apply_color_grading` function to `video/video_effects.py` for basic color adjustments (brightness, contrast, saturation).
-   **Enhanced LLM Prompting:**
    -   `llm/llm_interaction.py` now supports an optional `user_prompt` parameter in `three_pass_llm_extraction` to allow users to guide clip selection with specific instructions.
    -   `agents/llm_selection_agent.py` was updated to pass the `user_prompt` from the context to the LLM.
    -   `main.py` now accepts a new command-line argument `--user_prompt` to provide custom instructions to the LLM.

### Changed

-   **Subtitle Generation:** `audio/subtitle_generation.py` now dynamically uses subtitle styling parameters (font size, colors) loaded from `core/config.py`, allowing for easier customization.
-   **FFmpeg Path Configuration:** Reverted `FFMPEG_PATH` in `core/config.py` to point to the newly compiled FFmpeg at `/usr/local/bin/ffmpeg`.
-   **FFmpegNormalize Initialization:** Modified `audio/audio_processing.py` to initialize `FFmpegNormalize` with keyword arguments directly in the constructor.

### Fixed

-   **FFmpeg/NVENC Integration:**
    -   Resolved "Unknown encoder 'h264_nvenc'" error by compiling FFmpeg from source with NVENC support.
    -   Ensured MoviePy correctly uses the compiled FFmpeg by explicitly setting `FFMPEG_BINARY` in `main.py`.
    -   Fixed "No such file or directory: 'ffprobe'" error by updating `core/utils.py` to use the absolute path to the compiled `ffprobe` executable.
-   **Audio Normalization:**
    -   Corrected `FFmpegNormalize` initialization in `audio/audio_processing.py` to properly pass constructor arguments, resolving the `TypeError`.
-   **YouTube Download:**
    -   Addressed `HTTP Error 400: Bad Request` during YouTube video download by removing the version constraint for `pytubefix` in `requirements.txt` and upgrading it to the latest available version.
-   **Face Tracking:**
    -   Adjusted cropping logic in `video/frame_processor.py` to ensure the main detected face remains within the output frame, preventing it from being cut off.

## [4.0.5] - 2025-06-30

### Added

-   **Failure Recovery and Resumption System:**
    -   New module `core/state_manager.py` introduced to manage application state, enabling checkpointing and resumption of processing after failures.
    -   State is saved to `.temp_clips.json` in the new `temp_processing` directory.
    -   `main.py` now includes logic to load/save state, and new CLI arguments `--retry` (auto-resume) and `--nocache` (force fresh start) are added.
    -   `core/config.py` now defines `STATE_FILE = ".temp_clips.json"`.
-   **Process Termination Utility:**
    -   New function `terminate_existing_processes()` in `core/utils.py` to terminate other running instances of `main.py` at startup, preventing conflicts.
    -   `psutil` dependency added for process management.
-   **Ollama Model Cleanup:**
    -   New `cleanup()` function in `llm/llm_interaction.py` to explicitly unload the Ollama model and clear associated GPU memory, improving resource management.

### Changed

-   **Temporary Directory Location:** `TEMP_DIR` in `core/config.py` changed from `".temp"` to `"temp_processing"` to better reflect its purpose and avoid conflicts.
-   **Main Application Flow:** `main.py` has been significantly refactored to integrate the new state management system, allowing for seamless resumption of the pipeline from the last completed stage.
-   **LLM Interaction:** `llm/llm_interaction.py` now includes `subprocess` and `torch` imports for the new cleanup functionality.

## [4.0.4] - 2025-06-30

### Added

-   **Advanced Subtitle Generation:**
    -   Replaced the previous subtitle implementation with a more robust system that generates `.ass` (Advanced SubStation Alpha) subtitle files. This allows for word-by-word highlighting and more complex animations, providing a more professional and engaging viewing experience.
    -   The `audio/subtitle_generation.py` module has been completely rewritten to support this new format, utilizing the `whisper-timestamped` library for precise word timings.
-   **Enhanced Transcription:**
    -   The transcription process in `audio/audio_processing.py` now generates word-level timestamps, which is a critical prerequisite for the new word-by-word subtitle feature.
-   **Failure Recovery and Resumption Plan:**
    -   A detailed plan for failure recovery and resumption has been documented in `FAILURE RECOVERY AND RESUMPTION LOGIC.md`. This lays the groundwork for a more resilient and fault-tolerant application in future updates.

### Changed

-   **Dependency Updates:**
    -   Added `whisper-timestamped==1.15.8` in `requirements.txt` to support the new subtitle generation feature.
    -   Upgraded `mediapipe` from `0.10.9` to `0.10.21`.
    -   Upgraded `tqdm` from `4.0.0` to `4.67.1`.
-   **`video_editing.py`:**
    -   Significantly refactored to integrate the new `.ass` subtitle workflow. Instead of composing subtitles with MoviePy, it now leverages FFmpeg's capabilities to directly burn the `.ass` file into the video, improving performance and subtitle quality.
    -   The `create_enhanced_individual_clip` function was updated to pass the full transcript to the subtitle generation and to use the new subtitle creation method.
-   **`llm_interaction.py`:**
    -   Minor code consistency improvement by using `LLM_MAX_RETRIES` constant instead of a hardcoded value in the `get_clips_with_retry` function.

## [4.0.3] - 2025-06-29

### Changed

-   **Optimized Tracking Logic:** In `analysis/analysis_and_reporting.py`, added logic to disable object tracking if face tracking is enabled, preventing potential interference and optimizing processing.
-   **Smoother Frame Transitions:** Adjusted `SMOOTHING_FACTOR` in `core/config.py` from `0.1` to `0.05` for even smoother transitions during dynamic framing.
-   **LLM Interaction Robustness:**
    -   Introduced `LLM_MAX_RETRIES` and `LLM_MIN_CLIPS_NEEDED` in `core/config.py` for more configurable and robust LLM clip selection.
    -   Modified `llm/llm_interaction.py`'s `three_pass_llm_extraction` prompts to allow for more flexible clip identification (e.g., slight overlaps, not strictly "complete thoughts") in the initial pass, improving the LLM's ability to find potential clips.
-   **Explicit FFmpeg Path Configuration:** Set `FFMPEG_PATH` in `core/config.py` to `"/usr/bin/ffmpeg"` to explicitly define the FFmpeg executable path, improving reliability across different environments.

## [4.0.2] - 2025-06-28

### Added

-   **Command-Line Argument Support:** `main.py` now supports `argparse` for specifying video input (local file path or YouTube URL with quality) directly via command-line arguments, enabling non-interactive execution.
-   **FFmpeg Command Logging:** A new `core/ffmpeg_command_logger.py` module and integration in `video/video_editing.py` allow for capturing and printing the exact FFmpeg commands executed by MoviePy, aiding in debugging and understanding the underlying video processing.
-   **Dynamic Frame Processing:** Introduction of `video/frame_processor.py` centralizes logic for dynamic cropping and face/object-aware framing, ensuring optimal composition of clips.
-   **AV1 Video Conversion:** `core/utils.py` now includes functionality to detect and automatically convert AV1 video streams to H.265 (HEVC) using FFmpeg, improving compatibility with downstream processing.
-   **Centralized Video Encoding Configuration:** `core/config.py` now includes `VIDEO_ENCODER`, `FFMPEG_GLOBAL_PARAMS`, and `FFMPEG_ENCODER_PARAMS`, allowing for easy switching and configuration of different video encoders (e.g., `h264_nvenc`, `libx264`, `hevc_nvenc`, `av1_nvenc`).

### Changed

-   **Modularization:** The project structure has been significantly refactored into a more modular design. Core functionalities are now organized into dedicated subdirectories:
    -   `analysis/`: Contains modules for video content analysis and processing reports.
    -   `audio/`: Houses audio-related processing, including transcription and subtitle generation.
    -   `core/`: Stores central configuration, temporary file management, utility functions, and FFmpeg command logging.
    -   `llm/`: Dedicated to Large Language Model (LLM) interactions for clip selection.
    -   `video/`: Contains modules for video-specific processing, such as face tracking, object tracking, frame processing, scene detection, video effects, and video input.
-   **Robust JSON Extraction:** `llm/llm_interaction.py` features improved `extract_json_from_text` logic, making it more resilient to varied LLM output formats by attempting direct parsing and then using multiple regex patterns, including fixing trailing commas.
-   **Comprehensive System Checks:** `core/utils.py`'s `system_checks` now performs more thorough validations, including checks for Ollama service and LLM model availability, as well as installations of PyTorch, OpenCV, MediaPipe, and spaCy.
-   **Detailed System Information:** `core/utils.py`'s `print_system_info` provides more extensive system details for debugging, such as OS version, architecture, Python compiler, RAM, and GPU information.
-   **Intelligent Video Input:** `video/video_input.py`'s `choose_input_video` now intelligently detects whether user input is a local file path or a YouTube URL.
-   **YouTube AV1 Stream Filtering:** `video/video_input.py`'s `download_youtube_video` now filters out AV1 streams by default during quality selection, preventing issues with unsupported codecs.
-   **Smoother Frame Transitions:** `video/frame_processor.py` implements a `SMOOTHING_FACTOR` from `core/config.py` for smoother transitions during dynamic framing.
-   **Optimized Tracker Initialization:** `video/video_editing.py`'s `batch_process_with_analysis` now initializes `FaceTracker` and `ObjectTracker` instances once for the entire batch, improving efficiency by avoiding redundant model loading.
-   **Enhanced Error Reporting:** `main.py` now provides more detailed troubleshooting suggestions in case of errors, including specific `pip install` commands for missing Python packages and instructions for downloading spaCy models.
-   **File Relocation:** Numerous files have been moved to new, more logical directories.
-   **Function/Class Extraction:** Large classes and functions from `video_editing.py` have been extracted into their own dedicated modules for better separation of concerns.
-   **Resource Cleanup:** `video/face_tracking.py` and `video/object_tracking.py` now include explicit `cleanup` methods to release MediaPipe and PyTorch model resources, respectively.
-   **Consistent Imports:** All modules have updated import paths to reflect the new modular structure.
-   **Explicit FFmpeg Path:** `main.py` now explicitly sets `os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"` to ensure MoviePy uses the correct FFmpeg executable.
-   **Pinned Dependencies:** `requirements.txt` now explicitly pins all dependencies to specific versions, enhancing reproducibility and preventing unexpected breaking changes from upstream library updates.
-   **New `.gitignore` Entries:** Added entries for `test_nvenc_output.mp4`, `videos/`, `dummy_video.mp4`, `video.mp4`, `ffmpeg`, and `tests/` to prevent unnecessary files from being tracked by Git.
-   **Centralized Subtitle Fonts:** `core/config.py` now defines `SUBTITLE_FONT_FAMILIES` for consistent subtitle styling.
-   **Dependency Updates:**
    -   `torch`: Pinned to `2.3.0`
    -   `torchvision`: Pinned to `0.18.0`
    -   `torchaudio`: Pinned to `2.3.0`
    -   `pytubefix`: Pinned to `1.0.0`
    -   `pytube3`: Removed
    -   `mediapipe`: Pinned to `0.10.9`
    -   `scikit-learn`: Pinned to `1.2.2`
    -   `librosa`: Pinned to `0.9.2`
    -   `webcolors`: Pinned to `1.13`