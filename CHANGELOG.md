# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
