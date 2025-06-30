Version 4.0.2 Beta

## Major Architectural Changes

-   **Modularization:** The project structure has been significantly refactored into a more modular design. Core functionalities are now organized into dedicated subdirectories:
    -   `analysis/`: Contains modules for video content analysis and processing reports.
    -   `audio/`: Houses audio-related processing, including transcription and subtitle generation.
    -   `core/`: Stores central configuration, temporary file management, utility functions, and FFmpeg command logging.
    -   `llm/`: Dedicated to Large Language Model (LLM) interactions for clip selection.
    -   `video/`: Contains modules for video-specific processing, such as face tracking, object tracking, frame processing, scene detection, video effects, and video input.
    This refactoring greatly improves code organization, maintainability, and scalability.

## New Features

-   **Command-Line Argument Support:** `main.py` now supports `argparse` for specifying video input (local file path or YouTube URL with quality) directly via command-line arguments, enabling non-interactive execution.
-   **FFmpeg Command Logging:** A new `core/ffmpeg_command_logger.py` module and integration in `video/video_editing.py` allow for capturing and printing the exact FFmpeg commands executed by MoviePy, aiding in debugging and understanding the underlying video processing.
-   **Dynamic Frame Processing:** Introduction of `video/frame_processor.py` centralizes logic for dynamic cropping and face/object-aware framing, ensuring optimal composition of clips.
-   **AV1 Video Conversion:** `core/utils.py` now includes functionality to detect and automatically convert AV1 video streams to H.265 (HEVC) using FFmpeg, improving compatibility with downstream processing.
-   **Centralized Video Encoding Configuration:** `core/config.py` now includes `VIDEO_ENCODER`, `FFMPEG_GLOBAL_PARAMS`, and `FFMPEG_ENCODER_PARAMS`, allowing for easy switching and configuration of different video encoders (e.g., `h264_nvenc`, `libx264`, `hevc_nvenc`, `av1_nvenc`).

## Enhancements & Improvements

-   **Robust JSON Extraction:** `llm/llm_interaction.py` features improved `extract_json_from_text` logic, making it more resilient to varied LLM output formats by attempting direct parsing and then using multiple regex patterns, including fixing trailing commas.
-   **Comprehensive System Checks:** `core/utils.py`'s `system_checks` now performs more thorough validations, including checks for Ollama service and LLM model availability, as well as installations of PyTorch, OpenCV, MediaPipe, and spaCy.
-   **Detailed System Information:** `core/utils.py`'s `print_system_info` provides more extensive system details for debugging, such as OS version, architecture, Python compiler, RAM, and GPU information.
-   **Intelligent Video Input:** `video/video_input.py`'s `choose_input_video` now intelligently detects whether user input is a local file path or a YouTube URL.
-   **YouTube AV1 Stream Filtering:** `video/video_input.py`'s `download_youtube_video` now filters out AV1 streams by default during quality selection, preventing issues with unsupported codecs.
-   **Smoother Frame Transitions:** `video/frame_processor.py` implements a `SMOOTHING_FACTOR` from `core/config.py` for smoother transitions during dynamic framing.
-   **Optimized Tracker Initialization:** `video/video_editing.py`'s `batch_process_with_analysis` now initializes `FaceTracker` and `ObjectTracker` instances once for the entire batch, improving efficiency by avoiding redundant model loading.
-   **Enhanced Error Reporting:** `main.py` now provides more detailed troubleshooting suggestions in case of errors, including specific `pip install` commands for missing Python packages and instructions for downloading spaCy models.

## Refactoring & Code Quality

-   **File Relocation:** Numerous files have been moved to new, more logical directories:
    -   `audio_processing.py` -> `audio/audio_processing.py`
    -   `config.py` -> `core/config.py`
    -   `llm_interaction.py` -> `llm/llm_interaction.py`
    -   `temp_manager.py` -> `core/temp_manager.py`
    -   `utils.py` -> `core/utils.py`
    -   `video_editing.py` -> `video/video_editing.py` (and significantly refactored)
    -   `video_input.py` -> `video/video_input.py`
-   **Function/Class Extraction:** Large classes and functions from `video_editing.py` have been extracted into their own dedicated modules for better separation of concerns:
    -   `analyze_video_content`, `optimize_processing_settings`, `create_processing_report`, `save_processing_report` -> `analysis/analysis_and_reporting.py`
    -   `SubtitleGenerator` class and related functions -> `audio/subtitle_generation.py`
    -   `FaceTracker` class -> `video/face_tracking.py`
    -   `ObjectTracker` class -> `video/object_tracking.py`
    -   `SceneDetector` class -> `video/scene_detection.py`
    -   `VideoEffects` class -> `video/video_effects.py`
-   **Resource Cleanup:** `video/face_tracking.py` and `video/object_tracking.py` now include explicit `cleanup` methods to release MediaPipe and PyTorch model resources, respectively.
-   **Consistent Imports:** All modules have updated import paths to reflect the new modular structure.
-   **Explicit FFmpeg Path:** `main.py` now explicitly sets `os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"` to ensure MoviePy uses the correct FFmpeg executable.

## Configuration Changes

-   **Pinned Dependencies:** `requirements.txt` now explicitly pins all dependencies to specific versions, enhancing reproducibility and preventing unexpected breaking changes from upstream library updates.
-   **New `.gitignore` Entries:** Added entries for `test_nvenc_output.mp4`, `videos/`, `dummy_video.mp4`, `video.mp4`, `ffmpeg`, and `tests/` to prevent unnecessary files from being tracked by Git.
-   **Centralized Subtitle Fonts:** `core/config.py` now defines `SUBTITLE_FONT_FAMILIES` for consistent subtitle styling.

## Dependency Updates

-   `torch`: Pinned to `2.3.0`
-   `torchvision`: Pinned to `0.18.0`
-   `torchaudio`: Pinned to `2.3.0`
-   `pytubefix`: Pinned to `1.0.0`
-   `pytube3`: Removed
-   `mediapipe`: Pinned to `0.10.9`
-   `scikit-learn`: Pinned to `1.2.2`
-   `librosa`: Pinned to `0.9.2`
-   `webcolors`: Pinned to `1.13`

## Bug Fixes

-   Addressed potential issues with MoviePy not finding the FFmpeg executable by explicitly setting the environment variable in `main.py`.
-   Improved robustness of JSON parsing from LLM output in `llm/llm_interaction.py` to handle various formatting inconsistencies.