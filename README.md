# Ollama-Clip-Anything

## Overview

This program is a sophisticated, multi-agent system that automates the creation of engaging, short-form video clips from longer source videos. It leverages a suite of AI-powered agents to perform a comprehensive analysis of video and audio content, enabling intelligent editing decisions that go far beyond simple content selection. The system can process local video files or YouTube URLs, and it produces professionally styled clips with dynamic framing, animated subtitles, and synchronized background music.

## Pipeline Overview

The video processing pipeline is orchestrated by an `AgentManager` that executes a series of specialized agents in a specific order. Each agent is responsible for a distinct aspect of the analysis and content creation process, ensuring a modular and extensible architecture. The key agents in the pipeline include:

- **Video Input Agent:** Handles the initial video input, whether from a local file or a YouTube URL.
- **Frame Preprocessing Agent:** Decomposes the video into frames for analysis by vision models.
- **Qwen-VL Integration Agent:** Utilizes the Qwen-VL model for advanced video understanding, including object localization and scene description.
- **Audio Transcription Agent:** Transcribes the audio content of the video.
- **Audio Rhythm Analysis Agent:** Analyzes the audio to detect tempo, beats, and speech rhythm patterns.
- **Enhanced Audio Analysis Agent:** Performs speaker diarization, sentiment analysis, and theme identification.
- **Enhanced Video Analysis Agent:** Analyzes video frames for facial expressions, gestures, visual complexity, and energy levels.
- **Engagement Analysis Agent:** Scores video segments for their engagement potential based on a combination of audio and video cues.
- **Layout Detection Agent:** Detects the number of people in a frame, identifies screen sharing, and classifies the overall layout.
- **Storyboarding Agent:** Generates a scene-by-scene storyboard of the video, incorporating multimodal analysis.
- **Content Alignment Agent:** Synchronizes the audio and video tracks, aligning the transcript with visual events.
- **Speaker Tracking Agent:** Identifies and tracks individual speakers throughout the video.
- **Hook Identification Agent:** Pinpoints compelling moments in the video that can serve as hooks to capture viewer attention.
- **LLM Video Director Agent:** Orchestrates the overall editing process, making intelligent decisions about cuts, transitions, and pacing.
- **Improved LLM Selection Agent:** Selects the most engaging and coherent clips based on the comprehensive analysis from all previous agents.
- **Viral Potential Agent:** Scores the selected clips for their potential to go viral on social media.
- **Dynamic Editing Agent:** Generates a plan for dynamic editing effects, such as rhythm-based zooms and beat-matched transitions.
- **Music Sync Agent:** Selects and synchronizes background music with the video content.
- **Layout Optimization Agent:** Determines the optimal visual layout for each clip based on the content and number of speakers.
- **Subtitle Animation Agent:** Generates animated, word-by-word subtitles with advanced styling.
- **B-roll Integration Agent:** Suggests and integrates relevant B-roll footage to enhance the visual narrative.
- **Content Enhancement Pipeline:** A final integration layer that coordinates all agents and ensures a high-quality output.
- **Final Quality Assurance:** Performs a final check of the generated clips to ensure they meet quality standards.

## Features

- **Intelligent Content Selection:** Uses a multi-pass LLM analysis to select the most engaging and coherent clips.
- **Dynamic Framing and Cropping:** Automatically converts videos to a 9:16 aspect ratio with intelligent cropping that follows the action.
- **Face and Object Tracking:** Keeps the focus on the main subjects of the video.
- **Rhythm-Based Editing:** Synchronizes cuts and effects with the rhythm of the audio.
- **Beat-Synced Effects:** Adds dynamic zoom and other effects that are synchronized with the beat of the music.
- **Advanced Layout Management:** Automatically selects the optimal visual layout for each scene, including single-person, multi-person, and presentation modes.
- **Animated Subtitles:** Generates professional-quality, word-by-word animated subtitles.
- **Speaker Detection and Tracking:** Identifies individual speakers and can apply speaker-specific styling.
- **Music Synchronization:** Automatically selects and synchronizes background music with the video content.
- **B-roll Integration:** Intelligently suggests and integrates relevant B-roll footage.
- **Comprehensive System Checks:** Verifies that all dependencies are installed and configured correctly.
- **Robust Error Handling:** Includes detailed error reporting and troubleshooting suggestions.

## Configuration

The application's behavior can be customized through the `core/config.yaml` file. This file allows you to configure a wide range of settings, including:

- **Model Selection:** Specify which models to use for transcription, language understanding, and image recognition.
- **Subtitle Styling:** Customize the font, size, color, and animation of the subtitles.
- **Video Processing:** Adjust the desired clip duration, video encoder, and other processing parameters.
- **Agent Configuration:** Enable or disable individual agents in the pipeline.
- **And much more...**

For a complete list of all available configuration options, please refer to the `core/config.yaml` file.

## Installation and Usage

### 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Python 3.12+**: You can download it from the [official Python website](https://www.python.org/downloads/).
*   **FFmpeg**: This is a crucial dependency for video and audio processing. You can download it from the [official FFmpeg website](https://ffmpeg.org/download.html) or install it using your system's package manager (e.g., `sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS).
*   **Git**: You'll need Git to clone the repository. You can download it from the [official Git website](https://git-scm.com/downloads).
*   **Ollama**: This project uses Ollama to run the large language models locally. You can download it from the [official Ollama website](https://ollama.com/).

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bukizzz/Ollama-Clip-Anything.git
    cd Ollama-Clip-Anything
    ```

2.  **Create a virtual environment:**
    It is highly recommended to use a virtual environment to manage the project's dependencies.
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the required models:**
    This project uses several models for its various agents. You can download them using the provided script:
    ```bash
    python tools/download_models.py
    ```
    This will download the necessary models from Ollama and other sources.

### 3. Configuration

1.  **Configure Ollama:**
    Make sure the Ollama application is running. You can start it by simply running `ollama serve` in your terminal.

2.  **Configure the application:**
    The main configuration file for the application is `core/config.yaml`. You can customize various settings in this file, such as the models to use, subtitle styles, and more.

### 4. Usage

You can run the application from the command line using `main.py`. Here are the available arguments:

*   `--video_path`: Path to a local MP4 video file.
*   `--youtube_url`: URL of a YouTube video to download.
*   `--youtube_quality`: Desired YouTube video quality option (e.g., 0, 1, 2...).
*   `--user_prompt`: Optional: A specific prompt for the LLM to guide clip selection.
*   `--retry`: Automatically resume from a previous failed session.
*   `--nocache`: Force a fresh start, deleting any existing state and temporary files.

**Example usage:**

*   **Processing a local video file:**
    ```bash
    python main.py --video_path /path/to/your/video.mp4
    ```

*   **Processing a YouTube video:**
    ```bash
    python main.py --youtube_url https://www.youtube.com/watch?v=your_video_id --youtube_quality 1
    ```

*   **Using a custom prompt:**
    ```bash
    python main.py --video_path /path/to/your/video.mp4 --user_prompt "Find the most exciting moments in the video."
    ```
    
*   **Using the interactive menu**
    Run the app without --retry to reach the interactive menu.
    
