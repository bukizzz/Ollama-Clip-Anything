# Overview

This program automatically extracts engaging 60-second clips from longer videos (local files or YouTube URLs) with intelligent content selection, automatic face tracking, and professional subtitle generation. It's designed for content creators who want to repurpose long-form videos into short, shareable clips.

## Installation Guide

### Prerequisites

- Python 3.11 (recommended) or later
- FFmpeg (must be installed and in system PATH)
- NVIDIA GPU (optional but recommended for faster processing)

### Step-by-Step Installation

#### 1. Install Python 3.11

**Windows:**
- Download Python 3.11 installer from python.org
- Run installer
- Check "Add Python to PATH" during installation
- Verify installation:
  ```bash
  python --version
  # should show 3.11.x
  ```

**macOS:**
```bash
brew install python@3.11
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

#### 2. Create and Activate Virtual Environment

```bash
python3.11 -m venv clipgen_env
source clipgen_env/bin/activate  # Linux/macOS
clipgen_env\Scripts\activate     # Windows
```

#### 3. Install FFmpeg

**Windows:**
- Download from ffmpeg.org
- Extract and add bin folder to system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

#### 4. Install Requirements

```bash
pip install -r requirements.txt
```

#### 5. Install Ollama (for AI clip selection)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b
```

## Usage Instructions

### Basic Usage

1. Run the program:
   ```bash
   python main.py
   ```

2. Choose input method:
   - Local MP4 file (enter path or drag file to terminal)
   - YouTube URL (enter URL and select quality)

3. The program will:
   - Transcribe the video
   - Analyze content for engaging segments
   - Create 60-second clips with face tracking
   - Add synchronized subtitles
   - Save clips to organized folders

### Advanced Options

To customize processing, you can modify these parameters in the code:
- **Clip duration:** Change the `45 <= duration <= 90` range in `sanitize_segments()`
- **Video quality:** Adjust NVENC/CPU encoding settings in the encoding functions
- **Subtitle style:** Modify the style string in `create_individual_clip()`

### Output Structure

Clips are saved in:
```
videos/
└── source_video_name_1234/
    ├── clip_batch1_1.mp4
    ├── clip_batch1_2.mp4
    └── ...
```

## Features

### 1. Flexible Video Input
- **Local files:** Directly process MP4 files
- **YouTube downloads:** Automatically download and process YouTube videos with quality selection
- **Adaptive streams:** Handles both progressive and adaptive YouTube streams

### 2. AI-Powered Content Selection
- **Three-pass LLM analysis:**
  - Initial content identification
  - JSON structure conversion
  - Final validation and cleanup
- **Context-aware selection:** Chooses complete thoughts/stories rather than arbitrary segments
- **Retry logic:** Automatically retries failed analyses

### 3. Professional Video Processing
- **Smart cropping:** Automatic 9:16 aspect ratio conversion
- **Face tracking:** Dynamic crop positioning based on face detection
- **Multiple encoding methods** (with automatic fallback):
  - NVENC (GPU accelerated)
  - CPU encoding
  - Safe mode (ultra-compatible)
- **Audio normalization:** Consistent volume levels across clips

### 4. Subtitle Generation
- **Precise synchronization:** Compensates for processing delays
- **Automatic wrapping:** Formats text for optimal readability
- **Professional styling:** Clean, readable subtitles with shadow/outline

### 5. Robust Error Handling
- **Comprehensive validation:** Checks at every processing stage
- **Automatic cleanup:** Removes temporary files
- **Detailed error reporting:** Helps troubleshoot issues

### 6. Performance Optimization
- **GPU acceleration:** Uses CUDA when available
- **Memory management:** Properly unloads models
- **Disk space monitoring:** Warns about low space

## Troubleshooting

### Common Issues

**FFmpeg not found:**
- Verify FFmpeg is installed and in PATH
- Test with:
  ```bash
  ffmpeg -version
  ```

**CUDA errors:**
- Ensure you have compatible NVIDIA drivers
- Try:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```

**YouTube download failures:**
- Try a different video quality
- Check your internet connection

**Face tracking not working:**
- Ensure OpenCV is properly installed
- Try well-lit videos with clear faces

## Getting Help

For additional support, please open an issue on GitHub with:
- The exact error message
- Your system specifications
- The video you were processing (if possible)

## License

This project is licensed under the Source First License - see the LICENSE file for details.