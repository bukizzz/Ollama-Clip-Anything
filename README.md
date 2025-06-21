# Video Segment Extraction with LLM-Assisted Transcription

---

## Overview

This script automates the extraction of relevant video segments based on transcription analysis enhanced by a Large Language Model (LLM). It performs:

- Audio extraction from video  
- Speech transcription with OpenAI Whisper  
- Multi-stage LLM processing to identify key segments  
- Video editing to extract and concatenate clips with fade effects  

---

## Installation Instructions

### Step 1: Prepare Project Directory

Place the script file(s) into your working folder or clone this repo.

```bash
git clone https://github.com/bukizzz/Ollama-Clip-Anything.git
```

---

### Step 2: Create and Activate Python Virtual Environment (Recommended)

- Linux/macOS:

```bash
python3 -m venv venv  
source venv/bin/activate
```

- Windows (PowerShell):

```bash
python -m venv venv  
.\venv\Scripts\Activate.ps1
```

---

### Step 3: Install Dependencies

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

---

### Step 4: Verify FFmpeg Installation

Ensure FFmpeg is installed and available in your system PATH:

```bash
ffmpeg -version
```

If not installed:

- Ubuntu/Debian:

```bash
sudo apt-get update  
sudo apt-get install ffmpeg
```

- macOS (Homebrew):

```bash
brew install ffmpeg
```

- Windows:

Download from https://ffmpeg.org/download.html and add its `bin` folder to your system PATH.

---

## Usage Manual

1. Activate your virtual environment:

- Linux/macOS:

```bash
source venv/bin/activate
```

- Windows (PowerShell):

```bash
.env\Scripts\Activate.ps1
```

2. Place the input video file as `input_video.mp4` in the working directory or adjust `input_video` path in the script.

3. Customize the `user_query` variable in the `main()` function to specify segment extraction criteria.  
Example:

```python
user_query = "5-10min long part. output start, end, summary. no other output"
```

4. Run the script:

```bash
python app.py
```

5. The output video will be saved as `edited_output.mp4` (modifiable in `main()`).

---

## Core Functional Components

- Audio extraction: Uses FFmpeg to convert video audio to a mono 16kHz MP3 for transcription.  
- Transcription: Runs Whisper speech-to-text model producing timestamped segments.  
- LLM Processing: Runs multi-pass Ollama LLM prompts to extract, clean, and validate JSON segments matching user query, with retry logic.  
- Video Editing: Extracts clips corresponding to segments, applies fade-in/out effects, concatenates, and encodes final output.

---

## Customization Options

- Change Whisper model in `transcribe_video()` (e.g., "base", "small", "medium", "large").  
- Change LLM model string in multi-stage functions (default "qwen2.5-coder:7b").  
- Modify `user_query` for different segment extraction goals.  
- Adjust retry parameters `max_retries` and `retry_delay` in `get_relevant_segments_multistage_with_retry()`.  
- Rename input/output video paths in `main()`.

---

## Troubleshooting

- FFmpeg issues: Confirm installation and PATH inclusion.  
- Whisper load errors: Verify correct PyTorch installation and CUDA drivers if using GPU.  
- Ollama errors: Ensure Ollama server is running and models are pulled.  
- JSON parse failures: Ensure `json5` package is installed; check raw LLM output logs.  
- Memory constraints: Monitor GPU/CPU usage; reduce video length or model size if needed.

---

## Logging

The script provides verbose console logs at each step, including:

- FFmpeg command execution  
- Transcription progress and memory cleanup  
- LLM prompt passes and raw output  
- JSON extraction and sanitation results  
- Video clip extraction timing and errors  
- Retry attempts and final status

Use logs for debugging and performance insights.

---

## Extension Ideas

- Add CLI arguments or config file support for flexible input/output and query parameters.  
- Batch process multiple videos in parallel.  
- Support other video/audio formats or enhanced audio extraction options.  
- Overlay subtitles or other metadata onto output clips.  
- Integrate progress bars or GUI front-end.
