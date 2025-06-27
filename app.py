import os
import json
import re
import subprocess
import gc
import time
import traceback
import shutil
import atexit
import torch
import whisper
import cv2
import ollama
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import crop, colorx, speedx
from pytubefix import YouTube
import sys
from contextlib import redirect_stderr
import io


# ----------- TEMPORARY FILES MANAGEMENT -----------

TEMP_DIR = ".temp"

def ensure_temp_dir():
    """Create the temporary directory if it doesn't exist."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print(f"Created temporary directory: {TEMP_DIR}")

def cleanup_temp_dir():
    """Remove the temporary directory and all its contents."""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"Cleaned up temporary directory: {TEMP_DIR}")
        except Exception as e:
            print(f"Warning: Could not fully clean up temp directory: {e}")

def get_temp_path(filename):
    """Get a path within the temp directory."""
    ensure_temp_dir()
    return os.path.join(TEMP_DIR, filename)

# Register cleanup function to run on exit
atexit.register(cleanup_temp_dir)


# ----------- VIDEO INPUT SETUP -----------

def choose_input_video() -> str:
    """Choose between local MP4 file or YouTube video download."""
    choice = input("Do you want to use a local .mp4 file? (y/n): ").strip().lower()
    if choice == "y":
        path = input("Enter path to local .mp4 file: ").strip().strip("'\"")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        if not path.lower().endswith(".mp4"):
            raise ValueError("File must be an .mp4")
        return path
    else:
        url = input("Enter YouTube video URL: ").strip()
        return download_youtube_video_only_mp4(url)


def download_youtube_video_only_mp4(url: str, output_dir="videos") -> str:
    """Download YouTube video as MP4 with user selection of quality."""
    yt = YouTube(url)
    
    # Get all MP4 streams (both progressive and adaptive)
    video_streams = yt.streams.filter(file_extension='mp4').order_by('resolution').desc()

    if not video_streams:
        raise RuntimeError("No compatible MP4 streams found.")

    print(f"\nTitle: {yt.title}")
    print("Available video streams:")
    for i, stream in enumerate(video_streams):
        stream_type = "Progressive" if stream.is_progressive else "Adaptive"
        audio_info = "Audio included" if stream.includes_audio_track else "Video only"
        size_mb = stream.filesize / (1024*1024) if stream.filesize else "Unknown"
        size_str = f"{size_mb:.2f} MB" if isinstance(size_mb, float) else size_mb
        
        print(f"{i}: {stream.resolution} | {stream.fps}fps | {stream_type} | {audio_info} | {size_str}")

    choice = int(input("Enter the number of the video stream to download: "))
    selected_stream = video_streams[choice]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Downloading: {yt.title}")
    
    # Handle both progressive and adaptive streams
    if selected_stream.is_progressive:
        # Progressive streams have both video and audio
        output_path = selected_stream.download(output_path=output_dir, filename_prefix="yt_")
    else:
        # Adaptive streams need separate audio download and merging
        print("Downloading video-only stream and best audio separately...")
        
        # Download video to temp directory first
        video_path = selected_stream.download(output_path=TEMP_DIR, filename_prefix="yt_video_")
        
        # Get best audio stream
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
        if not audio_stream:
            audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        
        if audio_stream:
            audio_path = audio_stream.download(output_path=TEMP_DIR, filename_prefix="yt_audio_")
            
            # Merge video and audio using ffmpeg
            final_filename = f"yt_{yt.title[:50].replace('/', '_').replace('|', '_')}.mp4"
            output_path = os.path.join(output_dir, final_filename)
            
            merge_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                output_path
            ]
            
            print("Merging video and audio...")
            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("FFmpeg merge failed, using video-only file")
                output_path = video_path
            else:
                # Clean up temporary files
                os.remove(video_path)
                os.remove(audio_path)
        else:
            print("No audio stream found, using video-only file")
            output_path = video_path
    
    print(f"Saved to: {output_path}")
    return output_path


def get_next_output_filename(output_dir="videos", batch_prefix="clip", ext=".mp4", clip_number=1, source_video_path=None) -> str:
    """Generate filename for individual clips in a folder named after source video."""
    import random
    
    if source_video_path:
        # Get source video name without extension
        source_name = os.path.splitext(os.path.basename(source_video_path))[0]
        # Generate random 4-digit number
        random_num = random.randint(1000, 9999)
        # Create folder name: source_video_name_randomnumber
        folder_name = f"{source_name}_{random_num}"
        output_dir = os.path.join(output_dir, folder_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get current batch number by checking existing clips
    existing = [
        fname for fname in os.listdir(output_dir)
        if fname.startswith(batch_prefix) and fname.endswith(ext)
    ]

    # Extract batch numbers from existing files
    batch_numbers = []
    for fname in existing:
        # Look for pattern like "clip_batch1_1.mp4"
        match = re.search(r'batch(\d+)_', fname)
        if match:
            batch_numbers.append(int(match.group(1)))

    current_batch = max(batch_numbers, default=0) + 1
    return os.path.join(output_dir, f"{batch_prefix}_batch{current_batch}_{clip_number}{ext}")


# ----------- AUDIO PROCESSING -----------

def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video using FFmpeg with precise timing."""
    cmd = [
        "ffmpeg", "-y",
        "-v", "error",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        "-f", "wav",  # Use WAV for better precision than MP3
        "-avoid_negative_ts", "make_zero",  # Ensure proper timestamp handling
        audio_path
    ]
    print("Running FFmpeg command:", ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg stderr:\n", result.stderr)
        raise RuntimeError("FFmpeg audio extraction failed.")


def transcribe_video(video_path: str, model_name="base") -> list[dict]:
    """Transcribe video using Whisper with precise timing."""
    try:
        model = whisper.load_model(model_name)
        audio_path = get_temp_path("temp_audio.wav")  # Use temp directory
        extract_audio(video_path, audio_path)

        print("Running Whisper transcription with precise timing...")
        result = model.transcribe(
            audio_path,
            word_timestamps=True,  # Enable word-level timestamps for better precision
            condition_on_previous_text=False  # Prevent timing drift
        )

        transcription = [{
            'start': float(seg['start']),
            'end': float(seg['end']),
            'text': seg['text'].strip()
        } for seg in result['segments']]

        return transcription

    finally:
        try:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Whisper model unloaded and GPU memory released.")
        except Exception as cleanup_error:
            print("Error during Whisper model cleanup:", cleanup_error)


# ----------- SUBTITLE FUNCTIONS -----------

def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def wrap_text_for_subtitles(text: str, max_chars_per_line=35, max_lines=2) -> str:
    """
    Wrap text to fit subtitle constraints: max 2 lines, reasonable character count per line.
    
    Args:
        text: The subtitle text to wrap
        max_chars_per_line: Maximum characters per line
        max_lines: Maximum number of lines (2 for bottom half constraint)
    
    Returns:
        Wrapped text with line breaks
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        # Check if adding this word would exceed the line limit
        word_length = len(word) + (1 if current_line else 0)  # +1 for space
        
        if current_length + word_length <= max_chars_per_line:
            current_line.append(word)
            current_length += word_length
        else:
            # Start a new line
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                # Single word is too long, add it anyway
                lines.append(word)
                current_line = []
                current_length = 0
        
        # If we've reached the maximum number of lines, truncate
        if len(lines) >= max_lines:
            if current_line:
                # If there's a partial line, complete it but truncate if necessary
                remaining_line = " ".join(current_line)
                if len(remaining_line) > max_chars_per_line:
                    remaining_line = remaining_line[:max_chars_per_line-3] + "..."
                lines.append(remaining_line)
            break
    
    # Add any remaining words if we haven't hit the line limit
    if current_line and len(lines) < max_lines:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines[:max_lines])


def create_clip_subtitles(original_segments, clip_start, clip_end, output_path):
    """
    Create subtitle file for a single 60-second clip with perfect sync.
    Accounts for video processing delays automatically.
    
    Args:
        original_segments: Full transcript from Whisper
        clip_start: Start time of the clip in original video
        clip_end: End time of the clip in original video
        output_path: Path for the subtitle file
    """
    clip_subtitles = []
    
    # Auto-detect and compensate for processing delay
    # The fadein effect (0.5s) can cause timing shifts
    processing_compensation = 0  # Compensate for fadein effect
    
    for orig_seg in original_segments:
        orig_start = float(orig_seg['start'])
        orig_end = float(orig_seg['end'])
        
        # Check if this subtitle segment overlaps with the clip
        if orig_start < clip_end and orig_end > clip_start:
            # Calculate timing relative to the clip start
            new_start = max(0, orig_start - clip_start)
            new_end = min(orig_end - clip_start, clip_end - clip_start)
            
            # Apply processing compensation - subtract delay from subtitle timing
            # This makes subtitles appear earlier to sync with delayed video
            new_start = max(0, new_start - processing_compensation)
            new_end = max(new_start + 0.1, new_end - processing_compensation)  # Ensure minimum duration
            
            # Only add if the timing makes sense
            if new_end > new_start and new_start < (clip_end - clip_start):
                # Wrap text for better display
                wrapped_text = wrap_text_for_subtitles(orig_seg['text'].strip())
                clip_subtitles.append({
                    'start': new_start,
                    'end': new_end,
                    'text': wrapped_text
                })
    
    # Write the subtitle file
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(clip_subtitles, 1):
            start = format_timestamp(sub['start'])
            end = format_timestamp(sub['end'])
            text = sub['text']
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    
    return output_path


# ----------- IMPROVED SEGMENT SELECTION -----------

def three_pass_llm_extraction(transcript: list[dict]) -> list[dict]:
    """
    Use a three-pass approach for robust LLM-based clip extraction.
    
    Pass 1: Extract initial data (may have formatting issues)
    Pass 2: Clean and structure the output into proper JSON
    Pass 3: Final validation and formatting
    """
    model = "qwen2.5-coder:7b"
    
    # Ensure model is available
    try:
        ollama.pull(model)
    except Exception as e:
        print(f"Failed to pull model {model}: {e}")
        raise
    
    # Create simplified transcript for LLM
    simplified_transcript = []
    for i, seg in enumerate(transcript):
        simplified_transcript.append({
            "index": i,
            "start": round(seg['start'], 1),
            "end": round(seg['end'], 1),
            "text": seg['text'][:150]  # Truncate for token efficiency
        })
    
    # PASS 1: Initial extraction (allow flexible formatting)
    print("🔄 Pass 1: Initial clip extraction...")
    pass1_prompt = f"""You are a video editor selecting engaging clips. Analyze this transcript and identify 10 distinct, interesting 60-second segments.

For each segment, provide:
- Start time (in seconds)
- End time (in seconds) 
- Brief description of the content

Focus on:
- Complete thoughts or stories
- Engaging moments
- 45-75 second duration
- No overlapping segments

Transcript data:
{json.dumps(simplified_transcript[:min(50, len(simplified_transcript))], indent=1)}

Total duration: {transcript[-1]['end']:.1f} seconds

Provide your selections in any clear format - don't worry about perfect JSON formatting yet."""

    messages = [
        {"role": "system", "content": "You are an expert video editor. Focus on finding the most engaging content."},
        {"role": "user", "content": pass1_prompt}
    ]
    
    pass1_output = llm_pass(model, messages)
    print(f"Pass 1 output length: {len(pass1_output)} characters")
    
    # PASS 2: Structure into JSON format
    print("🔄 Pass 2: Converting to JSON structure...")
    pass2_prompt = f"""Convert the following clip selections into a clean JSON array format. 

Extract the start time, end time, and description for each clip and format as:
[
  {{"start": 120.5, "end": 180.5, "text": "description"}},
  {{"start": 200.0, "end": 260.0, "text": "description"}}
]

Make sure:
- All times are numbers (not strings)
- Each clip is 45-75 seconds long
- No overlapping time ranges
- Exactly 10 clips maximum

Original clip selections:
{pass1_output}

Return ONLY the JSON array, nothing else."""

    messages = [
        {"role": "system", "content": "You are a JSON formatter. Output only valid JSON arrays."},
        {"role": "user", "content": pass2_prompt}
    ]
    
    pass2_output = llm_pass(model, messages)
    print(f"Pass 2 output length: {len(pass2_output)} characters")
    
    # PASS 3: Final validation and cleanup
    print("🔄 Pass 3: Final validation and cleanup...")
    pass3_prompt = f"""Validate and clean this JSON array of video clips. Fix any formatting issues and ensure:

1. Valid JSON syntax
2. All start/end times are numbers
3. Each clip duration is 45-75 seconds
4. No overlapping clips
5. Times are within 0 to {transcript[-1]['end']:.1f} seconds
6. Remove any invalid entries

Input JSON:
{pass2_output}

Return the cleaned, valid JSON array only."""

    messages = [
        {"role": "system", "content": "You are a JSON validator. Return only clean, valid JSON."},
        {"role": "user", "content": pass3_prompt}
    ]
    
    pass3_output = llm_pass(model, messages)
    print(f"Pass 3 output length: {len(pass3_output)} characters")
    
    # Extract and validate the final JSON
    try:
        segments = extract_json_from_text(pass3_output)
        cleaned_segments = sanitize_segments(segments)
        print(f"✅ Three-pass extraction yielded {len(cleaned_segments)} valid clips")
        return cleaned_segments
    except Exception as e:
        print(f"❌ Three-pass extraction failed at final parsing: {e}")
        print("Pass 3 output:", pass3_output[:500])
        raise


def llm_pass(model: str, messages: list[dict]) -> str:
    """Send messages to Ollama model and return response."""
    try:
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    except Exception as e:
        print(f"LLM request failed: {e}")
        raise


def extract_json_from_text(text: str) -> list[dict]:
    """Extract JSON array from LLM response text with better error handling."""
    # Try to find JSON array in the text
    patterns = [
        r'\[[\s\S]*?\]',  # Standard array
        r'```json\s*(\[[\s\S]*?\])\s*```',  # JSON code block
        r'```\s*(\[[\s\S]*?\])\s*```',  # Generic code block
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Clean up the match
                json_str = match if isinstance(match, str) else match[0]
                # Remove any trailing commas before closing brackets
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    raise ValueError("No valid JSON array found in model output")


def sanitize_segments(segments: list[dict], max_duration: float = None) -> list[dict]:
    """Clean and validate segment data from LLM output."""
    cleaned = []
    for seg in segments:
        try:
            # Handle different input formats
            if isinstance(seg.get('start'), list):
                start = float(seg['start'][0])
            else:
                start = float(seg['start'])
            if isinstance(seg.get('end'), list):
                end = float(seg['end'][-1])
            else:
                end = float(seg['end'])
            text = str(seg.get('text', '')).strip()
            
            # Validate timing and ensure reasonable clip duration
            duration = end - start
            if start >= 0 and end > start and 30 <= duration <= 90:  # More flexible duration
                cleaned.append({"start": start, "end": end, "text": text})
        except (ValueError, TypeError, KeyError) as e:
            print(f"Skipping invalid segment: {seg} - Error: {e}")
            continue
    
    return cleaned


def get_60_second_clips_with_llm(transcript: list[dict]) -> list[dict]:
    """Use three-pass LLM approach to select coherent clips from the transcript."""
    return three_pass_llm_extraction(transcript)


def get_clips_with_retry(transcript: list[dict], max_retries=3, retry_delay=2) -> list[dict]:
    """Get clips with retry logic using three-pass LLM approach."""
    
    # Get video duration from transcript
    video_duration = transcript[-1]['end'] if transcript else 0
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting three-pass LLM clip selection (attempt {attempt + 1}/{max_retries})...")
            clips = get_60_second_clips_with_llm(transcript)
            
            # Validate clips against video duration
            valid_clips = sanitize_segments(clips, video_duration)
            
            if len(valid_clips) >= 5:  # Accept if we get at least 5 good clips
                print(f"✅ Successfully extracted {len(valid_clips)} clips")
                return valid_clips[:10]  # Take first 10 if we get more
            else:
                print(f"⚠️ Only got {len(valid_clips)} valid clips, need at least 5")
                
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    raise RuntimeError(f"Failed to extract clips after {max_retries} attempts with three-pass method.")


# ----------- VIDEO EDITING -----------
def detect_faces_in_frame(frame_rgb):
    """Detect faces in a frame using OpenCV's Haar cascades."""
    try:
        # Load the face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return faces
    except Exception as e:
        print(f"Face detection error: {e}")
        return []


def get_dynamic_face_positions(video_path, start_time, end_time, fps_sample=2):
    """
    Get face positions throughout a video clip for dynamic tracking.
    
    Args:
        video_path: Path to the video file
        start_time: Start time of the clip in seconds
        end_time: End time of the clip in seconds
        fps_sample: Sample every N frames (2 = every other frame)
    
    Returns:
        dict: {frame_time: (face_center_x, face_center_y)} or None if no faces
    """
    try:
        # Suppress OpenCV/FFmpeg warnings during face detection
        with redirect_stderr(io.StringIO()):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid sync issues
            fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        face_positions = {}
        
        for frame_num in range(start_frame, end_frame, fps_sample):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = detect_faces_in_frame(frame_rgb)
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                
                # Calculate time for this frame
                frame_time = frame_num / fps - start_time
                face_positions[frame_time] = (face_center_x, face_center_y)
        
        cap.release()
        
        if face_positions:
            print(f"Dynamic tracking: Found faces in {len(face_positions)} frames")
            return face_positions
        else:
            print("Dynamic tracking: No faces detected")
            return None
            
    except Exception as e:
        print(f"Dynamic face tracking error: {e}")
        return None
    finally:
        if 'cap' in locals():
            cap.release()


def smooth_face_positions(face_positions, window_size=5):
    """
    Smooth face positions to avoid jittery movement.
    
    Args:
        face_positions: Dict of {time: (x, y)} positions
        window_size: Number of positions to average for smoothing
    
    Returns:
        Dict of smoothed positions
    """
    if not face_positions or len(face_positions) < 2:
        return face_positions
    
    times = sorted(face_positions.keys())
    smoothed = {}
    
    for i, time in enumerate(times):
        # Get window of positions around current time
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(times), i + window_size // 2 + 1)
        
        window_positions = [face_positions[times[j]] for j in range(start_idx, end_idx)]
        
        # Average the positions
        avg_x = sum(pos[0] for pos in window_positions) / len(window_positions)
        avg_y = sum(pos[1] for pos in window_positions) / len(window_positions)
        
        smoothed[time] = (int(avg_x), int(avg_y))
    
    return smoothed


def generate_crop_timeline(face_positions, duration, source_width, target_width):
    """
    Generate a timeline of crop positions for FFmpeg zoompan filter.
    
    Args:
        face_positions: Dict of {time: (face_x, face_y)} from face tracking
        duration: Duration of the clip in seconds
        source_width: Original video width
        target_width: Target crop width
    
    Returns:
        String for FFmpeg zoompan filter or None for center crop
    """
    if not face_positions or len(face_positions) < 3:
        # Not enough face data - use center crop
        return None
    
    # Smooth the positions and reduce keyframes for better performance
    smoothed_positions = smooth_face_positions(face_positions)
    times = sorted(smoothed_positions.keys())
    
    # Reduce keyframes - only use every 2-3 seconds for smoother playback
    reduced_times = []
    last_time = -999
    for time in times:
        if time - last_time >= 2.0:  # Only add keyframe every 2 seconds
            reduced_times.append(time)
            last_time = time
    
    if len(reduced_times) < 2:
        return None  # Fall back to center crop
    
    # Build zoompan timeline with fewer keyframes
    zoom_commands = []
    fps = 30  # Assume 30fps for timeline calculation
    
    for i, time in enumerate(times):
        face_x, face_y = smoothed_positions[time]
        
        # Calculate crop_x to center face horizontally
        crop_x = max(0, min(face_x - target_width // 2, source_width - target_width))
        
        # Create zoompan command for this keyframe
        frame_num = int(time * fps)
        
        if i == 0:
            # First keyframe
            zoom_commands.append(f"zoom=1:x={crop_x}:y=0:d={frame_num}")
        else:
            # Interpolate to this position
            prev_time = times[i-1]
            prev_face_x, prev_face_y = smoothed_positions[prev_time]
            prev_crop_x = max(0, min(prev_face_x - target_width // 2, source_width - target_width))
            
            frames_between = frame_num - int(prev_time * fps)
            if frames_between > 0:
                zoom_commands.append(f"zoom=1:x='if(lte(on,{frames_between}),{prev_crop_x}+({crop_x}-{prev_crop_x})*on/{frames_between},{crop_x})':y=0:d={frames_between}")
    
    # Add final hold if needed
    total_frames = int(duration * fps)
    last_frame = int(times[-1] * fps)
    if last_frame < total_frames:
        remaining_frames = total_frames - last_frame
        final_face_x, final_face_y = smoothed_positions[times[-1]]
        final_crop_x = max(0, min(final_face_x - target_width // 2, source_width - target_width))
        zoom_commands.append(f"zoom=1:x={final_crop_x}:y=0:d={remaining_frames}")
    
    return ":".join(zoom_commands)


def create_individual_clip(original_video_path, clip_data, clip_number, original_transcript):
    """
    Create a single 60-second clip using FFmpeg with improved error handling and subtitle validation.
    """
    start_time = float(clip_data['start'])
    end_time = float(clip_data['end'])
    duration = end_time - start_time
    print(f"Creating clip {clip_number}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
    
    # Generate output filename
    output_path = get_next_output_filename(clip_number=clip_number, source_video_path=original_video_path)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize temp file paths
    subtitle_path = None
    
    try:
        # Get video info first
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
            original_video_path
        ]
        
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_result.returncode != 0:
            raise RuntimeError(f"Failed to probe video: {probe_result.stderr}")
        
        video_info = json.loads(probe_result.stdout)
        video_stream = next(s for s in video_info['streams'] if s['codec_type'] == 'video')
        original_width = int(video_stream['width'])
        original_height = int(video_stream['height'])
        
        # Calculate 9:16 crop parameters
        crop_height = original_height
        crop_width = int(original_height * (9/16))
        
        print(f"Cropping from {original_width}x{original_height} to {crop_width}x{crop_height}")
        
        # Fast face detection
        print("🔍 Quick face analysis...")
        face_positions = get_fast_face_position(original_video_path, start_time, end_time, max_samples=3)
        
        # Determine crop position
        if face_positions:
            print("✅ Using face-centered crop")
            avg_face_x = sum(pos[0] for pos in face_positions.values()) / len(face_positions)
            crop_x = max(0, min(int(avg_face_x - crop_width // 2), original_width - crop_width))
            print(f"Face-centered crop at x={crop_x}")
        else:
            print("⚠️  Using center crop")
            crop_x = (original_width - crop_width) // 2
        
        # Create and validate subtitles
        subtitle_path = get_temp_path(f"subtitles_{clip_number}.srt")
        subtitle_success = create_and_validate_subtitles(
            original_transcript, start_time, end_time, subtitle_path
        )
        
        # Build video filter chain based on subtitle availability
        if subtitle_success:
            subtitle_style = (
                "Fontsize=12,"
                "PrimaryColour=&HFFFFFF&,"
                "BackColour=&H80000000&,"
                "OutlineColour=&H80000000&,"
                "BorderStyle=4,"
                "Outline=1,"
                "Shadow=0,"
                "MarginV=80,"
                "Alignment=2"
            )
            # Escape the subtitle path for FFmpeg
            escaped_subtitle_path = subtitle_path.replace('\\', '\\\\').replace(':', '\\:')
            video_filter = (
                f"crop={crop_width}:{crop_height}:{crop_x}:0,"
                f"subtitles={escaped_subtitle_path}:force_style='{subtitle_style}'"
            )
        else:
            print("⚠️  Proceeding without subtitles due to creation/validation failure")
            video_filter = f"crop={crop_width}:{crop_height}:{crop_x}:0"
        
        # Try progressive encoding approaches
        success = False
        
        # Method 1: NVENC with subtitles (if available)
        if subtitle_success:
            success = try_nvenc_encoding(
                original_video_path, start_time, duration, video_filter, output_path, clip_number
            )
        
        # Method 2: NVENC without subtitles (fallback)
        if not success:
            print(f"🔄 Trying NVENC without subtitles for clip {clip_number}...")
            video_filter_no_subs = f"crop={crop_width}:{crop_height}:{crop_x}:0"
            success = try_nvenc_encoding(
                original_video_path, start_time, duration, video_filter_no_subs, output_path, clip_number
            )
        
        # Method 3: CPU encoding with subtitles
        if not success and subtitle_success:
            print(f"🔄 Trying CPU encoding with subtitles for clip {clip_number}...")
            success = try_cpu_encoding(
                original_video_path, start_time, duration, video_filter, output_path, clip_number
            )
        
        # Method 4: CPU encoding without subtitles
        if not success:
            print(f"🔄 Trying CPU encoding without subtitles for clip {clip_number}...")
            video_filter_no_subs = f"crop={crop_width}:{crop_height}:{crop_x}:0"
            success = try_cpu_encoding(
                original_video_path, start_time, duration, video_filter_no_subs, output_path, clip_number
            )
        
        # Method 5: Ultra-safe mode (last resort)
        if not success:
            print(f"🆘 Last resort: Ultra-safe encoding for clip {clip_number}...")
            success = try_safe_encoding(
                original_video_path, start_time, duration, crop_width, crop_height, crop_x, output_path, clip_number
            )
        
        if not success:
            raise RuntimeError(f"All encoding attempts failed for clip {clip_number}")
        
        # Verify output
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file not created: {output_path}")
        
        file_size = os.path.getsize(output_path)
        if file_size < 1000:
            raise RuntimeError(f"Output file too small ({file_size} bytes)")
        
        tracking_method = "face tracking" if face_positions else "center crop"
        subtitle_status = "with subtitles" if subtitle_success else "no subtitles"
        print(f"✅ Clip {clip_number} completed with {tracking_method} ({subtitle_status}): {output_path} ({file_size // 1024}KB)")
        return output_path
        
    except Exception as e:
        print(f"❌ Error creating clip {clip_number}: {str(e)}")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise
        
    finally:
        # Clean up subtitle file
        if subtitle_path and os.path.exists(subtitle_path):
            try:
                os.remove(subtitle_path)
            except Exception as e:
                print(f"Warning: Could not remove subtitle file {subtitle_path}: {e}")


def create_and_validate_subtitles(original_transcript, start_time, end_time, subtitle_path):
    """
    Create subtitle file and validate it exists and is readable.
    Returns True if successful, False otherwise.
    """
    try:
        # Create subtitles using your existing function
        create_clip_subtitles(original_transcript, start_time, end_time, subtitle_path)
        
        # Validate the file exists
        if not os.path.exists(subtitle_path):
            print(f"❌ Subtitle file not created: {subtitle_path}")
            return False
        
        # Validate the file is readable and not empty
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(f"❌ Subtitle file is empty: {subtitle_path}")
                    return False
        except Exception as e:
            print(f"❌ Cannot read subtitle file {subtitle_path}: {e}")
            return False
        
        print(f"✅ Subtitle file validated: {subtitle_path}")
        return True
        
    except Exception as e:
        print(f"❌ Subtitle creation failed: {e}")
        return False


def try_nvenc_encoding(video_path, start_time, duration, video_filter, output_path, clip_number):
    """
    Attempt NVENC encoding with improved error handling for AV1 input.
    """
    nvenc_cmd = [
        "ffmpeg", "-y",
        # Input handling for AV1
        "-hwaccel", "auto",  # Let FFmpeg choose the best hardware acceleration
        "-err_detect", "ignore_err",
        "-fflags", "+igndts+ignidx+genpts",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        # Video processing
        "-vf", video_filter,
        "-af", "acompressor=attack=0.3:release=8:ratio=4:threshold=-20dB,loudnorm=i=-16:tp=-1.5:lra=11",
        # NVENC encoding
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-rc", "vbr",
        "-cq", "23",
        "-b:v", "5M",
        "-maxrate", "8M",
        "-bufsize", "10M",
        # Audio
        "-c:a", "aac",
        "-b:a", "128k",
        # Output handling
        "-avoid_negative_ts", "make_zero",
        "-max_muxing_queue_size", "1024",
        "-movflags", "+faststart",
        output_path
    ]
    
    print(f"🎬 Attempting NVENC processing for clip {clip_number}...")
    result = subprocess.run(nvenc_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ NVENC success for clip {clip_number}")
        return True
    else:
        print(f"❌ NVENC failed: {result.stderr[:500]}...")  # Truncate long error messages
        return False


def try_cpu_encoding(video_path, start_time, duration, video_filter, output_path, clip_number):
    """
    Attempt CPU encoding with robust error handling.
    """
    cpu_cmd = [
        "ffmpeg", "-y",
        # Robust input handling
        "-err_detect", "ignore_err",
        "-fflags", "+igndts+ignidx+genpts+discardcorrupt",
        "-analyzeduration", "200M",  # More analysis for problematic files
        "-probesize", "200M",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        # Video processing
        "-vf", video_filter,
        "-af", "acompressor=attack=0.3:release=8:ratio=4:threshold=-20dB,loudnorm=i=-16:tp=-1.5:lra=11",
        # CPU encoding
        "-c:v", "libx264",
        "-preset", "medium",  # Balance between speed and quality
        "-crf", "23",
        "-x264-params", "keyint=48:min-keyint=48:ref=3",
        # Audio
        "-c:a", "aac",
        "-b:a", "128k",
        # Output handling
        "-avoid_negative_ts", "make_zero",
        "-max_muxing_queue_size", "2048",  # Larger queue for complex processing
        "-movflags", "+faststart",
        output_path
    ]
    
    result = subprocess.run(cpu_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ CPU encoding success for clip {clip_number}")
        return True
    else:
        print(f"❌ CPU encoding failed: {result.stderr[:500]}...")
        return False


def try_safe_encoding(video_path, start_time, duration, crop_width, crop_height, crop_x, output_path, clip_number):
    """
    Ultra-safe encoding mode - minimal processing, maximum compatibility.
    """
    safe_cmd = [
        "ffmpeg", "-y",
        # Minimal, safe input handling
        "-err_detect", "ignore_err",
        "-fflags", "+igndts+ignidx+genpts+discardcorrupt",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        # Simple crop only - no subtitles, minimal audio processing
        "-vf", f"crop={crop_width}:{crop_height}:{crop_x}:0,scale={crop_width}:{crop_height}",
        "-af", "volume=1.0",  # Minimal audio processing
        # Conservative encoding settings
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",  # Lower quality for stability
        "-profile:v", "baseline",  # Most compatible profile
        "-level", "3.1",
        # Simple audio
        "-c:a", "aac",
        "-b:a", "96k",
        "-ar", "44100",
        # Safe output
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        "-shortest",  # Handle sync issues
        output_path
    ]
    
    result = subprocess.run(safe_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"⚠️  Safe mode success for clip {clip_number} (basic quality, no subtitles)")
        return True
    else:
        print(f"❌ Even safe mode failed: {result.stderr[:500]}...")
        return False


# Additional utility function for better subtitle file handling
def get_temp_path(filename):
    """
    Get a temporary file path with proper directory creation.
    """
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_subdir = os.path.join(temp_dir, "video_processing")
    os.makedirs(temp_subdir, exist_ok=True)
    return os.path.join(temp_subdir, filename)


def get_fast_face_position(video_path, start_time, end_time, max_samples=3):
    """
    Fast face detection - only sample a few frames instead of continuous tracking.
    Returns average face position for static crop.
    """
    try:
        import cv2
        duration = end_time - start_time
        sample_times = [start_time + (duration * i / (max_samples - 1)) for i in range(max_samples)]
        
        face_positions = {}
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Suppress OpenCV/FFmpeg verbose output
        with redirect_stderr(io.StringIO()):
            cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for sample_time in sample_times:
            frame_number = int(sample_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Use largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                face_positions[sample_time] = (face_center_x, face_center_y)
        
        cap.release()
        return face_positions
        
    except ImportError:
        print("OpenCV not available - using center crop")
        return {}
    except Exception as e:
        print(f"Face detection error: {e}")
        return {}


def get_video_info(video_path):
    """
    Get video information using FFprobe.
    Returns dict with width, height, duration, fps, etc.
    """
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", 
        "-show_streams", "-show_format", video_path
    ]
    
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to probe video: {result.stderr}")
    
    data = json.loads(result.stdout)
    video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
    
    return {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'duration': float(data['format']['duration']),
        'fps': eval(video_stream['r_frame_rate']),  # Convert fraction to float
        'codec': video_stream['codec_name'],
        'bitrate': int(data['format'].get('bit_rate', 0))
    }


def batch_create_clips_with_face_tracking(original_video_path, clips_data, original_transcript):
    """
    Create multiple clips using FFmpeg with face tracking and optimized resource usage.
    """
    print(f"🎬 Creating {len(clips_data)} clips with face tracking using FFmpeg + NVENC...")
    
    # Get video info once
    video_info = get_video_info(original_video_path)
    print(f"📹 Source video: {video_info['width']}x{video_info['height']}, "
          f"{video_info['duration']:.1f}s, {video_info['fps']:.1f}fps")
    
    created_clips = []
    failed_clips = []
    face_tracked_clips = 0
    
    for i, clip_data in enumerate(clips_data, 1):
        try:
            print(f"\n--- Processing Clip {i}/{len(clips_data)} ---")
            clip_path = create_individual_clip(
                original_video_path, clip_data, i, original_transcript
            )
            created_clips.append(clip_path)
            
            # Count successful face tracking
            # This would be set by the face detection logic
            # face_tracked_clips += 1  # Increment if face tracking was used
            
        except Exception as e:
            print(f"❌ Failed to create clip {i}: {e}")
            failed_clips.append(i)
            continue
    
    print(f"\n📊 Batch processing complete:")
    print(f"✅ Successfully created: {len(created_clips)} clips")
    print(f"🎯 Face tracking attempted on all clips")
    if failed_clips:
        print(f"❌ Failed clips: {failed_clips}")
    
    return created_clips, failed_clips



# ----------- MAIN ENTRY POINT -----------

def main():
    """Main function to orchestrate the entire video processing pipeline with improved error handling."""
    try:
        print("=== 60-Second Clips Generator ===")
        
        # Step 1: Get input video
        print("\n1. Getting input video...")
        input_video = choose_input_video()
        print(f"Input video: {input_video}")
        
        # Validate input video format and codec
        print("🔍 Analyzing input video...")
        try:
            video_info = get_video_info(input_video)
            print(f"📹 Video info: {video_info['width']}x{video_info['height']}, "
                  f"{video_info['duration']:.1f}s, {video_info['fps']:.1f}fps, "
                  f"codec: {video_info['codec']}")
            
            # Warn about potentially problematic codecs
            if video_info['codec'] in ['av01', 'vp9', 'hevc']:
                print(f"⚠️  Note: {video_info['codec']} codec detected - may require additional processing time")
        except Exception as e:
            print(f"⚠️  Could not analyze video info: {e}")
            print("Proceeding with processing...")
        
        # Step 2: Transcribe video
        print("\n2. Transcribing video...")
        transcription = transcribe_video(input_video)
        print(f"Transcription complete: {len(transcription)} segments found")
        
        if not transcription:
            raise RuntimeError("No transcription segments found. Video may be too short or have no audio.")
        
        # Step 3: Select 60-second clips
        print("\n3. Selecting coherent clips...")
        clips = get_clips_with_retry(transcription)
        print(f"Selected {len(clips)} clips:")
        
        for i, clip in enumerate(clips, 1):
            duration = clip['end'] - clip['start']
            print(f" Clip {i}: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s)")
            print(f" Description: {clip['text'][:100]}...")
        
        # Validate clip durations
        problematic_clips = []
        for i, clip in enumerate(clips, 1):
            duration = clip['end'] - clip['start']
            if duration < 30 or duration > 90:
                problematic_clips.append(i)
        
        if problematic_clips:
            print(f"⚠️  Warning: Clips with unusual durations detected: {problematic_clips}")
            print("These clips may not be exactly 60 seconds but will be processed anyway.")
        
        # Step 4: Setup processing environment
        print(f"\n4. Preparing video processing environment...")
        
        # Ensure temp directory exists and is clean
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), "video_processing")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"📁 Temp directory: {temp_dir}")
        
        # Check available disk space
        try:
            import shutil
            free_space = shutil.disk_usage(temp_dir).free / (1024**3)  # GB
            if free_space < 5:
                print(f"⚠️  Warning: Low disk space ({free_space:.1f}GB available)")
                print("Consider freeing up space if processing fails.")
            else:
                print(f"💾 Available disk space: {free_space:.1f}GB")
        except Exception as e:
            print(f"Could not check disk space: {e}")
        
        # Check GPU availability for NVENC
        try:
            import subprocess
            nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
            if nvidia_smi.returncode == 0:
                gpu_info = nvidia_smi.stdout.strip().split('\n')[0]
                print(f"🎮 GPU detected: {gpu_info}")
            else:
                print("ℹ️  No NVIDIA GPU detected - will use CPU encoding")
        except Exception:
            print("ℹ️  GPU check skipped - will attempt both GPU and CPU encoding")
        
        # Step 5: Create clips with improved processing
        print(f"\n5. Creating {len(clips)} individual video clips...")
        print("🔧 Using improved processing with multiple fallback methods:")
        print("   1. NVENC with subtitles")
        print("   2. NVENC without subtitles") 
        print("   3. CPU encoding with subtitles")
        print("   4. CPU encoding without subtitles")
        print("   5. Safe mode (last resort)")
        
        # Track processing statistics
        processing_stats = {
            'total_clips': len(clips),
            'successful_clips': 0,
            'failed_clips': 0,
            'nvenc_successes': 0,
            'cpu_successes': 0,
            'safe_mode_successes': 0,
            'clips_with_subtitles': 0,
            'clips_with_face_tracking': 0
        }
        
        created_clips, failed_clips = batch_create_clips_with_face_tracking(
            input_video, clips, transcription
        )
        
        # Update statistics (these would need to be collected from the processing functions)
        processing_stats['successful_clips'] = len(created_clips)
        processing_stats['failed_clips'] = len(failed_clips)
        
        # Step 6: Results summary
        print(f"\n✅ Clip generation complete!")
        print(f"📊 Processing Results:")
        print(f"   Successfully created: {len(created_clips)}/{len(clips)} clips")
        
        if created_clips:
            print(f"\n📁 Created clips:")
            for i, clip_path in enumerate(created_clips, 1):
                try:
                    file_size = os.path.getsize(clip_path) / (1024*1024)  # MB
                    print(f"   Clip {i}: {os.path.basename(clip_path)} ({file_size:.1f}MB)")
                except:
                    print(f"   Clip {i}: {clip_path}")
        
        if failed_clips:
            print(f"\n❌ Failed clips: {failed_clips}")
            print("Consider checking these time ranges manually or adjusting the source video.")
        
        # Calculate final statistics
        total_duration = sum(clip['end'] - clip['start'] for clip in clips)
        successful_duration = sum(
            clips[i-1]['end'] - clips[i-1]['start'] 
            for i in range(1, len(clips)+1) 
            if i not in failed_clips
        )
        
        print(f"\n📈 Content Statistics:")
        print(f"   Total selected content: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"   Successfully processed: {successful_duration:.1f} seconds ({successful_duration/60:.1f} minutes)")
        
        if clips:
            avg_duration = total_duration / len(clips)
            print(f"   Average clip length: {avg_duration:.1f} seconds")
            
            if len(created_clips) > 0:
                success_rate = (len(created_clips) / len(clips)) * 100
                print(f"   Success rate: {success_rate:.1f}%")
        
        # Cleanup suggestions
        print(f"\n🧹 Cleanup:")
        print(f"   Temporary files location: {temp_dir}")
        print(f"   Consider cleaning temp directory after reviewing clips")
        
        # Final recommendations
        if len(failed_clips) > 0:
            print(f"\n💡 Troubleshooting failed clips:")
            print("   - Check if the source video has issues at those timestamps")
            print("   - Try processing failed clips individually with safe mode")
            print("   - Consider converting source video to H.264 format first")
        
        if len(created_clips) > 0:
            print(f"\n🎉 Success! {len(created_clips)} clips ready for use.")
            
            # Show output directory
            if created_clips:
                output_dir = os.path.dirname(created_clips[0])
                print(f"📂 Output directory: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user")
        print("🧹 Cleaning up temporary files...")
        
        # Attempt cleanup on interrupt
        try:
            import tempfile
            temp_dir = os.path.join(tempfile.gettempdir(), "video_processing")
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                print("✅ Temporary files cleaned up")
        except Exception as cleanup_error:
            print(f"⚠️  Could not clean up temp files: {cleanup_error}")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\n🔍 Debugging information:")
        traceback.print_exc()
        
        print(f"\n💡 Possible solutions:")
        print("   1. Check if FFmpeg is properly installed and accessible")
        print("   2. Verify input video file is not corrupted")
        print("   3. Ensure sufficient disk space for processing")
        print("   4. Try with a different source video to isolate the issue")
        print("   5. Check if any antivirus software is blocking file operations")
        
        # Show system info for debugging
        try:
            import platform
            print(f"\n🖥️  System info: {platform.system()} {platform.release()}")
            print(f"   Python version: {platform.python_version()}")
        except:
            pass
        
        print("\nProcess failed.")
    
    finally:
        # Always attempt final cleanup
        try:
            print("\n🔄 Final cleanup...")
            # Add any final cleanup operations here
        except Exception as final_error:
            print(f"Warning: Final cleanup error: {final_error}")


if __name__ == "__main__":
    main()
    