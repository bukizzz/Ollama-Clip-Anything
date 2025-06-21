import os
import json
import re
import subprocess
import gc
import time
import traceback
import torch
import whisper
import ollama
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import crop, colorx, speedx
from pytubefix import YouTube


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
        
        # Download video
        video_path = selected_stream.download(output_path=output_dir, filename_prefix="yt_video_")
        
        # Get best audio stream
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
        if not audio_stream:
            audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        
        if audio_stream:
            audio_path = audio_stream.download(output_path=output_dir, filename_prefix="yt_audio_")
            
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


def get_next_output_filename(output_dir="videos", batch_prefix="clip", ext=".mp4", clip_number=1) -> str:
    """Generate filename for individual clips."""
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
        audio_path = "temp_audio.wav"  # Use WAV for better precision
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

        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

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
                clip_subtitles.append({
                    'start': new_start,
                    'end': new_end,
                    'text': orig_seg['text'].strip()
                })
    
    # Write the subtitle file
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(clip_subtitles, 1):
            start = format_timestamp(sub['start'])
            end = format_timestamp(sub['end'])
            text = sub['text'].replace('\n', ' ').strip()
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
{json.dumps(simplified_transcript[:25], indent=1)}

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


def sanitize_segments(segments: list[dict]) -> list[dict]:
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
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting three-pass LLM clip selection (attempt {attempt + 1}/{max_retries})...")
            clips = get_60_second_clips_with_llm(transcript)
            
            if len(clips) >= 5:  # Accept if we get at least 5 good clips
                print(f"✅ Successfully extracted {len(clips)} clips")
                return clips[:10]  # Take first 10 if we get more
            else:
                print(f"⚠️ Only got {len(clips)} clips, need at least 5")
                
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    raise RuntimeError(f"Failed to extract clips after {max_retries} attempts with three-pass method.")


# ----------- VIDEO EDITING -----------

def apply_9_16_crop_and_tracking(clip: VideoFileClip) -> VideoFileClip:
    """
    Apply 9:16 aspect ratio crop with face/head tracking and color adjustments.
    Correctly handles aspect ratio conversion by only cropping from width or height.
    """
    # Get source dimensions
    source_width = clip.w
    source_height = clip.h
    
    # Calculate what width would give us 9:16 aspect ratio with current height
    target_width_for_height = int(source_height * 9/16)
    
    # Check if source is wide enough to crop to 9:16
    if source_width >= target_width_for_height:
        # Source is wide enough - crop from width only
        target_width = target_width_for_height
        target_height = source_height
        
        print(f"Converting {source_width}x{source_height} to 9:16 format: {target_width}x{target_height}")
        print("Cropping from width (source is wide enough)")
        
    else:
        # Source is too narrow - we need to crop from height instead
        # Calculate what height would give us 9:16 with current width
        target_height = int(source_width * 16/9)
        target_width = source_width
        
        print(f"Converting {source_width}x{source_height} to 9:16 format: {target_width}x{target_height}")
        print("Cropping from height (source is too narrow)")
    
    # Calculate crop position
    if target_width < source_width:
        # Cropping from width - center horizontally
        crop_x = (source_width - target_width) // 2
        crop_y = 0
    else:
        # Cropping from height - bias toward upper portion for head tracking
        crop_x = 0
        crop_y = int((source_height - target_height) * 0.3)  # 30% from top
    
    # Ensure crop coordinates are within bounds
    crop_x = max(0, min(crop_x, source_width - target_width))
    crop_y = max(0, min(crop_y, source_height - target_height))
    
    # Apply the crop
    cropped = crop(clip, 
                  x1=crop_x, 
                  y1=crop_y, 
                  x2=crop_x + target_width, 
                  y2=crop_y + target_height)
    
    # Apply color adjustments
    adjusted = colorx(cropped, 1.1)
    return adjusted.fl_image(lambda img: (img * 1.02).clip(0, 255).astype("uint8"))


def adjust_audio(audio_clip):
    """Apply audio speed adjustment."""
    return speedx(audio_clip, 1.0)


def create_individual_clip(original_video, clip_data, clip_number, original_transcript):
    """
    Create a single 60-second clip with perfect audio-video-subtitle synchronization.
    
    Args:
        original_video: VideoFileClip object of the original video
        clip_data: Dict with start, end, and text for the clip
        clip_number: Number of this clip (1-10)
        original_transcript: Full transcript for subtitle timing
    
    Returns:
        Path to the created clip file
    """
    start_time = float(clip_data['start'])
    end_time = float(clip_data['end'])
    duration = end_time - start_time
    
    print(f"Creating clip {clip_number}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
    
    # Generate output filename
    output_path = get_next_output_filename(clip_number=clip_number)
    
    # Extract the clip with precise timing
    clip = original_video.subclip(start_time, end_time)
    
    # Apply effects in order that minimizes timing disruption
    # 1. First apply 9:16 crop and color adjustments (these don't affect timing)
    clip = apply_9_16_crop_and_tracking(clip)
    
    # 2. Adjust audio if present (maintain sync)
    if clip.audio:
        clip = clip.set_audio(adjust_audio(clip.audio))
    
    # 3. Apply fade effects LAST (these can affect timing)
    #clip = clip.fadein(0.5).fadeout(0.5)
    
    # Create subtitles BEFORE video processing to get accurate timing
    subtitle_path = f"temp_subtitles_{clip_number}.srt"
    create_clip_subtitles(original_transcript, start_time, end_time, subtitle_path)
    
    # Write video with high precision timing
    temp_output = f"temp_clip_{clip_number}.mp4"
    clip.write_videofile(
        temp_output, 
        codec="libx264", 
        audio_codec="aac", 
        verbose=False, 
        logger=None,
        preset='medium',  # Better encoding precision
        ffmpeg_params=['-avoid_negative_ts', 'make_zero']  # Ensure proper timestamp handling
    )
    
    # Add subtitles with precise timing control
    ffmpeg_cmd = [
        "ffmpeg", "-y", 
        "-i", temp_output,
        "-vf", f"subtitles={subtitle_path}:force_style='Fontsize=24,PrimaryColour=&HFFFFFF&'",
        "-c:a", "copy",
        "-avoid_negative_ts", "make_zero",  # Prevent timing issues
        "-fflags", "+genpts",  # Generate proper timestamps
        output_path
    ]
    
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg stderr for clip {clip_number}:", result.stderr)
        # If subtitle addition fails, use the clip without subtitles
        os.rename(temp_output, output_path)
    else:
        # Clean up temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
    
    # Clean up subtitle file
    if os.path.exists(subtitle_path):
        os.remove(subtitle_path)
    
    # Close the clip to free memory
    clip.close()
    
    print(f"✅ Clip {clip_number} saved to: {output_path}")
    return output_path


def create_all_clips(original_video_path, clips_data, original_transcript):
    """
    Create all 60-second clips from the original video.
    
    Args:
        original_video_path: Path to original video file
        clips_data: List of clip data with start, end, and text
        original_transcript: Full transcript for subtitle timing
    
    Returns:
        List of paths to created clip files
    """
    print("Loading original video...")
    video = VideoFileClip(original_video_path)
        
    created_clips = []

    MAX_RETRIES = 3

    try:
        for i, clip_data in enumerate(clips_data, 1):
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"\n🎬 Creating clip {i} (attempt {attempt}/{MAX_RETRIES})...")
                    clip_path = create_individual_clip(video, clip_data, i, original_transcript)
                    created_clips.append(clip_path)
                    break  # success
                except Exception as e:
                    print(f"❌ Attempt {attempt} failed for clip {i}: {e}")
                    if attempt == MAX_RETRIES:
                        print(f"⛔ Giving up on clip {i} after {MAX_RETRIES} attempts.")
                        traceback.print_exc()
    finally:
    # Always close the video to free memory
        video.close()

    return created_clips



# ----------- MAIN ENTRY POINT -----------

def main():
    """Main function to orchestrate the entire video processing pipeline."""
    try:
        print("=== 60-Second Clips Generator ===")
        
        # Step 1: Get input video
        print("\n1. Getting input video...")
        input_video = choose_input_video()
        print(f"Input video: {input_video}")

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
            print(f"  Clip {i}: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s)")
            print(f"    Description: {clip['text'][:100]}...")

        # Step 4: Create all clips
        print(f"\n4. Creating {len(clips)} individual video clips...")
        created_clips = create_all_clips(input_video, clips, transcription)
        
        print(f"\n✅ Clip generation complete!")
        print(f"Successfully created {len(created_clips)} clips:")
        
        for i, clip_path in enumerate(created_clips, 1):
            print(f"  Clip {i}: {clip_path}")
        
        # Calculate statistics
        total_duration = sum(clip['end'] - clip['start'] for clip in clips)
        print(f"\nTotal extracted content: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        if clips:
            print(f"Average clip length: {total_duration/len(clips):.1f} seconds")

    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        print("Process failed.")


if __name__ == "__main__":
    main()