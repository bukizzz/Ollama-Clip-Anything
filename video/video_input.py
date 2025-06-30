# video_input.py
"""
Handles user input for selecting or downloading the source video.
"""
import os
import subprocess
from pytubefix import YouTube
from core.temp_manager import get_temp_path
from core.config import OUTPUT_DIR
import argparse

def get_video_input(video_path: str = None, youtube_url: str = None, youtube_quality: int = None) -> str:
    """Handles video input based on provided arguments."""
    if video_path:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Local video file not found: {video_path}")
        if not video_path.lower().endswith(".mp4"):
            raise ValueError("Local video file must be an .mp4")
        print(f"Using local video: {video_path}")
        return video_path
    elif youtube_url:
        print(f"Downloading YouTube video from: {youtube_url}")
        return download_youtube_video(youtube_url, youtube_quality)
    else:
        raise ValueError("Either --video_path or --youtube_url must be provided.")






def choose_input_video() -> str:
    """Prompts the user for a video source (local file path or YouTube URL) and processes it."""
    while True:
        user_input = input("Enter path to local MP4 file or YouTube URL: ").strip()
        if user_input.startswith("http") or user_input.startswith("www."):
            print(f"Detected YouTube URL: {user_input}")
            return download_youtube_video(user_input)
        elif os.path.isfile(user_input):
            print(f"Detected local file: {user_input}")
            return get_video_input(video_path=user_input)
        else:
            print("Invalid input. Please enter a valid local MP4 file path or a YouTube URL.")


def download_youtube_video(url: str, quality_choice: int = None) -> str:
    """Download YouTube video as MP4 with user selection of quality."""
    yt = YouTube(url)
    video_streams = yt.streams.filter(file_extension='mp4').order_by('resolution').desc()
    # Filter out AV1 streams
    video_streams = [s for s in video_streams if not (any('av01' in codec for codec in (s.codecs or [])) or ('av01' in (s.mime_type or '')) or ('av01' in (s.subtype or '')))]

    if not video_streams:
        raise RuntimeError("No compatible MP4 streams found.")

    print(f"\nTitle: {yt.title}")
    print("Available video streams:")
    for i, stream in enumerate(video_streams):
        stream_type = "Progressive" if stream.is_progressive else "Adaptive"
        audio_info = "Audio included" if stream.includes_audio_track else "Video only"
        size_mb = stream.filesize / (1024 * 1024) if stream.filesize else "Unknown"
        size_str = f"{size_mb:.2f} MB" if isinstance(size_mb, float) else size_mb
        print(f"{i}: {stream.resolution} | {stream.fps}fps | {stream_type} | {audio_info} | {size_str}")

    if quality_choice is not None:
        if 0 <= quality_choice < len(video_streams):
            selected_stream = video_streams[quality_choice]
            print(f"Selected stream: {quality_choice} - {selected_stream.resolution}")
        else:
            raise ValueError(f"Invalid quality choice: {quality_choice}. Please choose a number within the displayed range.")
    else:
        while True:
            try:
                choice = int(input("Enter the number of the video stream to download: "))
                if 0 <= choice < len(video_streams):
                    selected_stream = video_streams[choice]
                    break
                else:
                    print("Invalid choice. Please enter a number within the displayed range.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Downloading: {yt.title}")

    if selected_stream.is_progressive:
        output_path = selected_stream.download(output_path=OUTPUT_DIR, filename_prefix="yt_")
    else:
        print("Downloading video-only stream and best audio separately...")
        video_path = selected_stream.download(output_path=get_temp_path(""), filename_prefix="yt_video_")
        
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
        if not audio_stream:
             audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()

        if audio_stream:
            audio_path = audio_stream.download(output_path=get_temp_path(""), filename_prefix="yt_audio_")
            final_filename = f"yt_{yt.title[:50].replace('/', '_').replace('|', '_')}.mp4"
            output_path = os.path.join(OUTPUT_DIR, final_filename)
            
            merge_cmd = [
                "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                "-c", "copy", "-map", "0:v:0", "-map", "1:a:0", output_path
            ]
            print("Merging video and audio...")
            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("FFmpeg merge failed. FFmpeg stderr:\n", result.stderr)
                raise RuntimeError("Failed to merge video and audio with FFmpeg.")
            else:
                os.remove(video_path)
                os.remove(audio_path)
        else:
            print("No audio stream found, using video-only file.")
            output_path = video_path
    
    print(f"Saved to: {output_path}")
    return output_path
