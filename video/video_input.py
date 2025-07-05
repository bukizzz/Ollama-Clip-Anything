# video_input.py
"""
Hanldes user input for selecting or downloading the source video.
"""
import os
import subprocess
import re # Import re for regex operations
from pytubefix import YouTube
from core.temp_manager import get_temp_path
from core.config import config
from typing import Optional # Import Optional


def get_video_input(video_path: str, youtube_url: Optional[str] = None, youtube_quality: Optional[int] = None) -> str:
    """Handles video input based on provided arguments."""
    if video_path:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Local video file not found: {video_path}")
        if not video_path.lower().endswith(".mp4"):
            raise ValueError("Local video file must be an .mp4")
        print(f"📁 \033[92mUsing local video: {video_path}\033[0m\n")
        return video_path
    elif youtube_url:
        print(f"⬇️ \033[94mDownloading YouTube video from: {youtube_url}\033[0m\n")
        return download_youtube_video(youtube_url, youtube_quality)
    else:
        raise ValueError("Either --video_path or --youtube_url must be provided.")


def choose_input_video() -> str:
    """Prompts the user for a video source (local file path or YouTube URL) and processes it."""
    while True:
        user_input = input("Enter path to local MP4 file or YouTube URL: ").strip()
        if user_input.startswith("http") or user_input.startswith("www."):
            print(f"🔗 \033[96mDetected YouTube URL: {user_input}\033[0m")
            return download_youtube_video(user_input)
        elif os.path.isfile(user_input):
            print(f"📁 \033[96mDetected local file: {user_input}\033[0m")
            return get_video_input(video_path=user_input)
        else:
            print("❌ \033[91mInvalid input. Please enter a valid local MP4 file path or a YouTube URL.\033[0m")


def download_youtube_video(url: str, quality_choice: Optional[int] = None) -> str:
    """Download YouTube video as MP4 with user selection of quality."""
    # Robustly extract video ID from various YouTube URL formats
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?.*)?$", url)
    if not video_id_match:
        video_id_match = re.search(r"youtu\.be\/([0-9A-Za-z_-]{11})(?:\?.*)?$", url)
    
    if not video_id_match:
        raise ValueError(f"Could not extract YouTube video ID from URL: {url}. Please provide a valid YouTube video URL.")
    
    video_id = video_id_match.group(1)
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}") # Reconstruct URL with extracted ID

    video_streams = yt.streams.filter(file_extension='mp4').order_by('resolution').desc()
    original_stream_count = len(video_streams)
    # Filter out AV1 streams
    video_streams = [s for s in video_streams if not (any('av01' in codec for codec in (s.codecs or [])) or ('av01' in (s.mime_type or '')) or ('av01' in (s.subtype or '')))]
    if len(video_streams) < original_stream_count:
        print("⚠️ \033[38;5;208mExcluded some AV1 encoded streams due to compatibility issues.\033[0m")

    if not video_streams:
        raise RuntimeError("No compatible MP4 streams found.")

    print(f"\n🎬 \033[94mTitle: {yt.title}\033[0m\n")
    print("📺 \033[96mAvailable video streams:\033[0m")
    for i, stream in enumerate(video_streams):
        stream_type = "Progressive" if stream.is_progressive else "Adaptive"
        audio_info = "Audio included" if stream.includes_audio_track else "Video only"
        size_mb = stream.filesize / (1024 * 1024) if stream.filesize else "Unknown"
        size_str = f"{size_mb:.2f} MB" if isinstance(size_mb, float) else size_mb
        print(f"  {i}: {stream.resolution} | {stream.fps}fps | {stream_type} | {audio_info} | {size_str}")

    selected_stream = None # Initialize selected_stream
    if quality_choice is not None:
        if 0 <= quality_choice < len(video_streams):
            selected_stream = video_streams[quality_choice]
            print(f"✅ \033[92mSelected stream: {quality_choice} - {selected_stream.resolution}\033[0m\n")
        else:
            print(f"❌ \033[91mInvalid quality choice: {quality_choice}. Please choose a number within the displayed range.\033[0m")
            # Fallback to interactive choice if quality_choice is invalid
            while True:
                try:
                    choice = int(input("Enter the number of the video stream to download: "))
                    if 0 <= choice < len(video_streams):
                        selected_stream = video_streams[choice]
                        break
                    else:
                        print("❌ \033[91mInvalid choice. Please enter a number within the displayed range.\033[0m")
                except ValueError:
                    print("❌ \033[91mInvalid input. Please enter a number.\033[0m")
    else:
        while True:
            try:
                choice = int(input("Enter the number of the video stream to download: "))
                if 0 <= choice < len(video_streams):
                    selected_stream = video_streams[choice]
                    break
                else:
                    print("❌ \033[91mInvalid choice. Please enter a number within the displayed range.\033[0m")
            except ValueError:
                print("❌ \033[91mInvalid input. Please enter a number.\033[0m")

    if selected_stream is None: # Ensure a stream is selected before proceeding
        raise RuntimeError("No valid video stream was selected for download.")

    if not os.path.exists(config.get('output_dir')):
        os.makedirs(config.get('output_dir'))

    print(f"⬇️ \033[94mDownloading: {yt.title}\033[0m\n")

    video_path_temp: Optional[str] = None # Renamed to avoid conflict with function parameter
    audio_path_temp: Optional[str] = None # Renamed to avoid conflict with function parameter
    output_path: str

    if selected_stream.is_progressive:
        output_path = selected_stream.download(output_path=config.get('output_dir'), filename_prefix="yt_")
    else:
        print("⬇️ \033[94mDownloading video-only stream and best audio separately...\033[0m\n")
        video_path_temp = selected_stream.download(output_path=get_temp_path(""), filename_prefix="yt_video_")
        
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
        if not audio_stream:
             audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()

        if audio_stream:
            audio_path_temp = audio_stream.download(output_path=get_temp_path(""), filename_prefix="yt_audio_")
            final_filename = f"yt_{yt.title[:50].replace('/', '_').replace('|', '_')}.mp4"
            output_path = os.path.join(config.get('output_dir'), final_filename)
            
            merge_cmd = [
                "ffmpeg", "-y", "-i", video_path_temp, "-i", audio_path_temp,
                "-c", "copy", "-map", "0:v:0", "-map", "1:a:0", output_path
            ]
            print("➕ \033[94mMerging video and audio...\033[0m\n")
            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # print("FFmpeg merge failed. FFmpeg stderr:\n", result.stderr)
                raise RuntimeError("Failed to merge video and audio with FFmpeg.")
            else:
                if video_path_temp and os.path.exists(video_path_temp):
                    os.remove(video_path_temp)
                if audio_path_temp and os.path.exists(audio_path_temp):
                    os.remove(audio_path_temp)
        else:
            print("🔇 \033[90mNo audio stream found, using video-only file.\033[0m\n")
            output_path = video_path_temp if video_path_temp else "" # Ensure output_path is not None if video_path is None
            if not output_path:
                raise RuntimeError("Failed to download video-only stream.")
    
    print(f"💾 \033[92mSaved to: {output_path}\033[0m\n")
    return output_path
