# video_input.py
"""
Handles user input for selecting or downloading the source video.
"""
import os
import subprocess
from pytubefix import YouTube
from temp_manager import get_temp_path
from config import OUTPUT_DIR

def choose_input_video() -> str:
    """Choose between a local MP4 file or a YouTube video download."""
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
        return download_youtube_video(url)

def download_youtube_video(url: str) -> str:
    """Download YouTube video as MP4 with user selection of quality."""
    yt = YouTube(url)
    video_streams = yt.streams.filter(file_extension='mp4').order_by('resolution').desc()

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
