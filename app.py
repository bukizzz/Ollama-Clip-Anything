import os
import json
import re
import subprocess
from moviepy.editor import VideoFileClip, concatenate_videoclips
import ollama
import whisper
import gc
import torch


def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg", "-y",
        "-v", "error",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        "-f", "mp3",
        audio_path
    ]
    print("Running FFmpeg command:", ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg stderr:\n", result.stderr)
        raise RuntimeError("FFmpeg audio extraction failed.")


def transcribe_video(video_path, model_name="base"):
    try:
        model = whisper.load_model(model_name)
        audio_path = "temp_audio.wav"
        extract_audio(video_path, audio_path)
        print("Running Whisper transcription...")
        result = model.transcribe(audio_path)
        transcription = [{
            'start': float(seg['start']),
            'end': float(seg['end']),
            'text': seg['text'].strip()
        } for seg in result['segments']]
        return transcription
    except Exception as e:
        print("Transcription failed:", e)
        raise
    finally:
        # Force model unload
        try:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Whisper model unloaded and GPU memory released.")
        except Exception as cleanup_error:
            print("Error during Whisper model cleanup:", cleanup_error)



def get_relevant_segments(transcript, user_query):
    import time

    prompt = f"""You are an expert video editor who can read video transcripts and perform video editing. Given a transcript with segments, your task is to identify the most interesting continuous part of the conversation. Make the segment no less than 5 minutes long. Output the start time, end time and quick summary as text. Return your output as a JSON with this format:
[
  {{
    "start": float,
    "end": float,
    "text": string
  }},
  ...
]
Transcript:
{json.dumps(transcript)}

User query:
{user_query}"""

    model = "granite3.3:8b"
    ollama.pull(model)

    attempt = 0
    while True:
        attempt += 1
        print(f"\nAttempt #{attempt}: Sending prompt to LLaMA...")
        try:
            response = ollama.chat(model=model, messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ])
            raw_output = response['message']['content']
            print("Response received.")
            print("Raw content:\n", raw_output)
        except Exception as e:
            print("Ollama inference failed:", e)
            continue  # Retry on failure

        try:
            # Extract JSON block
            match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
            if match:
                json_text = match.group(0)
                conversations = json.loads(json_text)
                print(f"Valid JSON parsed with {len(conversations)} segments.")
                return conversations
            else:
                print("No valid JSON block found in response. Retrying...")
        except Exception as e:
            print("JSON parsing failed:", e)

        time.sleep(1)  # brief pause to avoid tight loop



def edit_video(original_video_path, segments, output_video_path):
    print(f"Editing video from {original_video_path} to {output_video_path}")
    video = VideoFileClip(original_video_path)
    clips = []

    for seg in segments:
        try:
            start = float(seg['start'])
            end = float(seg['end'])
            print(f"Extracting subclip: {start:.2f}s → {end:.2f}s")
            clip = video.subclip(start, end).fadein(0.5).fadeout(0.5)
            clips.append(clip)
        except Exception as e:
            print(f"Failed to extract subclip {seg}: {e}")

    if not clips:
        print("No clips extracted.")
        return

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", verbose=True)


def main():
    input_video = "input_video.mp4"
    output_video = "edited_output.mp4"
    user_query = ""

    print("Transcribing video...")
    transcription = transcribe_video(input_video)

    print("Extracting relevant segments...")
    relevant_segments = get_relevant_segments(transcription, user_query)

    print("Editing video...")
    edit_video(input_video, relevant_segments, output_video)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Fatal error:", e)
