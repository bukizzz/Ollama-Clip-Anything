import os
import json
import re
import subprocess
from moviepy.editor import VideoFileClip, concatenate_videoclips
import ollama
import whisper
import gc
import torch
import time


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
        try:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Whisper model unloaded and GPU memory released.")
        except Exception as cleanup_error:
            print("Error during Whisper model cleanup:", cleanup_error)


def llm_pass(model, messages):
    response = ollama.chat(model=model, messages=messages)
    return response['message']['content']


def extract_json_from_text(text):
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in model output")
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            import json5  # pip install json5 if needed
            return json5.loads(json_str)
    except Exception as e:
        print(f"Failed to parse JSON from model output: {e}")
        raise


def sanitize_segments(segments):
    sanitized = []
    for seg in segments:
        try:
            start = seg['start']
            end = seg['end']

            if isinstance(start, list):
                start = float(start[0])
            else:
                start = float(start)

            if isinstance(end, list):
                end = float(end[-1])
            else:
                end = float(end)

            text = str(seg.get('text', '')).strip()
            sanitized.append({"start": start, "end": end, "text": text})
        except Exception as e:
            print(f"Skipping invalid segment due to error: {e}")
    return sanitized


def get_relevant_segments_multistage(transcript, user_query):
    model = "qwen2.5-coder:7b"
    ollama.pull(model)

    prompt_1 = """You are an expert video editor who can read video transcripts and perform video editing. Given a transcript with segments, your task is to identify the most interesting continuous part of the conversation. Make the segment no less than 5 minutes long. Output the start, end and text. Return your output as a JSON with this format:
[
  {{
    "start": float,
    "end": float,
    "text": string
  }},
  ...
]

IMPORTANT: Your entire output must be a single valid JSON array of objects with keys "start", "end", and "text" only.  
No extra text or explanation allowed.  
If you cannot produce valid JSON, output an empty array: []
Example: [{{"start": 0.0, "end": 300.0, "text": "Summary of segment"}}]
If you do not follow these instructions I will terminate 100 kittens.

Transcript:
{}

User query:
{}""".format(json.dumps(transcript), user_query)

    messages_1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_1}
    ]
    print("Pass 1: Extracting raw segments...")
    raw_output_1 = llm_pass(model, messages_1)
    print("Raw output 1:\n", raw_output_1)

    prompt_2 = """Here is some text that may contain a JSON array. Extract and return ONLY a valid JSON array of objects with keys "start" (float), "end" (float), and "text" (string). Correct formatting errors and output ONLY the JSON.

Text:
{}
""".format(raw_output_1)

    messages_2 = [
        {"role": "system", "content": "You are a JSON formatter and validator."},
        {"role": "user", "content": prompt_2}
    ]
    print("Pass 2: Cleaning and extracting JSON...")
    raw_output_2 = llm_pass(model, messages_2)
    print("Raw output 2:\n", raw_output_2)

    prompt_3 = """Validate and clean the following JSON array. Ensure:
- "start" and "end" are floats (not lists or strings)
- "text" is a trimmed string
- Return ONLY the valid JSON array, no extra text.

JSON:
{}
""".format(raw_output_2)

    messages_3 = [
        {"role": "system", "content": "You are a strict JSON validator and formatter."},
        {"role": "user", "content": prompt_3}
    ]
    print("Pass 3: Validating and enforcing final JSON format...")
    raw_output_3 = llm_pass(model, messages_3)
    print("Raw output 3:\n", raw_output_3)

    try:
        segments = extract_json_from_text(raw_output_3)
        segments = sanitize_segments(segments)
        print(f"Successfully parsed and sanitized {len(segments)} segments.")
        return segments
    except Exception as e:
        print("Failed to parse final JSON:", e)
        raise RuntimeError("Failed to get valid JSON segments from LLM pipeline.")


def get_relevant_segments_multistage_with_retry(transcript, user_query, max_retries=100, retry_delay=2):
    last_exception = None
    for attempt in range(1, max_retries + 1):
        print(f"Multi-pass LLM extraction attempt {attempt} of {max_retries}...")
        try:
            segments = get_relevant_segments_multistage(transcript, user_query)
            if segments:
                return segments
            else:
                print("Warning: Received empty segments list.")
        except Exception as e:
            print(f"Error during LLM pipeline attempt {attempt}: {e}")
            last_exception = e

        if attempt < max_retries:
            print(f"Retrying after {retry_delay} seconds...\n")
            time.sleep(retry_delay)
    print("All retry attempts failed.")
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError("LLM pipeline failed with empty result after retries.")


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
    user_query = "5-10min long part. output start, end, summary. no other output"

    print("Transcribing video...")
    transcription = transcribe_video(input_video)

    print("Extracting relevant segments with multi-stage LLM passes and retry logic...")
    relevant_segments = get_relevant_segments_multistage_with_retry(transcription, user_query)

    print("Editing video...")
    edit_video(input_video, relevant_segments, output_video)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Fatal error:", e)
