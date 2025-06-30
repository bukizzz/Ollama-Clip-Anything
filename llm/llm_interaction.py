# llm_interaction.py
"""
Uses a Large Language Model (LLM) to select engaging clips from the transcript.
"""

import json
import re
import time
import ollama
import subprocess # Import subprocess to run shell commands
import torch # Import torch for CUDA memory management
from core.config import LLM_MODEL, CLIP_DURATION_RANGE, CLIP_VALIDATION_RANGE, LLM_MAX_RETRIES, LLM_MIN_CLIPS_NEEDED


def llm_pass(model: str, messages: list[dict]) -> str:
    """Send messages to Ollama model and return response content."""
    try:
        response = ollama.chat(model=model, messages=messages)
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            raise ValueError(f"Unexpected response format: {response}")
    except Exception as e:
        print(f"LLM request failed: {e}")
        raise

def cleanup():
    """Release resources held by the Ollama client by unloading the model."""
    try:
        # Explicitly unload the Ollama model using a shell command
        # This is necessary as the Python client does not expose a direct unload method.
        print(f"Attempting to unload Ollama model: {LLM_MODEL}...")
        result = subprocess.run(["ollama", "unload", LLM_MODEL], check=True, capture_output=True, text=True)
        print(f"Ollama model {LLM_MODEL} unloaded successfully.")
        print(f"Ollama unload stdout: {result.stdout}")
        print(f"Ollama unload stderr: {result.stderr}")
    except FileNotFoundError:
        print("Warning: 'ollama' command not found. Ensure Ollama is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error unloading Ollama model: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred during Ollama cleanup: {e}")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Ollama related GPU memory released.")
    except ImportError:
        pass # torch not installed or not relevant


def extract_json_from_text(text: str) -> list[dict]:
    """Extract JSON array from LLM response text, with improved robustness."""
    # Attempt direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # If direct parsing fails, try to extract JSON from common patterns
    patterns = [
        r'```json\s*(\[[\s\S]*?\])\s*```',  # fenced code block
        r'\[[\s\S]*?\]'  # bare JSON array
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                json_str = match if isinstance(match, str) else match[0]
                # Attempt to fix common JSON issues like trailing commas
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not decode extracted JSON string. Error: {e}")
                continue
    raise ValueError("No valid JSON array found in model output after multiple attempts.")


def sanitize_segments(segments: list[dict], max_duration: float = None) -> list[dict]:
    """Clean and validate segment data from LLM output."""
    cleaned = []
    min_dur, max_dur = CLIP_VALIDATION_RANGE
    for seg in segments:
        try:
            start = float(seg['start'])
            end = float(seg['end'])
            text = str(seg.get('text', '')).strip()
            duration = end - start
            if start >= 0 and end > start and min_dur <= duration <= max_dur:
                if max_duration is None or end <= max_duration:
                    cleaned.append({
                        "start": start,
                        "end": end,
                        "text": text
                    })
        except (ValueError, TypeError, KeyError) as e:
            print(f"Skipping invalid segment: {seg} - Error: {e}")
            continue
    return cleaned


def three_pass_llm_extraction(transcript: list[dict]) -> list[dict]:
    """Use a three-pass approach for robust LLM-based clip extraction."""
    min_dur, max_dur = CLIP_DURATION_RANGE
    total_duration = transcript[-1]['end']

    simplified_transcript = [{
        "index": i,
        "start": round(seg['start'], 1),
        "end": round(seg['end'], 1),
        "text": seg['text'][:150]
    } for i, seg in enumerate(transcript)]

    print("🔄 Pass 1: Initial clip extraction...")
    pass1_prompt = f"""
You are a video editor selecting up to 10 engaging clips. Analyze this transcript.
For each segment, provide start time, end time, and a brief description.
Focus on engaging moments and a duration between {min_dur}-{max_dur} seconds. Aim to identify as many distinct, valid clips as possible, even if there are slight overlaps or they are not perfectly 'complete thoughts' – these will be refined in later steps.
Total duration: {total_duration:.1f} seconds.
Transcript data: {json.dumps(simplified_transcript[:min(50, len(simplified_transcript))], indent=1)}
Provide selections as a numbered list, with each item clearly indicating 'Start:', 'End:', and 'Description:'.
"""
    pass1_output = llm_pass(LLM_MODEL, [
        {"role": "system", "content": "You are an expert video editor."},
        {"role": "user", "content": pass1_prompt.strip()}
    ])

    print("🔄 Pass 2: Converting to JSON...")
    pass2_prompt = f"""
Convert the following selections into a clean JSON array of objects, each with "start", "end", and "text" keys.
Example: [{{"start": 120.5, "end": 180.5, "text": "description"}}]
Ensure times are numbers, duration is {min_dur}-{max_dur}s, no overlaps, max 10 clips.
Original selections: {pass1_output}
Return ONLY the JSON array.
"""
    pass2_output = llm_pass(LLM_MODEL, [
        {"role": "system", "content": "You are a JSON formatter."},
        {"role": "user", "content": pass2_prompt.strip()}
    ])

    print("🔄 Pass 3: Final validation...")
    pass3_prompt = f"""
Validate and clean this JSON. Fix syntax, ensure times are numbers, duration is {min_dur}-{max_dur}s,
no overlaps, times are within 0 to {total_duration:.1f}s.
Input JSON: {pass2_output}
Return the cleaned, valid JSON array only.
"""
    pass3_output = llm_pass(LLM_MODEL, [
        {"role": "system", "content": "You are a JSON validator."},
        {"role": "user", "content": pass3_prompt.strip()}
    ])

    try:
        segments = extract_json_from_text(pass3_output)
        cleaned_segments = sanitize_segments(segments, total_duration)
        print(f"✅ Three-pass extraction yielded {len(cleaned_segments)} valid clips")
        return cleaned_segments
    except Exception as e:
        print(f"❌ Three-pass extraction failed at final parsing: {e}")
        raise


if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Ollama related GPU memory released.")

def get_clips_with_retry(transcript: list[dict], retry_delay=2) -> list[dict]:
    """Get clips with retry logic using the three-pass LLM approach."""
    for attempt in range(LLM_MAX_RETRIES):
        try:
            print(f"Attempting LLM clip selection ({attempt + 1}/{LLM_MAX_RETRIES})...")
            clips = three_pass_llm_extraction(transcript)
            if len(clips) >= LLM_MIN_CLIPS_NEEDED:
                print(f"✅ Successfully extracted {len(clips)} clips")
                return clips[:10]
            else:
                print(f"⚠️ Only got {len(clips)} valid clips, need at least {LLM_MIN_CLIPS_NEEDED}.")
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < LLM_MAX_RETRIES - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    raise RuntimeError(f"Failed to extract clips after {LLM_MAX_RETRIES} attempts.")

