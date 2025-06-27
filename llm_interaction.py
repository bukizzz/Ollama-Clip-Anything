# llm_interaction.py
"""
Uses a Large Language Model (LLM) to select engaging clips from the transcript.
"""

import json
import re
import time
import ollama
from config import LLM_MODEL, CLIP_DURATION_RANGE, CLIP_VALIDATION_RANGE


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


def extract_json_from_text(text: str) -> list[dict]:
    """Extract JSON array from LLM response text."""
    patterns = [
        r'\[[\s\S]*?\]',  # bare JSON array
        r'```json\s*(\[[\s\S]*?\])\s*```'  # fenced code block
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                json_str = match if isinstance(match, str) else match[0]
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # trailing comma fix
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON array found in model output")


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
You are a video editor selecting 10 engaging clips. Analyze this transcript.
For each segment, provide start time, end time, and a brief description.
Focus on complete thoughts, engaging moments, {min_dur}-{max_dur} second duration, and no overlaps.
Total duration: {total_duration:.1f} seconds.
Transcript data: {json.dumps(simplified_transcript[:min(50, len(simplified_transcript))], indent=1)}
Provide selections in any clear format.
"""
    pass1_output = llm_pass(LLM_MODEL, [
        {"role": "system", "content": "You are an expert video editor."},
        {"role": "user", "content": pass1_prompt.strip()}
    ])

    print("🔄 Pass 2: Converting to JSON...")
    pass2_prompt = f"""
Convert the following selections into a clean JSON array:
[{{"start": 120.5, "end": 180.5, "text": "description"}}]
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


def get_clips_with_retry(transcript: list[dict], max_retries=100, retry_delay=2) -> list[dict]:
    """Get clips with retry logic using the three-pass LLM approach."""
    for attempt in range(max_retries):
        try:
            print(f"Attempting LLM clip selection ({attempt + 1}/{max_retries})...")
            clips = three_pass_llm_extraction(transcript)
            if len(clips) >= 1:
                print(f"✅ Successfully extracted {len(clips)} clips")
                return clips[:10]
            else:
                print(f"⚠️ Only got {len(clips)} valid clips, need at least 5.")
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    raise RuntimeError(f"Failed to extract clips after {max_retries} attempts.")

