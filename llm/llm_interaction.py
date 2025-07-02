# llm_interaction.py
"""
Uses a Large Language Model (LLM) to select engaging clips from the transcript.
"""

import json
import re
import time
import ollama
import torch
import gc
import subprocess # Import subprocess for running shell commands
import os # Import os for path operations
from typing import Any, Dict, List, Optional
import logging

# Disable HTTPX logging for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)

from core.config import LLM_MODEL, IMAGE_RECOGNITION_MODEL, CLIP_DURATION_RANGE, CLIP_VALIDATION_RANGE, LLM_MAX_RETRIES, LLM_MIN_CLIPS_NEEDED, LLM_CONFIG
from core.gpu_manager import release_gpu_memory

# Define common hallucinated words/units to remove from numerical values
HALLUCINATED_UNITS = r'\b(?:seconds?|s|minutes?|min|m|milliseconds?|ms|hours?|h)\b'

def clean_numerical_value(text: str) -> float:
    """
    Robustly extracts a float value from a string, removing common hallucinated words/units.
    """
    # Remove common hallucinated words/units
    cleaned_text = re.sub(HALLUCINATED_UNITS, '', text, flags=re.IGNORECASE).strip()
    
    # Extract the first numerical pattern (integer or float)
    match = re.search(r'[-+]?\d*\.?\d+', cleaned_text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            pass # Fall through to raise error if conversion fails
    
    raise ValueError(f"Could not extract a valid numerical value from: '{text}' (cleaned: '{cleaned_text}')")


def cleanup():
    """
    Attempts to clear GPU memory and provides guidance for Ollama model unloading.
    """
    print("🧹 Attempting to clear GPU memory...")
    # Clear PyTorch CUDA cache and run garbage collector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ PyTorch CUDA cache cleared.")
    gc.collect()
    print("✅ Python garbage collector run.")
    print("ℹ️ To ensure Ollama models are fully unloaded, consider restarting the Ollama server or running 'ollama unload' manually if issues persist.")

def llm_pass(model: str, messages: list[dict]) -> str:
    """Send messages to Ollama model and return response content."""
    try:
        response = ollama.chat(model=model, messages=messages)
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            raise ValueError(f"Unexpected response format: {response}")
    except Exception as e:
        print(f"❌ LLM request failed: {e}")
        raise

def extract_json_from_text(text: str) -> Any:
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
                
                # Targeted pre-cleaning for numerical fields within the JSON string
                # This regex looks for "key": "value" where value might contain numbers and words
                # It's a heuristic and might need refinement based on actual LLM output patterns.
                # For now, focusing on 'start' and 'end' as they are critical.
                json_str = re.sub(r'("start"\s*:\s*")([^"]+?)(")', 
                                  lambda m: f'{m.group(1)}{clean_numerical_value(m.group(2))}{m.group(3)}', 
                                  json_str, flags=re.IGNORECASE)
                json_str = re.sub(r'("end"\s*:\s*")([^"]+?)(")', 
                                  lambda m: f'{m.group(1)}{clean_numerical_value(m.group(2))}{m.group(3)}', 
                                  json_str, flags=re.IGNORECASE)

                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Could not decode extracted JSON string. Error: {e}")
                continue
            except ValueError as e:
                print(f"⚠️ Warning: Numerical cleaning failed for a field in extracted JSON. Error: {e}")
                continue
    raise ValueError("No valid JSON array found in model output after multiple attempts.")


def sanitize_segments(segments: List[Dict[str, Any]], max_duration: Optional[float] = None) -> List[Dict[str, Any]]:
    """Clean and validate segment data from LLM output."""
    cleaned = []
    min_dur, max_dur = CLIP_VALIDATION_RANGE
    for seg in segments:
        try:
            # Convert all keys to lowercase for robust access
            seg_lower = {k.lower(): v for k, v in seg.items()}

            # Use the new clean_numerical_value function for start and end
            start = clean_numerical_value(str(seg_lower['start']))
            end = clean_numerical_value(str(seg_lower['end']))
            
            text = str(seg_lower.get('text', '')).strip()
            duration = end - start
            
            b_roll_image = str(seg_lower.get("b_roll_image", "")).strip()
            # Validate b_roll_image path
            if b_roll_image:
                from core.config import B_ROLL_ASSETS_DIR
                b_roll_full_path = os.path.join(B_ROLL_ASSETS_DIR, b_roll_image)
                if not os.path.exists(b_roll_full_path):
                    print(f"⏩ Skipping segment due to invalid b_roll_image path: {b_roll_image} (full path: {b_roll_full_path})")
                    b_roll_image = "" # Clear invalid path
            
            if start >= 0 and end > start and min_dur <= duration <= max_dur:
                if max_duration is None or end <= max_duration:
                    cleaned.append({
                        "start": start,
                        "end": end,
                        "text": text,
                        "split_screen": bool(seg_lower.get("split_screen", False)),
                        "b_roll_image": b_roll_image
                    })
        except (ValueError, TypeError, KeyError) as e:
            print(f"⏩ Skipping invalid segment: {seg} - Error: {e}")
            continue
    return cleaned


def robust_llm_json_extraction(system_prompt: str, user_prompt: str) -> Any:
    """
    A robust, single-pass approach for extracting JSON data from an LLM.
    """
    print("🔄 Attempting single-pass JSON extraction...")
    response_content = llm_pass(LLM_CONFIG.get('model', LLM_MODEL), [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    try:
        return extract_json_from_text(response_content)
    except Exception as e:
        print(f"❌ Single-pass extraction failed: {e}")
        raise

def get_clips_from_llm(transcript: List[Dict[str, Any]], user_prompt: Optional[str] = None, b_roll_data: Optional[List[Dict[str, Any]]] = None, retry_delay=2) -> List[Dict[str, Any]]:
    """Get clips with retry logic using the single-pass LLM approach."""
    min_dur, max_dur = CLIP_DURATION_RANGE
    total_duration = transcript[-1]['end'] if transcript else 0

    simplified_transcript = [{
        "index": i,
        "start": round(seg['start'], 1),
        "end": round(seg['end'], 1),
        "text": seg['text'][:150]
    } for i, seg in enumerate(transcript)]

    system_prompt = "You are an expert video editor. Your task is to select engaging clips from a video transcript. Provide the output as a JSON array of objects, where each object represents a clip. Each clip object must have 'start' (float), 'end' (float), 'text' (string), 'split_screen' (boolean), and optionally 'b_roll_image' (string, path to image). Ensure 'start' and 'end' values are raw float numbers in seconds, without any additional words or units. Focus on engaging moments and a duration between " + str(min_dur) + "-" + str(max_dur) + " seconds. Prioritize segments that represent a complete thought, story, or a natural conversational turn. Avoid cutting mid-sentence or mid-word. Aim to identify as many distinct, valid clips as possible, even if there are slight overlaps or they are not perfectly 'complete thoughts' – these will be refined in later steps. Return ONLY the JSON array."

    main_prompt = f"""
Analyze this transcript and select up to 10 engaging clips.
Total duration: {total_duration:.1f} seconds.
# Transcript data: {json.dumps(simplified_transcript[:min(50, len(simplified_transcript))], indent=1)}

Available B-roll images and their descriptions:
{json.dumps(b_roll_data, indent=2) if b_roll_data else "No B-roll images available."}

"""
    if user_prompt:
        main_prompt += f"\n\nUser's specific request: {user_prompt}"

    for attempt in range(LLM_MAX_RETRIES):
        try:
            print(f"🧠 Attempting LLM clip selection ({attempt + 1}/{LLM_MAX_RETRIES})...")
            segments = robust_llm_json_extraction(system_prompt, main_prompt)
            if not isinstance(segments, list):
                raise ValueError("LLM did not return a list of segments")
            cleaned_segments = sanitize_segments(segments, total_duration)
            if len(cleaned_segments) >= LLM_MIN_CLIPS_NEEDED:
                print(f"✅ Successfully extracted {len(cleaned_segments)} valid clips")
                # Clean up GPU memory after a successful operation
                release_gpu_memory()
                return cleaned_segments[:10]
            else:
                print(f"⚠️ Only got {len(cleaned_segments)} valid clips, need at least 1.")
        except Exception as e:
            print(f"❌ \033[91mAttempt {attempt + 1} failed: {e}\033[0m")
            if attempt < LLM_MAX_RETRIES - 1:
                print(f"🔄 \033[38;5;208mRetrying in {retry_delay} seconds...\033[0m")
                time.sleep(retry_delay)
    # Clean up GPU memory even if all retries fail
    release_gpu_memory()
    raise RuntimeError(f"Failed to extract clips after {LLM_MAX_RETRIES} attempts.")
