# llm_interaction.py
"""
Uses a Large Language Model (LLM) to select engaging clips from the transcript.
"""


import re
import time
import torch
import gc
import os
import base64
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union, cast


from core.config import config
from core.gpu_manager import gpu_manager

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI # Added for Gemini
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator, model_validator


# Disable HTTPX logging for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)

class Scene(BaseModel):
    start_time: float = Field(description="The start time of the scene in seconds.")
    end_time: float = Field(description="The end time of the scene in seconds.")
    description: str = Field(description="A concise summary of the spoken content in this scene.")

    @model_validator(mode='after')
    def validate_times(self) -> 'Scene':
        if not (self.start_time >= 0 and self.end_time > self.start_time):
            raise ValueError(f"Invalid scene times: start_time={self.start_time}, end_time={self.end_time}. Must be start_time >= 0 and end_time > start_time.")
        return self

class Clip(BaseModel):
    clip_description: str = Field(description="A high-level description of the entire clip, encompassing all its scenes.")
    total_duration: float = Field(description="The total duration of the clip in seconds, calculated from its scenes.")
    reason: str = Field(description="Why this clip was selected (e.g., 'contains a key argument', 'demonstrates a feature clearly', 'captures an emotional peak', 'part of a complete narrative arc').")
    viral_potential_score: int = Field(description="An assessment of the clip's potential to go viral, from 0 (low) to 10 (high).", ge=0, le=10)
    scenes: List[Scene] = Field(description="""A list of continuous, chronological segments (scenes) that collectively form the coherent narrative of the clip. Multiple scenes can be stitched together to create a single clip. Each scene should have:
        - `start_time`: The start time of this specific scene in seconds.
        - `end_time`: The end time of this specific scene in seconds.
        - `scene_duration`: The duration of this specific scene in seconds.
        - `description`: A detailed, factual summary of the key actions, dialogue, and significant visual changes occurring within *this continuous scene*. This description should highlight how the scene contributes to the overall clip's narrative.""")

    @model_validator(mode='after')
    def validate_clip_logic(self) -> 'Clip':
        if not self.scenes:
            raise ValueError("A clip must contain at least one scene.")
        
        # Ensure scenes are chronological
        sorted_scenes = sorted(self.scenes, key=lambda s: s.start_time)
        if sorted_scenes != self.scenes:
            raise ValueError("Scenes are not chronological.")
        
        # CORRECTED LOGIC: Calculate total_duration by summing individual scene durations
        calculated_duration = sum(s.end_time - s.start_time for s in self.scenes)
        
        # Allow for minor floating point discrepancies
        # if abs(self.total_duration - calculated_duration) > 0.1: # 0.1 seconds tolerance
        #     raise ValueError(f"Calculated total_duration ({calculated_duration:.1f}s) does not match provided total_duration ({self.total_duration:.1f}s).")
        
        self.total_duration = calculated_duration # Ensure consistency

        return self

class Clips(RootModel[List[Clip]]):
    @field_validator('root', mode='before')
    @classmethod
    def allow_single_dict_or_list(cls, v: Union[Dict[str, Any], List[Dict[str, Any]]]):
        if isinstance(v, dict):
            return [v]  # Wrap single dict in list
        if isinstance(v, list):
            return v
        raise TypeError("Input must be a dict or list of dicts representing Clip(s)")

# Define common hallucinated words/units to remove from numerical values
HALLYUCINATED_UNITS = r'\b(?:seconds?|s|minutes?|min|m|milliseconds?|ms|hours?|h)\b'

def clean_numerical_value(text: str) -> float:
    """
    Robustly extracts a float value from a string, removing common hallucinated words/units.
    """
    # Remove common hallucinated words/units
    cleaned_text = re.sub(HALLYUCINATED_UNITS, '', text, flags=re.IGNORECASE).strip()
    
    # Try to extract a numerical pattern (integer or float)
    # This regex is more comprehensive:
    # - `[-+]?`: optional sign
    # - `(?:\d+\.?\d*|\.\d+)`: matches either (digits.optional_digits) or (.digits)
    # - `(?:[eE][-+]?\d+)?`: optional scientific notation
    match = re.search(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', cleaned_text)
    if match:
        try:
            return float(match.group(0))
        except ValueError as e:
            raise ValueError(f"Could not convert matched string '{match.group(0)}' to float from text: '{text}'") from e
    
    raise ValueError(f"Could not extract a valid numerical value from: '{text}' (cleaned: '{cleaned_text}')")


# Define system and main prompts globally
system_prompt = """
You are an expert video editor and content strategist. Your task is to analyze a video transcript and identify the most engaging and viral-worthy clips.
Extract the start and end timestamps for every scene, calculate the total duration of each scene then calculate the total duration of the clip as u sum of all scenes durations.

CRITICAL REQUIREMENTS:
1. The SUM of all scene durations for each clip MUST be exactly 60-90 seconds long.
4. Every Scene MUST be at least 2 seconds long
2. You MUST respond with ONLY a valid JSON ARRAY - no other text, explanations, or markdown
3. The response must be a JSON object containing the following format:

REQUIRED JSON FORMAT:
{
    "clip_description": "Brief description of the clip",
    "total_duration": "The sum of all scene durations in seconds",
    "reason": "Why this clip was selected",
    "viral_potential_score": 8,
    "scenes": [
        {
            "start_time": 120.0,
            "end_time": 135.0,
            "scene_duration": 15.0,
            "description": "What happens in this scene"
        }
        {
            "start_time": 140.0,
            "end_time": 153.0,
            "scene_duration": 13.0,
            "description": "What happens in this scene"
        }
        {
            "start_time": 160.0,
            "end_time": 195.0,
            "scene_duration": 35.0,
            "description": "What happens in this scene"
        }
        {
            "start_time": 200,
            "end_time": 212.0,
            "scene_duration": 12.0,
            "description": "What happens in this scene"
        }
    ]
}


IMPORTANT: 
- Sum of all scene_duration MUST BE EQUAL to total_duration
- total_duration MUST BE between 60-90 seconds
- No ```json``` markdown blocks
- No explanatory text
- Do not reuse timestamps from this example above! Extract timestamps from the transcript and calculate all the durations with precision!
- Verify that the sum of scene durations is between 60-90 seconds before outputting.
"""

main_prompt = """
Analyze the following video transcript and extract the most engaging, funny clip that is exactly 60-90 seconds long.

VALIDATION CHECKLIST before responding:
1. Sum of all scene_duration is equal to total_duration
2. total_duration is between 60-90 seconds
3. Output is valid JSON
4. No markdown, no explanations, just JSON

Video Transcript:
{transcript}

Storyboarding Data:
{storyboarding_data}

User Instructions:
{user_prompt}

Remember: Output ONLY the JSON. No other text.

/no-think
"""


def setup_logging(video_name: str, video_duration: float, video_quality: str):
    """
    Sets up logging for session, LLM interactions, and errors.
    Logs initial session information.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(config.get('log_dir', 'logs'), f"run_{timestamp}") # Default log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure loggers
    session_logger = logging.getLogger('session')
    llm_logger = logging.getLogger('llm')
    error_logger = logging.getLogger('errors')
    
    # Set level for all loggers
    session_logger.setLevel(logging.INFO)
    llm_logger.setLevel(logging.INFO)
    error_logger.setLevel(logging.ERROR) # Changed to ERROR for actual errors

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if not session_logger.handlers:
        # Create handlers
        session_handler = logging.FileHandler(os.path.join(log_dir, 'session_info.log'))
        llm_handler = logging.FileHandler(os.path.join(log_dir, 'llm_interactions.log'))
        error_handler = logging.FileHandler(os.path.join(log_dir, 'errors.log'))

        # Create formatters and add them to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        session_handler.setFormatter(formatter)
        llm_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)

        # Add handlers to the loggers
        session_logger.addHandler(session_handler)
        llm_logger.addHandler(llm_handler)
        error_logger.addHandler(error_handler)

    # Log session info
    session_logger.info(f"Video: {video_name}")
    session_logger.info(f"Duration: {video_duration}")
    session_logger.info(f"Quality: {video_quality}")
    session_logger.info(f"Start Time: {datetime.now()}")

def cleanup():
    """
    Attempts to clear GPU memory and provides guidance for Ollama model unloading.
    """
    llm_logger = logging.getLogger('llm')
    error_logger = logging.getLogger('errors')

    llm_logger.info("Attempting to clear GPU memory...")
    # Clear PyTorch CUDA cache and run garbage collector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Unload all currently loaded Ollama models
    try:
        primary_llm_model = config.get('llm.model')
        if primary_llm_model and primary_llm_model.startswith("ollama"):
            gpu_manager.unload_ollama_model(primary_llm_model)

        image_llm_model = config.get('llm.image_model')
        if image_llm_model and image_llm_model.startswith("ollama"):
            gpu_manager.unload_ollama_model(image_llm_model)

    except Exception as e:
        error_logger.warning(f"Could not unload Ollama models: {e}", exc_info=True)

    llm_logger.info("Ollama Models cleaned up successfully")
    # Ensure GPU memory is cleared after Ollama models are signaled to unload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1) # Add a small delay to allow memory to be released


def llm_pass(model_name: str, messages: List[Dict[str, Any]], image_path: Optional[str] = None, use_json_format: bool = True) -> str:
    """Send messages to LLM model and return raw response content."""
    llm_logger = logging.getLogger('llm')
    error_logger = logging.getLogger('errors')
    
    # Determine the model to use based on whether an image path is provided
    # This logic is now handled by the caller (robust_llm_json_extraction)
    # The model_name passed here should already be the correct one.
    model_to_use = model_name 
    
    # Create a copy of messages to ensure we don't modify the original list passed in
    messages_to_send = [msg.copy() for msg in messages] 

    try:
        # Prepare image content if an image path is provided
        if image_path:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Find or add the user message and add the image content
            user_message_found = False
            for msg in messages_to_send:
                if msg['role'] == 'user':
                    if 'content' not in msg or msg['content'] is None:
                        msg['content'] = []
                    elif isinstance(msg['content'], str):
                        msg['content'] = [{"type": "text", "text": msg['content']}]
                    elif not isinstance(msg['content'], list):
                        # Handle unexpected content type by converting to string and wrapping
                        msg['content'] = [{"type": "text", "text": str(msg['content'])}]
                    
                    # Explicitly cast msg['content'] to List[Dict[str, Any]] to satisfy type checker
                    content_list: List[Dict[str, Any]] = cast(List[Dict[str, Any]], msg['content'])
                    content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})
                    
                    user_message_found = True
                    break
            
            if not user_message_found:
                messages_to_send.append({
                    'role': 'user',
                    'content': [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}]
                })

        # Create a censored version for logging
        censored_messages = []
        for msg in messages_to_send:
            if msg['role'] == 'user' and isinstance(msg.get('content'), list):
                censored_content = []
                for item in msg['content']:
                    if item.get('type') == 'image_url':
                        censored_content.append({"type": "image_url", "image_url": {"url": "[Base64 Image Data]"}})
                    else:
                        censored_content.append(item)
                censored_msg = msg.copy()
                censored_msg['content'] = censored_content
                censored_messages.append(censored_msg)
            else:
                censored_messages.append(msg)

        llm_logger.debug(f"Messages sent to LLM: {censored_messages}")
        llm_logger.info(f"Using model: {model_to_use}") # Log the determined model
        
        format_param = "json" if use_json_format else None # Apply JSON format if requested

        if model_to_use.startswith("gemini"):
            gemini_api_key = config.get('llm.api_keys.gemini')
            if not gemini_api_key:
                raise ValueError("Gemini API key not found in config (llm.api_keys.gemini).")
            
            # ChatGoogleGenerativeAI expects content in a specific format for multimodal
            # If image_path is provided, ensure content is a list of dicts
            if image_path:
                # Langchain's ChatGoogleGenerativeAI handles the image_url format directly
                # The messages_to_send should already be in the correct format from the image_path handling above
                pass
            
            llm_model = ChatGoogleGenerativeAI(
                model=model_to_use, # Use the determined model
                temperature=0,
                google_api_key=gemini_api_key,
                convert_system_message_to_human=True # Gemini often prefers system messages as human
            )
            # For Gemini, the 'format' parameter is not directly passed to the constructor
            # Instead, it's handled by the prompt or by parsing the output.
            
        else: # Default to Ollama for any other model name
            ollama_keep_alive = config.get('llm.ollama_keep_alive', -1)
            llm_model = ChatOllama(
                model=model_to_use, # Use the determined model
                temperature=0, 
                format=format_param, 
                keep_alive=ollama_keep_alive
            )
        
        response = llm_model.invoke(messages_to_send)
        return str(response.content)
    except Exception as e:
        error_logger.error(f"LLM request failed: {e}", exc_info=True)
        raise


def _extract_json_from_string(text: str) -> str:
    """
    Helper function to extract JSON string from a given text,
    handling markdown fences and bare JSON objects/arrays.
    Prioritizes markdown, then attempts robust bare JSON extraction.
    """
    if not text or not text.strip():
        raise ValueError("Empty or whitespace-only text provided")
    
    text = text.strip()
    
    # 1. Try to find a JSON block enclosed in markdown fences
    markdown_patterns = [
        r'```json\s*\n([\s\S]*?)\n\s*```', # ```json ... ```
        r'```\s*\n([\s\S]*?)\n\s*```',     # ``` ... ``` (generic)
        r'```json([\s\S]*?)```',           # ```json...``` (single line)
        r'```([\s\S]*?)```'                # ```...``` (single line generic)
    ]
    
    for pattern in markdown_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_candidate = match.group(1).strip()
            if json_candidate and (json_candidate.startswith('{') or json_candidate.startswith('[')):
                return json_candidate
    
    # 2. If no markdown fences, try to find a bare JSON object or array
    # Look for the first occurrence of '{' or '['
    json_start_match = re.search(r'[{[]', text)
    if json_start_match:
        start_index = json_start_match.start()
        potential_json_string = text[start_index:].strip()
        
        try:
            # Use json.JSONDecoder to find the end of the JSON
            decoder = json.JSONDecoder()
            # raw_decode returns (object, end_index)
            _, end_index = decoder.raw_decode(potential_json_string)
            # Return the string that was successfully parsed as JSON
            return potential_json_string[:end_index].strip()
        except json.JSONDecodeError:
            # If raw_decode fails, it's not a valid JSON string starting from that point.
            pass
    
    # 3. Last resort: try to parse the entire text as JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    
    raise ValueError(f"Could not extract valid JSON from text: {text[:200]}...")


def sanitize_segments(segments: List[Dict[str, Any]], config: Any) -> List[Clip]:
    """
    Cleans and validates clip data from LLM output using Pydantic models.
    Filters clips based on duration and logs invalid entries.
    """
    llm_logger = logging.getLogger('llm')
    error_logger = logging.getLogger('errors')
    
    cleaned_clips: List[Clip] = []
    
    min_clip_duration = config.get('clip_validation_min', 60)
    max_clip_duration = config.get('clip_validation_max', 90)

    if min_clip_duration == 60 and max_clip_duration == 90: # Only warn if defaults are used because config values are missing
        llm_logger.warning("clip_validation_min or clip_validation_max not found in config. Using default values (60-90s).")

    for clip_data in segments:
        try:
            # Attempt to parse the raw clip data into the Pydantic Clip model
            # This handles type conversion, field validation (e.g., viral_potential_score range)
            # and the custom model_validator for scenes and total_duration.
            clip = Clip.model_validate(clip_data)
            
            # Additional duration validation after Pydantic's internal checks
            if not (min_clip_duration <= clip.total_duration <= max_clip_duration):
                llm_logger.warning(
                    f"Skipping clip due to invalid total duration: {clip.total_duration:.1f}s "
                    f"(expected {min_clip_duration}-{max_clip_duration}s) for clip: {clip.clip_description}"
                )
                continue
            
            cleaned_clips.append(clip)

        except ValidationError as e:
            error_logger.error(f"Skipping invalid clip due to Pydantic validation error: {clip_data} - Error: {e}", exc_info=True)
            continue
        except Exception as e:
            error_logger.error(f"Skipping invalid clip due to unexpected error during sanitization: {clip_data} - Error: {e}", exc_info=True)
            continue
            
    return cleaned_clips


def robust_llm_json_extraction(system_prompt: str, user_prompt: str, output_schema: Type[BaseModel], image_path: Optional[str] = None, max_attempts: int = 3, raw_output_mode: bool = False) -> Any:
    """
    A robust, multi-pass approach for extracting JSON data from an LLM, with retry and correction.
    """
    llm_logger = logging.getLogger('llm')
    error_logger = logging.getLogger('errors')
    
    llm_model_name = config.get('llm.model') # Standardize config access
    if not llm_model_name:
        llm_model_name = config.get('llm_model') # Fallback for older config
        if not llm_model_name:
            raise ValueError("LLM model name not found in config (llm.model or llm_model).")

    # Determine the model name to pass to llm_pass based on image_path
    model_to_pass = config.get('llm.image_model') if image_path else llm_model_name

    for attempt in range(max_attempts):
        llm_logger.info(f"Attempting JSON extraction (Pass {attempt + 1}/{max_attempts})...")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        raw_llm_response_content = ""
        try:
            raw_llm_response_content = llm_pass(
                model_to_pass, # Use the determined model name
                messages, 
                image_path=image_path,
                use_json_format=(not image_path)
            )
            
            llm_logger.info(f"LLM Raw Response (Attempt {attempt + 1}):\n{raw_llm_response_content}")

            if raw_output_mode:
                return raw_llm_response_content
            
            json_string = _extract_json_from_string(raw_llm_response_content)
            
            if not json_string:
                raise ValueError("No JSON content found in LLM response.")

            # Pydantic's model_validate_json will handle json.loads internally
            response = output_schema.model_validate_json(json_string)
            
            llm_logger.info(f"Successfully extracted JSON on pass {attempt + 1}.")
            return response
            
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            error_logger.error(f"JSON validation/parsing failed on pass {attempt + 1}: {e}. Raw LLM response: {raw_llm_response_content[:500]}...", exc_info=True)
            
            if attempt < max_attempts - 1:
                llm_logger.info("Attempting LLM self-correction...")
                correction_prompt = f"""
The total_duration in this JSON is wrong: {str(e)[:200]}

Calculate the correct total_duration as a sum of all scene_duration values.

You must respond with ONLY a valid JSON. No explanations, no markdown, no other text.

Required JSON format:
{
    "clip_description": "Brief description of the clip",
    "total_duration": 74.2,
    "reason": "Why this clip was selected",
    "viral_potential_score": 8,
    "scenes": [
        {
            "start_time": 120.6,
            "end_time": 135.6,
            "scene_duration": 15.0,
            "description": "What happens in this scene"
        }
        {
            "start_time": 140.3,
            "end_time": 153.3,
            "scene_duration": 13.0,
            "description": "What happens in this scene"
        }
        {
            "start_time": 160.7,
            "end_time": 195.2,
            "scene_duration": 34.5,
            "description": "What happens in this scene"
        }
        {
            "start_time": 200.7,
            "end_time": 212.4,
            "scene_duration": 11.7,
            "description": "What happens in this scene"
        }
    ]
}

Your previous invalid response was:
{raw_llm_response_content[:300]}...

Provide ONLY the corrected JSON with the correct total_duration value:
"""
                correction_messages = [
                    {"role": "system", "content": "You are a mathematician working with JSON data. You must respond with ONLY valid JSON. No other text."},
                    {"role": "user", "content": correction_prompt.strip()}
                ]
                try:
                    corrected_raw_response = llm_pass(
                        model_to_pass, # Use the determined model name
                        correction_messages,
                        use_json_format=True
                    )
                    
                    llm_logger.info(f"LLM Correction Response: {corrected_raw_response}")
                    
                    corrected_json_string = _extract_json_from_string(corrected_raw_response)
                    if not corrected_json_string:
                        raise ValueError("No JSON content found in corrected LLM response.")
                    
                    corrected_response = output_schema.model_validate_json(corrected_json_string)
                    
                    llm_logger.info("Successfully corrected JSON with LLM self-correction.")
                    return corrected_response
                    
                except (ValidationError, ValueError, json.JSONDecodeError) as correction_e:
                    error_logger.error(f"LLM self-correction failed: {correction_e}", exc_info=True)
                except Exception as correction_e:
                    error_logger.error(f"LLM self-correction failed with unexpected error: {correction_e}", exc_info=True)
            
            if attempt == max_attempts - 1:
                raise ValueError(f"Failed to extract and correct valid JSON after {max_attempts} attempts. Last error: {e}")
                
        except Exception as e:
            error_logger.error(f"Unexpected error during LLM interaction: {e}", exc_info=True)
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Failed due to unexpected error after {max_attempts} attempts: {e}")
            else:
                llm_logger.info(f"Retrying in {config.get('llm_retry_delay', 2)} seconds...")
                time.sleep(config.get('llm_retry_delay', 2))


def get_clips_from_llm(transcript: List[Dict[str, Any]], user_prompt: Optional[str] = None, storyboarding_data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Get clips with retry logic using the robust LLM JSON extraction approach.
    """
    llm_logger = logging.getLogger('llm')
    error_logger = logging.getLogger('errors')
    
    max_retries = config.get('llm_max_retries', 3) # Default to 3 retries
    min_clips_needed = config.get('llm_min_clips_needed', 1) # Default to 1 clip
    max_clips_to_return = config.get('llm_max_clips_to_return', 10) # Default to 10 clips

    for attempt in range(max_retries):
        try:
            llm_logger.info(f"Attempting LLM clip selection ({attempt + 1}/{max_retries})...")
            
            transcript_str = "\n".join([f"[{item.get('start', 0):.1f}s - {item.get('end', 0):.1f}s]: {item.get('text', '')}" for item in transcript])
            
            storyboard_str = json.dumps(storyboarding_data, indent=2) if storyboarding_data else "None provided"
            
            formatted_main_prompt = main_prompt.format(
                transcript=transcript_str,
                storyboarding_data=storyboard_str,
                user_prompt=user_prompt if user_prompt else "No specific user instructions."
            )

            clips_data_root_model = robust_llm_json_extraction(system_prompt, formatted_main_prompt, output_schema=Clips)
            
            # clips_data_root_model is a Clips object, its actual list is in .root
            cleaned_segments = sanitize_segments(clips_data_root_model.root, config)
            
            if len(cleaned_segments) >= min_clips_needed:
                llm_logger.info(f"Successfully extracted {len(cleaned_segments)} valid clips")
                return [clip.model_dump() for clip in cleaned_segments[:max_clips_to_return]] # Convert Pydantic models back to dicts
            else:
                llm_logger.warning(f"Only got {len(cleaned_segments)} valid clips, need at least {min_clips_needed}.")
                if attempt < max_retries - 1:
                    llm_logger.info(f"Retrying in {config.get('llm_retry_delay', 2)} seconds...")
                    time.sleep(config.get('llm_retry_delay', 2))
                    continue
                else:
                    llm_logger.warning(f"Returning {len(cleaned_segments)} clips after {max_retries} attempts, even though less than {min_clips_needed} were found.")
                    return [clip.model_dump() for clip in cleaned_segments[:max_clips_to_return]] if cleaned_segments else []
                    
        except Exception as e:
            error_logger.error(f"Attempt {attempt + 1} failed: {e}", exc_info=True)
            if attempt < max_retries - 1:
                llm_logger.info(f"Retrying in {config.get('llm_retry_delay', 2)} seconds...")
                time.sleep(config.get('llm_retry_delay', 2))
            else:
                raise RuntimeError(f"Failed to extract clips after {max_retries} attempts. Last error: {e}")
    
    return [] # Should not be reached if max_retries > 0 and no exception is raised on last attempt.
