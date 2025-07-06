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
[
    clip_1{
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
    clip_2{
        ...
    }
    clip_3{
        ...
    }
    ...
]

IMPORTANT: 
- Sum of all scene_duration MUST BE EQUAL to total_duration
- total_duration MUST BE between 60-90 seconds
- No ```json``` markdown blocks
- No explanatory text
- Do not reuse timestamps from this example above! Extract timestamps from the transcript and calculate all the durations with precision!
- Verify that the sum of scene durations is between 60-90 seconds before outputting.
"""

main_prompt = """
Analyze the following video transcript and extract 5 of the most engaging, funny clips that are exactly 60-90 seconds long.

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
    llm_logger.setLevel(logging.WARNING)
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
        # Get current active models from config
        current_active_llm_model = config.get('llm.current_active_llm_model')
        if current_active_llm_model and current_active_llm_model.startswith("ollama"):
            gpu_manager.unload_ollama_model(current_active_llm_model)

        current_active_image_model = config.get('llm.current_active_image_model')
        if current_active_image_model and current_active_image_model.startswith("ollama"):
            gpu_manager.unload_ollama_model(current_active_image_model)

    except Exception as e:
        error_logger.warning(f"Could not unload Ollama models: {e}", exc_info=True)

    llm_logger.info("Ollama Models cleaned up successfully")
    # Ensure GPU memory is cleared after Ollama models are signaled to unload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1) # Add a small delay to allow memory to be released

    _log_daily_usage() # Log any unsaved usage data on cleanup


# Module-level dictionary to store daily usage data per model config
_daily_usage_data: Dict[str, Dict[str, Any]] = {} # Changed to be keyed by model name
_last_reset_date: str = "" # To store the date of the last reset
_last_request_timestamps: Dict[str, float] = {} # To track last request time for proactive rate limiting

def _initialize_daily_usage():
    """
    Initializes or resets daily usage counters based on the current date.
    Logs previous day's usage if a new day has started.
    """
    global _daily_usage_data, _last_reset_date
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = config.get('log_dir', 'logs')
    usage_log_file = os.path.join(log_dir, 'usage.logs')
    llm_logger = logging.getLogger('llm')

    # Load existing usage data if available and not already loaded for the current session
    if not _last_reset_date and os.path.exists(usage_log_file):
        try:
            with open(usage_log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Try to parse the last line as JSON for the most recent state
                    last_line = lines[-1].strip()
                    if last_line.startswith('{') and last_line.endswith('}'):
                        loaded_data = json.loads(last_line)
                        _last_reset_date = loaded_data.get('last_reset_date', "")
                        # Only load model specific data if it matches the current date
                        if _last_reset_date == current_date:
                            # Load usage data for each model found in the log
                            for model_name, usage_info in loaded_data.items():
                                if model_name != 'last_reset_date':
                                    _daily_usage_data[model_name] = usage_info
                            llm_logger.info(f"Loaded previous daily usage data for {current_date}: {_daily_usage_data}")
                        else:
                            llm_logger.info(f"Previous usage data is for {_last_reset_date}. Starting fresh for {current_date}.")
        except json.JSONDecodeError:
            llm_logger.warning(f"Could not decode JSON from {usage_log_file}. Starting fresh.")
        except Exception as e:
            llm_logger.error(f"Error loading usage data from {usage_log_file}: {e}")

    # Check if it's a new day or if data was not loaded for the current day
    if _last_reset_date != current_date:
        if _last_reset_date: # If there was old data (even if not for current day), log it before resetting
            llm_logger.info(f"New day detected. Logging previous day's usage for {_last_reset_date}:")
            _log_daily_usage(reset=False) # Log without resetting first
        
        # Reset for the current day
        _daily_usage_data = {} # Reset to empty dict
        _last_reset_date = current_date
        llm_logger.info(f"Daily usage counters reset for {current_date}.")
        _log_daily_usage(reset=False) # Log the reset state immediately

def _log_daily_usage(reset: bool = False):
    """
    Logs the current daily usage data to a file.
    If reset is True, it also clears the in-memory counters.
    """
    global _daily_usage_data, _last_reset_date
    log_dir = config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    usage_log_file = os.path.join(log_dir, 'usage.logs')
    
    llm_logger = logging.getLogger('llm')

    if _daily_usage_data:
        try:
            # Combine _daily_usage_data with last_reset_date for logging
            log_entry = {
                'last_reset_date': _last_reset_date,
                **_daily_usage_data
            }
            with open(usage_log_file, 'a') as f: # Append mode
                f.write(json.dumps(log_entry) + "\n")
            llm_logger.info(f"Daily usage data logged to {usage_log_file}")
        except Exception as e:
            llm_logger.error(f"Failed to write daily usage data to file: {e}")
    
    if reset:
        _daily_usage_data = {} # Reset to empty dict
        _last_reset_date = datetime.now().strftime("%Y-%m-%d")
        llm_logger.info("In-memory daily usage counters reset.")


def llm_pass(model_name: str, provider: str, requests_per_minute: float, requests_per_day: float, messages: List[Dict[str, Any]], image_path: Optional[str] = None, use_json_format: bool = True) -> str:
    """Send messages to LLM model and return raw response content."""
    llm_logger = logging.getLogger('llm')
    error_logger = logging.getLogger('errors')
    
    _initialize_daily_usage() # Ensure daily counters are up-to-date

    messages_to_send = [msg.copy() for msg in messages] 

    # Proactive Rate Limiting (Requests Per Minute)
    if requests_per_minute != float('inf'):
        min_interval_between_requests = (60 / requests_per_minute) * 1.1 # 10% buffer
        last_req_time = _last_request_timestamps.get(model_name, 0.0)
        time_since_last_request = time.time() - last_req_time

        if time_since_last_request < min_interval_between_requests:
            sleep_duration = min_interval_between_requests - time_since_last_request
            llm_logger.info(f"Proactive rate limit: Sleeping for {sleep_duration:.2f} seconds for {model_name}...")
            time.sleep(sleep_duration)
    
    _last_request_timestamps[model_name] = time.time() # Update timestamp after potential sleep

    # Proactive Daily Limit Check
    current_requests = _daily_usage_data.get(model_name, {}).get('requests', 0)
    if current_requests >= requests_per_day:
        llm_logger.warning(f"Proactive daily quota check: {model_name} has reached its daily limit ({current_requests}/{requests_per_day}).")
        # This function no longer handles model switching. It just raises an error.
        # The calling function (robust_llm_json_extraction) will handle the switch.
        raise RuntimeError(f"Daily quota exhausted for model: {model_name}")


    try:
        # Prepare image content if an image path is provided
        if image_path:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            user_message_found = False
            for msg in messages_to_send:
                if msg['role'] == 'user':
                    if 'content' not in msg or msg['content'] is None:
                        msg['content'] = []
                    elif isinstance(msg['content'], str):
                        msg['content'] = [{"type": "text", "text": msg['content']}]
                    elif not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": str(msg['content'])}]
                    
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
        llm_logger.info(f"Using model: {model_name}") # Log the determined model
        
        format_param = "json" if use_json_format else None # Apply JSON format if requested

        if provider == "gemini":
            gemini_api_key = config.get('llm.api_keys.gemini')
            if not gemini_api_key:
                raise ValueError("Gemini API key not found in config (llm.api_keys.gemini).")
            
            llm_model = ChatGoogleGenerativeAI(
                model=model_name, # Use the determined model
                temperature=0,
                google_api_key=gemini_api_key,
                convert_system_message_to_human=True # Gemini often prefers system messages as human
            )
            
        elif provider == "ollama": # Default to Ollama for any other model name
            ollama_keep_alive = config.get('llm.ollama_keep_alive', -1)
            llm_model = ChatOllama(
                model=model_name, # Use the determined model
                temperature=0, 
                format=format_param, 
                keep_alive=ollama_keep_alive
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        response = llm_model.invoke(messages_to_send)

        # Update daily usage counters after successful request
        _daily_usage_data.setdefault(model_name, {'requests': 0, 'tokens': 0})
        _daily_usage_data[model_name]['requests'] += 1
        # Safely access usage_metadata from response_metadata if available
        if hasattr(response, 'response_metadata') and 'usage_metadata' in response.response_metadata:
            usage_metadata = response.response_metadata['usage_metadata']
            total_tokens = usage_metadata.get('total_tokens', 0)
            _daily_usage_data[model_name]['tokens'] += total_tokens
            llm_logger.info(f"Updated daily usage for {model_name}: Requests={_daily_usage_data[model_name]['requests']}, Tokens={_daily_usage_data[model_name]['tokens']}")
        else:
            llm_logger.info(f"Updated daily usage for {model_name}: Requests={_daily_usage_data[model_name]['requests']} (token usage not available).")
        _log_daily_usage(reset=False) # Log after each successful request

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
    
    # Determine which model configuration to use
    config_key = 'image_model' if image_path else 'llm_model'
    model_name = config.get(f'llm.current_active_{config_key}')
    
    if not model_name:
        raise ValueError(f"Current active model not found in config for {config_key}.")

    # Get model-specific details from the 'models' dictionary
    all_models = config.get('llm.models')
    if not all_models or not isinstance(all_models, dict):
        raise ValueError("LLM models configuration not found or is invalid.")
    
    model_details = all_models.get(model_name)
    if not model_details:
        raise ValueError(f"Model details not found for '{model_name}' in llm.models config.")

    provider = model_details.get('provider')
    requests_per_minute = model_details.get('requests_per_minute', float('inf'))
    requests_per_day = model_details.get('requests_per_day', float('inf'))

    if not provider:
        raise ValueError(f"Provider not specified for model '{model_name}'.")

    for attempt in range(max_attempts):
        llm_logger.info(f"Attempting JSON extraction (Pass {attempt + 1}/{max_attempts})...")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        raw_llm_response_content = ""
        try:
            raw_llm_response_content = llm_pass(
                model_name=model_name,
                provider=provider,
                requests_per_minute=requests_per_minute,
                requests_per_day=requests_per_day,
                messages=messages, 
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
            
            # Check for daily quota exhaustion and switch models persistently
            if "Daily quota exhausted" in str(e) or "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in str(e):
                llm_logger.warning(f"Daily quota exhausted for {model_name}. Attempting to switch to secondary model.")
                try:
                    config.update_llm_active_model(config_key)
                    # Reload config to get the new active model
                    config._load_config() 
                    # Update model_name and its details for the next attempt
                    model_name = config.get(f'llm.current_active_{config_key}')
                    model_details = config.get(f'llm.models.{model_name}')
                    if not model_details:
                        raise ValueError(f"Model details not found for new active model '{model_name}'.")
                    provider = model_details.get('provider')
                    requests_per_minute = model_details.get('requests_per_minute', float('inf'))
                    requests_per_day = model_details.get('requests_per_day', float('inf'))

                    llm_logger.info(f"Switched to new active model: {model_name}. Retrying...")
                    # Reset attempt counter to retry with the new model
                    attempt = -1 # Will become 0 at the start of the next loop iteration
                    continue
                except Exception as switch_e:
                    error_logger.error(f"Failed to switch LLM model: {switch_e}", exc_info=True)
                    if attempt == max_attempts - 1:
                        raise ValueError(f"Failed to extract and correct valid JSON after {max_attempts} attempts. Last error: {e}") from e
            
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
                        model_name=model_name,
                        provider=provider,
                        requests_per_minute=requests_per_minute,
                        requests_per_day=requests_per_day,
                        messages=correction_messages,
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
