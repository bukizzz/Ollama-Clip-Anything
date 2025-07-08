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
from core.resource_manager import resource_manager # Import resource_manager
from core.monitoring import monitor # Import monitor

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI # Added for Gemini
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator, model_validator

from google.api_core.exceptions import ResourceExhausted # Import the specific exception

# Initialize loggers at the module level
llm_logger = logging.getLogger('llm')
error_logger = logging.getLogger('errors')

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED" # CLOSED, OPEN, HALF_OPEN

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"

    def allow_request(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True # Allow one request to test recovery
            return False
        elif self.state == "HALF_OPEN":
            return True # Allow the single test request
        return False

    def __repr__(self):
        return f"CircuitBreaker(state='{self.state}', failures={self.failures}, last_failure_time={self.last_failure_time})"

# Global circuit breaker instance
llm_circuit_breaker = CircuitBreaker()

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
        
        # Ensure scenes are chronological and do not overlap
        # First, sort them to ensure the loop logic is sound, even if LLM provides unsorted
        self.scenes.sort(key=lambda s: s.start_time)

        epsilon = 1e-6 # Small tolerance for floating point comparisons

        for i in range(len(self.scenes) - 1):
            current_scene = self.scenes[i]
            next_scene = self.scenes[i+1]

            # Check for valid numeric types for times
            if not isinstance(current_scene.start_time, (int, float)) or \
               not isinstance(current_scene.end_time, (int, float)) or \
               not isinstance(next_scene.start_time, (int, float)):
                raise ValueError(f"Invalid time types in scenes. Scene {i} times: ({current_scene.start_time}, {current_scene.end_time}), Scene {i+1} start: {next_scene.start_time}. All times must be numeric.")

            # Check for overlap or non-chronological order
            if current_scene.end_time > next_scene.start_time + epsilon:
                raise ValueError(f"Scenes are overlapping or non-chronological: Scene {i} ends at {current_scene.end_time:.4f}s, but Scene {i+1} starts at {next_scene.start_time:.4f}s. Overlap detected.")
            
            # Ensure strictly increasing start times (no duplicate start times)
            if current_scene.start_time >= next_scene.start_time:
                raise ValueError(f"Scenes are not strictly chronological: Scene {i} starts at {current_scene.start_time:.4f}s, but Scene {i+1} starts at {next_scene.start_time:.4f}s. Duplicate or out-of-order start time.")
        
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

# New Pydantic model for raw text suggestions
class Recommendation(BaseModel):
    recommendation: str = Field(description="An actionable recommendation to enhance virality.")

class SuggestionsResponse(BaseModel):
    suggestions_text: str = Field(description="Raw text containing B-roll suggestions.")

class TranscriptSummary(BaseModel):
    summary: str = Field(description="A concise, sentence-level summary of the entire transcript.")
    key_phrases: List[str] = Field(description="A list of important keywords and phrases from the transcript.")
    main_topics: List[str] = Field(description="A list of the main topics discussed in the transcript.")

# Define common hallucinated words/units to remove from numerical values
HALLYUCINATED_UNITS = r'\b(?:seconds?|s|minutes?|min|m|milliseconds?|ms|hours?|h)\b'

def clean_numerical_value(text: str) -> float:
    """
    Robustly extracts a float value from a string, removing common hallucinated words/units.
    """
    # Ensure text is a string before processing
    if not isinstance(text, str):
        if isinstance(text, (int, float)):
            return float(text)
        raise TypeError(f"Expected string, int, or float for clean_numerical_value, got {type(text)}")

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
3. Scenes must be in chronological order and must not overlap.
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
    llm_logger.setLevel(logging.INFO) # Changed to INFO for debugging
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
        llm_logger.addHandler(llm_handler) # Corrected: pass handler, not logger
        error_logger.addHandler(error_handler) # Corrected: pass handler, not logger

    # Log session info
    session_logger.info(f"Video: {video_name}")
    session_logger.info(f"Duration: {video_duration}")
    session_logger.info(f"Quality: {video_quality}")
    session_logger.info(f"Start Time: {datetime.now()}")

def cleanup():
    """
    Attempts to clear GPU memory and provides guidance for Ollama model unloading.
    """
    # Use the module-level loggers
    llm_logger.info("Attempting to clear GPU memory and unload models via resource manager...")
    try:
        resource_manager.unload_all_models()
    except Exception as e:
        error_logger.warning(f"Could not unload models via resource manager: {e}", exc_info=True)

    llm_logger.info("Models cleaned up successfully.")
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
_last_token_timestamps: Dict[str, float] = {} # To track last token usage time for proactive rate limiting

def _initialize_daily_usage():
    """
    Initializes or resets daily usage counters based on the current date.
    Logs previous day's usage if a new day has started.
    """
    global _daily_usage_data, _last_reset_date
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = config.get('log_dir', 'logs')
    usage_log_file = os.path.join(log_dir, 'usage.logs')
    daily_summary_log_file = os.path.join(log_dir, 'daily_usage.log')

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
            _write_daily_summary_log(_last_reset_date, _daily_usage_data) # Log summary for previous day
            _log_daily_usage(reset=False) # Log raw usage without resetting first
        
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
    
    # Use the module-level llm_logger
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

def _write_daily_summary_log(date: str, usage_data: Dict[str, Dict[str, Any]]):
    """
    Writes a daily summary of LLM usage to daily_usage.log.
    """
    log_dir = config.get('log_dir', 'logs')
    daily_summary_log_file = os.path.join(log_dir, 'daily_usage.log')

    try:
        with open(daily_summary_log_file, 'a') as f:
            f.write(f"{date}\n")
            for model_name, data in usage_data.items():
                requests = data.get('requests', 0)
                input_tokens = data.get('input_tokens', 0)
                output_tokens = data.get('output_tokens', 0)
                limit_hits = data.get('limit_hits', {})
                tpm_hits = limit_hits.get('tpm', 0)
                rpm_hits = limit_hits.get('rpm', 0)
                rpd_hits = limit_hits.get('rpd', 0)

                summary_line = (
                    f"- {model_name} - {requests} Requests - {input_tokens} Input Tokens - "
                    f"{output_tokens} Output Tokens - Limit hit counter : {tpm_hits} TPM, {rpm_hits} RPM, {rpd_hits} RPD\n"
                )
                f.write(summary_line)
            f.write("\n") # Add a blank line for separation
        llm_logger.info(f"Daily summary logged to {daily_summary_log_file} for {date}")
    except Exception as e:
        llm_logger.error(f"Failed to write daily summary log: {e}")


def llm_pass(model_name: str, provider: str, requests_per_minute: float, requests_per_day: float, tokens_per_minute: float, messages: List[Dict[str, Any]], image_path: Optional[str] = None, use_json_format: bool = True) -> str:
    """Send messages to LLM model and return raw response content."""
    # Use the module-level loggers
    
    _initialize_daily_usage() # Ensure daily counters are up-to-date

    # Check circuit breaker state
    if not llm_circuit_breaker.allow_request():
        llm_logger.warning(f"Circuit breaker is OPEN for {model_name}. Denying request.")
        raise RuntimeError(f"Circuit breaker is OPEN for model: {model_name}")

    messages_to_send = [msg.copy() for msg in messages] 

    # Proactive Rate Limiting (Requests Per Minute)
    if requests_per_minute != float('inf'):
        min_interval_between_requests = (60 / requests_per_minute) * 1.1 # 10% buffer
        last_req_time = _last_request_timestamps.get(model_name, 0.0)
        time_since_last_request = time.time() - last_req_time

        if time_since_last_request < min_interval_between_requests:
            sleep_duration = min_interval_between_requests - time_since_last_request
            llm_logger.info(f"Proactive rate limit: Sleeping for {sleep_duration:.2f} seconds for {model_name} (requests)...")
            time.sleep(sleep_duration)
    
    _last_request_timestamps[model_name] = time.time() # Update timestamp after potential sleep

    # Proactive Rate Limiting (Tokens Per Minute)
    if tokens_per_minute != float('inf'):
        min_interval_between_token_usage = (60 / tokens_per_minute) * 1.1 # 10% buffer
        last_token_time = _last_token_timestamps.get(model_name, 0.0)
        time_since_last_token_usage = time.time() - last_token_time

        # This check is more complex as we don't know token usage *before* the request.
        # We'll rely on the API's rate limiting for this, but keep the timestamp updated.
        # The primary proactive check for tokens will be handled by the daily limit.
        if time_since_last_token_usage < min_interval_between_token_usage:
            # This is a placeholder for more sophisticated token-based proactive limiting
            # which would require estimating tokens before sending the request.
            # For now, we'll just ensure the timestamp is updated.
            pass
    
    _last_token_timestamps[model_name] = time.time() # Update timestamp after potential sleep

    # Proactive Daily Limit Check (Requests)
    current_requests = _daily_usage_data.get(model_name, {}).get('requests', 0)
    if current_requests >= requests_per_day:
        llm_logger.warning(f"Proactive daily quota check: {model_name} has reached its daily request limit ({current_requests}/{requests_per_day}).")
        raise RuntimeError(f"Daily request quota exhausted for model: {model_name}")

    # Proactive Daily Limit Check (Tokens) - This would require estimating tokens before sending
    # For now, we'll let the API handle token exhaustion and catch the error.


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

        # Instantiate the LLM model here to ensure it picks up the latest active model
        if provider == "gemini":
            gemini_api_key = config.get('llm.api_keys.gemini')
            if not gemini_api_key:
                raise ValueError("Gemini API key not found in config (llm.api_keys.gemini).")
            
            llm_model = ChatGoogleGenerativeAI(
                model=model_name, # Use the determined model
                temperature=0,
                google_api_key=gemini_api_key,
                convert_system_message_to_human=True, # Gemini often prefers system messages as human
                max_retries=0 # Disable internal retries for immediate error propagation
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
        _daily_usage_data.setdefault(model_name, {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'limit_hits': {'tpm': 0, 'rpm': 0, 'rpd': 0}})
        _daily_usage_data[model_name]['requests'] += 1
        
        # Extract and log token counts for all providers if usage_metadata is available
        if hasattr(response, 'response_metadata') and 'usage_metadata' in response.response_metadata:
            usage_metadata = response.response_metadata['usage_metadata']
            input_tokens = usage_metadata.get('prompt_token_count', 0)
            output_tokens = usage_metadata.get('candidates_token_count', 0)
            _daily_usage_data[model_name]['input_tokens'] += input_tokens
            _daily_usage_data[model_name]['output_tokens'] += output_tokens
            llm_logger.info(f"Updated daily usage for {model_name}: Requests={_daily_usage_data[model_name]['requests']}, Input Tokens={_daily_usage_data[model_name]['input_tokens']}, Output Tokens={_daily_usage_data[model_name]['output_tokens']}")
            monitor.record_token_usage(input_tokens, output_tokens, model_name) # Record token usage
            monitor.estimate_cost(model_name, input_tokens, output_tokens, provider) # Estimate cost
        else:
            llm_logger.info(f"Updated daily usage for {model_name}: Requests={_daily_usage_data[model_name]['requests']} (token usage not available in response metadata for this provider/response).")
        _log_daily_usage(reset=False) # Log after each successful request

        return str(response.content)
    except ResourceExhausted as e:
        # Specific handling for Gemini ResourceExhausted errors
        if "quota_metric: \"generativelanguage.googleapis.com/generate_content_free_tier_input_token_count\"" in str(e):
            error_logger.error(f"LLM request failed due to token quota exhaustion: {e}", exc_info=True)
            raise RuntimeError(f"Token quota exhausted for model: {model_name}") from e
        else:
            error_logger.error(f"LLM request failed with ResourceExhausted error: {e}", exc_info=True)
            raise # Re-raise other ResourceExhausted errors
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
    # Use the module-level loggers
    
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
    # Use the module-level loggers
    
    # Determine which model configuration to use
    config_key = 'image_model' if image_path else 'llm_model'
    
    # Get initial model details
    model_name = config.get(f'llm.current_active_{config_key}')
    if not model_name:
        raise ValueError(f"Current active model not found in config for {config_key}.")

    all_models = config.get('llm.models')
    if not all_models or not isinstance(all_models, dict):
        raise ValueError("LLM models configuration not found or is invalid.")
    
    model_details = all_models.get(model_name)
    if not model_details:
        raise ValueError(f"Model details not found for '{model_name}' in llm.models config.")

    provider = model_details.get('provider')
    requests_per_minute = model_details.get('requests_per_minute', float('inf'))
    requests_per_day = model_details.get('requests_per_day', float('inf'))
    tokens_per_minute = model_details.get('tokens_per_minute', float('inf'))

    if not provider:
        raise ValueError(f"Provider not specified for model '{model_name}'.")

    current_attempt = 0
    retry_delay = config.get('llm_retry_delay', 2) # Initial retry delay
    while current_attempt < max_attempts:
        llm_logger.info(f"Attempting JSON extraction (Pass {current_attempt + 1}/{max_attempts})...")
        
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
                tokens_per_minute=tokens_per_minute,
                messages=messages, 
                image_path=image_path,
                use_json_format=(not raw_output_mode)
            )
            
            llm_logger.info(f"LLM Raw Response (Attempt {current_attempt + 1}):\n{raw_llm_response_content}")

            if raw_output_mode:
                return output_schema(suggestions_text=raw_llm_response_content)
            
            json_string = _extract_json_from_string(raw_llm_response_content)
            
            if not json_string:
                raise ValueError("No JSON content found in LLM response.")

            # Parse the JSON string into a Python object
            parsed_data = json.loads(json_string)
            llm_logger.info(f"Parsed data before sorting: {json.dumps(parsed_data, indent=2)}")

            # If the output schema is for Clips (list of Clip), sort scenes within each clip and filter invalid ones
            if output_schema == Clips and isinstance(parsed_data, list):
                processed_clips_data = []
                for i, clip_data in enumerate(parsed_data):
                    if 'scenes' in clip_data and isinstance(clip_data['scenes'], list):
                        valid_scenes = []
                        # Sort scenes first to handle chronological validation correctly
                        clip_data['scenes'].sort(key=lambda s: s.get('start_time', 0))

                        for j, scene in enumerate(clip_data['scenes']):
                            # Robustly clean start_time and end_time
                            try:
                                cleaned_start_time = clean_numerical_value(str(scene.get('start_time')))
                            except (ValueError, TypeError) as e:
                                error_logger.warning(f"Scene {j} in clip {i} has invalid start_time '{scene.get('start_time')}': {e}. Skipping scene.")
                                continue
                            
                            cleaned_end_time = None
                            if 'end_time' in scene and scene['end_time'] is not None:
                                try:
                                    cleaned_end_time = clean_numerical_value(str(scene['end_time']))
                                except (ValueError, TypeError) as e:
                                    error_logger.warning(f"Scene {j} in clip {i} has invalid end_time '{scene.get('end_time')}': {e}. Attempting to derive from scene_duration.")
                            
                            # If end_time is still problematic, try to derive it from scene_duration
                            if cleaned_end_time is None and 'scene_duration' in scene and scene['scene_duration'] is not None:
                                try:
                                    cleaned_scene_duration = clean_numerical_value(str(scene['scene_duration']))
                                    cleaned_end_time = cleaned_start_time + cleaned_scene_duration
                                    llm_logger.info(f"Derived end_time for scene {j} in clip {i}: {cleaned_end_time:.2f}s (from start_time + scene_duration).")
                                except (ValueError, TypeError) as e:
                                    error_logger.warning(f"Scene {j} in clip {i} has invalid scene_duration '{scene.get('scene_duration')}': {e}. Cannot derive end_time. Skipping scene.")
                                    continue
                            
                            if cleaned_end_time is None:
                                error_logger.warning(f"Scene {j} in clip {i} has no valid end_time and cannot be derived. Skipping scene.")
                                continue

                            # Update scene with cleaned times
                            scene['start_time'] = cleaned_start_time
                            scene['end_time'] = cleaned_end_time
                            
                            # Enforce minimum duration (e.g., 2 seconds as per prompt requirement)
                            if cleaned_end_time - cleaned_start_time < 2.0 - 1e-6: # Allow for floating point inaccuracies
                                error_logger.warning(f"Scene {j} in clip {i} has duration less than 2 seconds ({cleaned_end_time - cleaned_start_time:.2f}s). Skipping scene.")
                                continue
                            
                            valid_scenes.append(scene)
                        
                        if valid_scenes:
                            clip_data['scenes'] = valid_scenes
                            # Recalculate total_duration based on valid scenes
                            clip_data['total_duration'] = sum(s['end_time'] - s['start_time'] for s in valid_scenes)
                            processed_clips_data.append(clip_data)
                        else:
                            error_logger.warning(f"Clip {i} has no valid scenes after filtering. Skipping clip.")
                
                # Re-serialize the processed data back to a JSON string
                json_string = json.dumps(processed_clips_data)
                llm_logger.info(f"JSON string after sorting and filtering: {json_string}")

            response = output_schema.model_validate_json(json_string)
            
            llm_logger.info(f"Successfully extracted JSON on pass {current_attempt + 1}.")
            return response
            
        except (ValidationError, ValueError, json.JSONDecodeError, RuntimeError) as e:
            error_logger.error(f"JSON validation/parsing failed on pass {current_attempt + 1}: {e}. Raw LLM response: {raw_llm_response_content[:500]}...", exc_info=True)
            
            if "Daily request quota exhausted" in str(e) or "Token quota exhausted" in str(e):
                llm_logger.warning(f"Quota exhausted for {model_name}. Attempting to switch to secondary model.")
                try:
                    priority_list_key = f"{config_key}s_priority"
                    current_active_model_key = f"current_active_{config_key}"
                    
                    priority_list = config.get(f'llm.{priority_list_key}')
                    current_index = priority_list.index(model_name)
                    next_index = (current_index + 1) % len(priority_list)
                    new_active_model = priority_list[next_index]
                    
                    config.set(f'llm.{current_active_model_key}', new_active_model)
                    
                    # Re-fetch model details for the next attempt
                    model_name = config.get(f'llm.current_active_{config_key}')
                    model_details = all_models.get(model_name) # Use all_models here
                    if not model_details:
                        raise ValueError(f"Model details not found for new active model '{model_name}'.")
                    provider = model_details.get('provider')
                    requests_per_minute = model_details.get('requests_per_minute', float('inf'))
                    requests_per_day = model_details.get('requests_per_day', float('inf'))
                    tokens_per_minute = model_details.get('tokens_per_minute', float('inf'))

                    llm_logger.info(f"Switched to new active model: {model_name}. Retrying...")
                    # Do NOT increment current_attempt here, as we are retrying with a new model
                    # The while loop condition will be checked again.
                    continue # This will restart the while loop with the same current_attempt, but new model
                except Exception as switch_e:
                    error_logger.error(f"Failed to switch LLM model: {switch_e}", exc_info=True)
                    # If model switch fails, increment attempt and proceed to next check
                    current_attempt += 1
                    if current_attempt == max_attempts:
                        raise ValueError(f"Failed to extract and correct valid JSON after {max_attempts} attempts. Last error: {e}") from e
                    continue # Continue to next attempt if switch failed but not max attempts
            
            # If not a quota error, or if model switch failed, try self-correction
            if current_attempt < max_attempts - 1:
                llm_logger.info("Attempting LLM self-correction...")
                correction_prompt = f"""
The total_duration in this JSON is wrong: {str(e)[:200]}

Calculate the correct total_duration as a sum of all scene_duration values.

You must respond with ONLY a valid JSON. No explanations, no markdown, no other text.

Required JSON format:
{{
    "clip_description": "Brief description of the clip",
    "total_duration": 74.2,
    "reason": "Why this clip was selected",
    "viral_potential_score": 8,
    "scenes": [
        {{
            "start_time": 120.6,
            "end_time": 135.6,
            "scene_duration": 15.0,
            "description": "What happens in this scene"
        }}
        {{
            "start_time": 140.3,
            "end_time": 153.3,
            "scene_duration": 13.0,
            "description": "What happens in this scene"
        }}
        {{
            "start_time": 160.7,
            "end_time": 195.2,
            "scene_duration": 34.5,
            "description": "What happens in this scene"
        }}
        {{
            "start_time": 200.7,
            "end_time": 212.4,
            "scene_duration": 11.7,
            "description": "What happens in this scene"
        }}
    ]
}}

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
                        tokens_per_minute=tokens_per_minute,
                        messages=correction_messages,
                        use_json_format=True
                    )
                    
                    corrected_json_string = _extract_json_from_string(corrected_raw_response)
                    if not corrected_json_string:
                        raise ValueError("No JSON content found in corrected LLM response.")
                    
                    corrected_response = output_schema.model_validate_json(corrected_json_string)
                    
                    llm_logger.info("Successfully corrected JSON with LLM self-correction.")
                    llm_circuit_breaker.record_success() # Record success for self-correction
                    return corrected_response
                    
                except (ValidationError, ValueError, json.JSONDecodeError, RuntimeError) as correction_e:
                    llm_circuit_breaker.record_failure() # Record failure for self-correction
                    error_logger.error(f"LLM self-correction failed: {correction_e}", exc_info=True)
                except Exception as correction_e:
                    llm_circuit_breaker.record_failure() # Record failure for self-correction
                    error_logger.error(f"LLM self-correction failed with unexpected error: {correction_e}", exc_info=True)
            
            current_attempt += 1 # Increment attempt for non-quota errors or failed self-correction
            if current_attempt == max_attempts:
                raise ValueError(f"Failed to extract and correct valid JSON after {max_attempts} attempts. Last error: {e}")
                
        except Exception as e: # This catches any other unexpected errors not caught above
            llm_circuit_breaker.record_failure() # Record failure for unexpected errors
            error_logger.error(f"Unexpected error during LLM interaction: {e}", exc_info=True)
            current_attempt += 1 # Increment attempt for unexpected errors
            if current_attempt == max_attempts:
                raise RuntimeError(f"Failed due to unexpected error after {max_attempts} attempts: {e}")
            else:
                llm_logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
    
    return [] # Should not be reached if max_retries > 0 and no exception is raised on last attempt.


def summarize_transcript_with_llm(transcript: List[Dict[str, Any]]) -> TranscriptSummary:
    """
    Summarizes the full transcript using an LLM to extract a concise summary, key phrases, and main topics.
    """
    # Use the module-level loggers

    llm_logger.info("Starting transcript summarization with LLM...")

    transcript_text = "\n".join([segment['text'] for segment in transcript])

    system_prompt = """
You are an expert summarization AI. Your task is to provide a concise, sentence-level summary of the given transcript, extract key phrases, and identify the main topics discussed.
You MUST respond with ONLY a valid JSON object. No other text, explanations, or markdown.

REQUIRED JSON FORMAT:
{
    "summary": "A concise, sentence-level summary of the entire transcript.",
    "key_phrases": [
        "key phrase 1",
        "key phrase 2"
    ],
    "main_topics": [
        "main topic 1",
        "main topic 2"
    ]
}
"""

    user_prompt = f"""
Summarize the following transcript, extract key phrases, and identify main topics:

Transcript:
{transcript_text}
"""

    try:
        summary_data = robust_llm_json_extraction(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=TranscriptSummary,
            max_attempts=3
        )
        llm_logger.info("Successfully summarized transcript with LLM.")
        return summary_data
    except Exception as e:
        error_logger.error(f"Failed to summarize transcript with LLM: {e}", exc_info=True)
        raise RuntimeError(f"Transcript summarization failed: {e}") from e


def get_clips_from_llm(user_prompt: Optional[str] = None, storyboarding_data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:

    """
    Get clips with retry logic using the robust LLM JSON extraction approach.
    """
    # Use the module-level loggers
    
    max_retries = config.get('llm_max_retries', 3) # Default to 3 retries
    min_clips_needed = config.get('llm_min_clips_needed', 1) # Default to 1 clip
    max_clips_to_return = config.get('llm_max_clips_to_return', 10) # Default to 10 clips

    for attempt in range(max_retries):
        try:
            llm_logger.info(f"Attempting LLM clip selection ({attempt + 1}/{max_retries})...")
            
            # The transcript is now expected to be part of the user_prompt (formatted_llm_prompt)
            # from LLMSelectionAgent, so we don't need to format it here.
            
            storyboard_str = json.dumps(storyboarding_data, indent=2) if storyboarding_data else "None provided"
            
            formatted_main_prompt = main_prompt.format(
                transcript="", # No longer directly used, but kept for format string compatibility
                storyboarding_data=storyboard_str,
                user_prompt=user_prompt if user_prompt else "No specific user instructions."
            )

            clips_data_root_model = robust_llm_json_extraction(system_prompt, formatted_main_prompt, output_schema=Clips)


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

def get_viral_recommendations_batch(recommendation_contexts: List[Dict[str, Any]]) -> List[str]:
    """
    Generates viral potential recommendations for multiple clips in a batch.
    """
    # Use the module-level loggers
    
    recommendations = []
    for context_data in recommendation_contexts:
        start = context_data['start']
        end = context_data['end']
        viral_score = context_data['viral_score']
        clip_description = context_data['clip_description']
        avg_engagement = context_data['avg_engagement']
        hook_score = context_data['hook_score']
        sentiment = context_data['sentiment']
        sentiment_score = context_data['sentiment_score']

        prompt = f"""
        A video clip from {start:.2f}s to {end:.2f}s has a viral potential score of {viral_score:.2f}/10.
        Content: "{clip_description}"
        
        Additional context for virality assessment:
        - Average Engagement Score: {avg_engagement:.2f}
        - Hook/Quotability Score: {hook_score:.2f}
        - Dominant Audio Sentiment: {sentiment} (Score: {sentiment_score:.2f})
        
        Provide a brief, actionable recommendation to enhance its virality, considering the above context.
        
        Your response MUST be a JSON object matching the following Pydantic schema:
        {{
            "recommendation": "string"
        }}
        """
        try:
            recommendation_obj = robust_llm_json_extraction(
                system_prompt="You are an expert in generating actionable recommendations for video virality.",
                user_prompt=prompt,
                output_schema=Recommendation
            )
            recommendations.append(recommendation_obj.recommendation)
        except Exception as e:
            error_logger.error(f"Failed to get recommendation for clip {start:.2f}s - {end:.2f}s: {e}", exc_info=True)
            recommendations.append("N/A") # Fallback
    return recommendations
