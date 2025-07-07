from typing import Dict, Any, List, Optional
import json
import tiktoken # For token counting

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Counts the number of tokens in a given text using a specified LLM model's tokenizer.
    Defaults to gpt-4 tokenizer.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a common encoding if model_name is not found
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def build_adaptive_prompt(
    base_prompt: str,
    context_data: Dict[str, Any],
    max_tokens: int,
    model_name: str = "gpt-4",
    prioritize_keys: Optional[List[str]] = None,
    compress_strategies: Optional[Dict[str, Any]] = None
) -> str:
    """
    Builds an adaptive prompt by prioritizing relevant data and compressing less critical information
    to stay within a specified token limit.

    Args:
        base_prompt (str): The base prompt template with placeholders.
        context_data (Dict[str, Any]): A dictionary of data to inject into the prompt.
        max_tokens (int): The maximum number of tokens allowed for the final prompt.
        model_name (str): The name of the LLM model to use for token counting.
        prioritize_keys (Optional[List[str]]): List of keys in context_data to prioritize.
        compress_strategies (Optional[Dict[str, Any]]): Dictionary defining compression strategies
                                                        for specific keys (e.g., {'long_text': 'summarize'}).

    Returns:
        str: The optimized prompt string.
    """
    if prioritize_keys is None:
        prioritize_keys = []
    if compress_strategies is None:
        compress_strategies = {}

    # Start with a basic prompt structure, excluding data placeholders for initial token count
    current_prompt = base_prompt
    current_tokens = count_tokens(current_prompt, model_name)

    # Sort context_data keys based on prioritization
    sorted_keys = sorted(context_data.keys(), key=lambda k: prioritize_keys.index(k) if k in prioritize_keys else len(prioritize_keys))

    # Iterate through sorted keys and add data if within token limit
    for key in sorted_keys:
        data = context_data.get(key)
        if data is None:
            continue

        data_str = ""
        if key in compress_strategies:
            strategy = compress_strategies[key]
            if strategy == 'summarize' and isinstance(data, str):
                # Placeholder for actual summarization logic (e.g., calling another LLM)
                # For now, a simple truncation or keyword extraction
                data_str = f"Summary of {key}: {data[:200]}..."
            elif strategy == 'first_n' and isinstance(data, list):
                n = compress_strategies[key].get('n', 3)
                data_str = f"First {n} items of {key}: {json.dumps(data[:n], indent=2)}"
            elif strategy == 'count' and isinstance(data, (list, dict)):
                data_str = f"Count of {key}: {len(data)}"
            else:
                data_str = json.dumps(data, indent=2)
        else:
            data_str = json.dumps(data, indent=2)

        # Temporarily add data to check token count
        temp_prompt = f"{current_prompt}\n\n{key.replace('_', ' ').title()}:\n{data_str}"
        temp_tokens = count_tokens(temp_prompt, model_name)

        if temp_tokens <= max_tokens:
            current_prompt = temp_prompt
            current_tokens = temp_tokens
        else:
            # If adding full data exceeds limit, try to add a summarized version if not already summarized
            if key not in compress_strategies:
                # Attempt a generic compression if no specific strategy was defined
                if isinstance(data, str):
                    truncated_data_str = f"Summary of {key}: {data[:100]}... (truncated)"
                elif isinstance(data, list):
                    truncated_data_str = f"First 3 items of {key}: {json.dumps(data[:3], indent=2)}... (truncated)"
                elif isinstance(data, dict):
                    truncated_data_str = f"Keys of {key}: {list(data.keys())[:5]}... (truncated)"
                else:
                    truncated_data_str = f"Data for {key} (too large to include fully)"
                
                temp_prompt_truncated = f"{base_prompt}\n\n{key.replace('_', ' ').title()}:\n{truncated_data_str}"
                if count_tokens(temp_prompt_truncated, model_name) <= max_tokens:
                    current_prompt = temp_prompt_truncated
                    current_tokens = count_tokens(current_prompt, model_name)
                else:
                    # If even truncated data is too much, just mention its presence
                    current_prompt += f"\n\n{key.replace('_', ' ').title()}: [Data too large to include]"
                    current_tokens = count_tokens(current_prompt, model_name)
            else:
                # If already compressed and still too large, just mention its presence
                current_prompt += f"\n\n{key.replace('_', ' ').title()}: [Summarized data too large]"
                current_tokens = count_tokens(current_prompt, model_name)

    return current_prompt
