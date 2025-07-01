from typing import Dict, Any
from llm import llm_interaction
import json

def parse_user_prompt(user_prompt: str) -> Dict[str, Any]:
    """Parses a user's natural language prompt into structured parameters using an LLM."""
    print(f"Parsing user prompt: {user_prompt}")

    if not user_prompt:
        return {
            "theme": "default",
            "characters": [],
            "style": "standard",
            "keywords": []
        }

    llm_prompt = f"""
    Analyze the following user prompt and extract structured parameters. 
    Identify the main theme, any mentioned characters, the desired editing style, and relevant keywords.

    User Prompt: "{user_prompt}"

    Return the output as a JSON object with the following keys:
    - "theme": (string, e.g., "sports", "vlog", "news", or "default" if not specified)
    - "characters": (array of strings, e.g., ["John", "Jane"] or empty array if none)
    - "style": (string, e.g., "dramatic", "fast-paced", "calm", "standard" if not specified)
    - "keywords": (array of strings, important words from the prompt)
    """

    try:
        response = llm_interaction.llm_pass(llm_interaction.LLM_MODEL, [
            {"role": "system", "content": "You are an AI assistant that extracts structured information from user prompts."},
            {"role": "user", "content": llm_prompt.strip()}
        ])
        
        parsed_data = llm_interaction.extract_json_from_text(response)
        print("✅ User prompt parsed by LLM.")
        return parsed_data
    except Exception as e:
        print(f"Failed to parse user prompt with LLM: {e}")
        return {
            "theme": "default",
            "characters": [],
            "style": "standard",
            "keywords": user_prompt.split() # Fallback to simple split
        }