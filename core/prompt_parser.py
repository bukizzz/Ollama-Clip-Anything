from typing import Dict, Any, List
from llm import llm_interaction
from pydantic import BaseModel, Field

class UserPromptParameters(BaseModel):
    theme: str = Field(description="The main theme extracted from the user's prompt.")
    characters: List[str] = Field(description="A list of characters mentioned in the user's prompt.")
    style: str = Field(description="The desired editing style (e.g., 'standard', 'fast-paced', 'cinematic').")
    keywords: List[str] = Field(description="A list of relevant keywords from the user's prompt.")

def parse_user_prompt(user_prompt: str) -> Dict[str, Any]:
    """Parses a user's natural language prompt into structured parameters using an LLM."""
    print(f"📝 Parsing user prompt: {user_prompt}")

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

    Provide your response as a JSON object adhering to the UserPromptParameters schema.
    """

    try:
        parsed_data_model = llm_interaction.robust_llm_json_extraction(
            system_prompt="You are an AI assistant that extracts structured information from user prompts.",
            user_prompt=llm_prompt.strip(),
            output_schema=UserPromptParameters
        )
        print("✅ User prompt parsed by LLM.")
        return parsed_data_model.model_dump()
    except Exception as e:
        print(f"❌ \033[91mFailed to parse user prompt with LLM: {e}\033[0m")
        return {
            "theme": "default",
            "characters": [],
            "style": "standard",
            "keywords": user_prompt.split() # Fallback to simple split
        }
