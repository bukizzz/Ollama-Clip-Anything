
from agents.base_agent import Agent
from typing import Dict, Any
from llm import llm_interaction

class LLMSelectionAgent(Agent):
    """Agent responsible for selecting engaging clips using an LLM."""

    def __init__(self):
        super().__init__("LLMSelectionAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        transcription = context.get("transcription")
        current_stage = context.get("current_stage")
        clips = context.get("clips")

        print("\n3. Selecting coherent clips using LLM...")
        if current_stage == "transcription_complete":
            user_prompt = context.get("user_prompt")
            clips = llm_interaction.get_clips_with_retry(transcription, user_prompt=user_prompt)
            context.update({
                "clips": clips,
                "current_stage": "llm_selection_complete"
            })
            
        else:
            print("⏩ Skipping LLM clip selection. Loaded from state.")
            
        print(f"✅ Selected {len(clips)} clips:")
        for i, clip in enumerate(clips, 1):
            duration = clip['end'] - clip['start']
            print(f"  Clip {i}: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s) - {clip['text'][:70]}...")
        
        return context
