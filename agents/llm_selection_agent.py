from agents.base_agent import Agent
from typing import Dict, Any
from llm import llm_interaction

from core.state_manager import set_stage_status

class LLMSelectionAgent(Agent):
    """Agent responsible for selecting engaging clips using an LLM."""

    def __init__(self, agent_config, state_manager):
        super().__init__("LLMSelectionAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.llm_selection_config = self.config.get('llm_selection', {})

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        transcription = context.get("transcription")
        
        user_prompt = context.get("user_prompt")
        storyboarding_data = context.get('storyboarding_data') # Changed from qwen_vision_analysis_results

        print("\n🧠 \u001b[94mSelecting coherent clips using LLM...\u001b[0m")
        set_stage_status('llm_selection', 'running')

        if transcription is None:
            self.log_error("Transcription data is missing from context. Cannot select clips.")
            set_stage_status('llm_selection', 'failed', {'reason': 'Missing transcription'})
            return context
        
        try:
            clips = llm_interaction.get_clips_from_llm(
                transcript=transcription,
                user_prompt=user_prompt,
                storyboarding_data=storyboarding_data # Changed from qwen_vision_analysis_results
            )

            context.update({
                "clips": clips,
                "current_stage": "llm_selection_complete"
            })
            
            print(f"✅ Selected {len(clips)} clips:")
            
            set_stage_status('llm_selection_complete', 'complete', {'num_clips': len(clips)})
            llm_interaction.cleanup() # Clear VRAM after successful completion
            return context

        except Exception as e:
            self.log_error(f"Failed to select clips with LLM: {e}")
            set_stage_status('llm_selection', 'failed', {'reason': str(e)})
            return context
