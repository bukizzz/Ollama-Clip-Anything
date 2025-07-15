from agents.base_agent import Agent
from core.state_manager import set_stage_status
from llm.prompt_utils import build_adaptive_prompt
from llm.llm_interaction import robust_llm_json_extraction # Import the function directly
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field # Import BaseModel and Field

class DynamicEditingAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("DynamicEditingAgent")
        self.config = config
        self.state_manager = state_manager
        # No need for self.llm_interaction = LLMInteraction(config) as we call functions directly

    def execute(self, context):
        stage_name = self.name
        print(f"\nExecuting stage: {stage_name}")

        # --- Pre-flight Check ---
        if context.get('dynamic_editing_decisions') and context.get('pipeline_stages', {}).get(stage_name) == 'complete':
            print(f"✅ Skipping {stage_name}: Dynamic editing decisions already generated.")
            return context

        clips = context.get('current_analysis', {}).get('clips', [])
        transcript = context.get('current_analysis', {}).get('audio_analysis_results', {}).get('transcript', [])
        llm_director_cuts = context.get('current_analysis', {}).get('cut_decisions', [])

        if not clips:
            self.log_error("No clips for dynamic editing.")
            set_stage_status('dynamic_editing', 'skipped', {'reason': 'No clips'})
            return context

        print("✨ Generating dynamic editing decisions...")
        set_stage_status('dynamic_editing', 'running')

        try:
            editing_decisions = []
            all_zoom_events = []

            for clip in clips:
                scenes = clip.get('scenes')
                if not scenes:
                    self.log_warning(f"Clip '{clip.get('clip_description', 'N/A')}' has no scenes. Skipping.")
                    continue

                clip_start = scenes[0].get('start_time')
                clip_end = scenes[-1].get('end_time')

                if clip_start is None or clip_end is None:
                    self.log_warning(f"Clip '{clip.get('clip_description', 'N/A')}' has scenes with missing start/end times. Skipping.")
                    continue
                
                for cut in llm_director_cuts:
                    if clip_start <= cut['start_time'] <= clip_end:
                        editing_decisions.append({'time': cut['start_time'], 'type': 'cut', 'reason': cut['reason']})

                # --- LLM-driven Zoom Event Generation ---
                clip_transcript_segments = [
                    seg for seg in transcript 
                    if seg['start'] >= clip_start and seg['end'] <= clip_end
                ]

                if clip_transcript_segments:
                    transcript_text = " ".join([seg['text'] for seg in clip_transcript_segments])
                    
                    system_prompt = """
                    You are an expert video editor AI. Your task is to identify the single most impactful sentence within the provided transcript for a short video clip. 
                    For this sentence, you need to pinpoint 1-2 key words that would trigger a fast zoom-in, and the end of the sentence that would trigger a fast zoom-out.
                    Provide the output as a JSON object with the following structure:
                    {
                        "impactful_sentence": "The identified impactful sentence.",
                        "zoom_in_word_start_time": <float>, // Start time of the first key word for zoom-in
                        "zoom_out_sentence_end_time": <float> // End time of the impactful sentence for zoom-out
                    }
                    Ensure the times are accurate to the provided transcript segments. If no impactful sentence is found, return an empty JSON object {}.
                    """
                    user_prompt = f"Transcript for clip (start: {clip_start:.2f}s, end: {clip_end:.2f}s):\n\n{transcript_text}\n\nTranscript segments with timestamps:\n{json.dumps(clip_transcript_segments, indent=2)}"

                    # Use the robust_llm_json_extraction function directly
                    # Define a temporary Pydantic model for the expected LLM output for zoom events
                    class ZoomEventLLMOutput(BaseModel):
                        impactful_sentence: str
                        zoom_in_word_start_time: float
                        zoom_out_sentence_end_time: float

                    try:
                        zoom_info_obj = robust_llm_json_extraction(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            output_schema=ZoomEventLLMOutput, # Use the temporary Pydantic model
                            max_attempts=self.config.get('llm_agent_max_retries', 3) # Use agent's max retries
                        )
                        
                        # Convert Pydantic object back to dict for consistency
                        zoom_info = zoom_info_obj.model_dump()

                        if zoom_info and 'zoom_in_word_start_time' in zoom_info and 'zoom_out_sentence_end_time' in zoom_info:
                            zoom_level = 1.1 
                            all_zoom_events.append({
                                'start_time': zoom_info['zoom_in_word_start_time'],
                                'end_time': zoom_info['zoom_out_sentence_end_time'],
                                'zoom_level': zoom_level
                            })
                            self.log_info(f"Generated zoom event for clip: {zoom_info['impactful_sentence']}")
                        else:
                            self.log_warning("LLM did not return valid zoom information for this clip.")
                    except Exception as e: # Catch any exception from robust_llm_json_extraction
                        self.log_error(f"Error during LLM-driven zoom event generation: {e}")

            context['dynamic_editing_decisions'] = editing_decisions
            context['zoom_events'] = all_zoom_events
            print(f"✅ Generated {len(editing_decisions)} dynamic editing decisions and {len(all_zoom_events)} zoom events.")
            set_stage_status('dynamic_editing_complete', 'complete', {'num_decisions': len(editing_decisions), 'num_zoom_events': len(all_zoom_events)})
            return context

        except Exception as e:
            self.log_error(f"Error in DynamicEditingAgent: {e}")
            set_stage_status('dynamic_editing', 'failed', {'reason': str(e)})
            return context
