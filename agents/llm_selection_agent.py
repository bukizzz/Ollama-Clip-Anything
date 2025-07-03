from agents.base_agent import Agent
from typing import Dict, Any
from llm import llm_interaction
import json
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
        storyboard_data = context.get("storyboard_data")
        qwen_vision_analysis_results = context.get('qwen_vision_analysis_results')

        print("\n🧠 \u001b[94mSelecting coherent clips using LLM...\u001b[0m")
        set_stage_status('llm_selection', 'running')

        if transcription is None:
            self.log_error("Transcription data is missing from context. Cannot select clips.")
            set_stage_status('llm_selection', 'failed', {'reason': 'Missing transcription'})
            return context

        max_retries = self.llm_selection_config.get('max_retries', 3)  # Default to 3 retries
        min_clips = self.llm_selection_config.get('min_clips', 1)    # Default to 1 minimum clip
        json_processing_model = self.llm_selection_config.get('json_processing_model', "deepseek-coder:6.7b")

        for attempt in range(max_retries):
            try:
                self.log_info(f"🧠 \u001b[94mAttempt {attempt + 1}/{max_retries}: Identifying coherent clips and forming JSON with {json_processing_model}...\u001b[0m")
                clip_duration_min = self.config.get('clip_duration_min', 60) # Default to 60 seconds if not found
                clip_duration_max = self.config.get('clip_duration_max', 90) # Default to 90 seconds if not found
                pass1_prompt = f"""
                Analyze the following video data and identify coherent, self-contained story segments or "whole stories" that flow naturally, without abrupt cuts at the beginning or end. Focus on narrative completeness, key discussions, demonstrations, emotional moments, and clear narrative arcs. Avoid boring conversation parts. Argumentative or funny segments are preferred. All clips you create MUST BE OVER 60 seconds long and no longer than 90 seconds. 

                Consider the following data:

                Transcription:
                {json.dumps(transcription, indent=2)}

                Storyboard Data:
                {json.dumps(storyboard_data, indent=2)}

                Qwen-VL Vision Analysis Results:
                {json.dumps(qwen_vision_analysis_results, indent=2)}

                Output a JSON array of 20 different objects. Each object must strictly adhere to the following format, including data types:
                ```json
                [
                  {{
                    "start": 0.0,
                    "end": 0.0,
                    "text": "A concise summary of the spoken content in this clip.",
                    "reason": "Why this clip was selected (e.g., 'contains a key argument', 'demonstrates a feature clearly', 'captures an emotional peak', 'part of a complete narrative arc').",
                    "viral_potential_score": 7
                  }}
                ]
                ```
                - "start": float, The start time of the clip in seconds.
                - "end": float, The end time of the clip in seconds.
                - "text": string, A brief summary of the spoken content in this clip.
                - "reason": string, A clear explanation of why this clip was selected.
                - "viral_potential_score": integer (0-10), An assessment of the clip's potential to go viral, from 0 (low) to 10 (high).

                Ensure the output is ONLY a valid JSON array, exactly matching the structure and data types of the example provided. Do not include any additional text or explanations outside the JSON.
                """

                response = llm_interaction.llm_pass(
                    json_processing_model,
                    [
                        {"role": "system", "content": "You are an expert video content analyst and a precise JSON formatter. You will output only valid JSON."},
                        {"role": "user", "content": pass1_prompt.strip()}
                    ]
                )
                
                clips = llm_interaction.extract_json_from_text(response)

                if not isinstance(clips, list):
                    raise ValueError("LLM did not return a valid list of clips in JSON format.")
                
                # Filter clips based on duration
                filtered_clips = []
                for clip in clips:
                    duration = clip.get('end', 0) - clip.get('start', 0)
                    if clip_duration_min <= duration <= clip_duration_max:
                        filtered_clips.append(clip)
                    else:
                        if duration < clip_duration_min:
                            self.log_warning(f"✂️ Too short! {clip.get('start', 0):.1f}s - {clip.get('end', 0):.1f}s ({duration:.1f}s)")
                        else:
                            self.log_warning(f"📏 Too long! {clip.get('start', 0):.1f}s - {clip.get('end', 0):.1f}s ({duration:.1f}s)")

                if len(filtered_clips) < min_clips:
                    raise ValueError(f"After filtering, only {len(filtered_clips)} clips remain, which is less than the minimum required {min_clips}.")
                
                clips = filtered_clips

                context.update({
                    "clips": clips,
                    "current_stage": "llm_selection_complete"
                })
                
                self.log_info(f"Selected {len(clips)} clips:")
                for i, clip in enumerate(clips, 1):
                    duration = clip['end'] - clip['start']
                    self.log_info(f"  Clip {i}: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s) - {clip['text'][:70]}...")
                set_stage_status('llm_selection_complete', 'complete', {'num_clips': len(clips)})
                llm_interaction.cleanup() # Clear VRAM after successful completion
                return context

            except Exception as e:
                self.log_error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    self.log_error(f"All {max_retries} attempts failed. Failed to select clips with LLM.")
                    set_stage_status('llm_selection', 'failed', {'reason': str(e)})
                    return context
