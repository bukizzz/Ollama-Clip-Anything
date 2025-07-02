from agents.base_agent import BaseAgent
from typing import Dict, Any
from llm import llm_interaction
import json
from core.state_manager import set_stage_status, get_stage_status

class LLMSelectionAgent(BaseAgent):
    """Agent responsible for selecting engaging clips using an LLM."""

    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        transcription = context.get("transcription")
        storyboard_data = context.get("storyboard_data")
        audio_rhythm_data = context.get('audio_rhythm_data')
        engagement_analysis_results = context.get('engagement_analysis_results')
        layout_detection_results = context.get('layout_detection_results')
        speaker_tracking_results = context.get('speaker_tracking_results')
        qwen_vision_analysis_results = context.get('qwen_vision_analysis_results')
        content_alignment_data = context.get('content_alignment_data')
        identified_hooks = context.get('identified_hooks')
        llm_cut_decisions = context.get('llm_cut_decisions')

        print("\n🧠 \u001b[94mSelecting coherent clips using LLM...\u001b[0m")
        set_stage_status('llm_selection', 'running')

        if transcription is None:
            self.log_error("Transcription data is missing from context. Cannot select clips.")
            set_stage_status('llm_selection', 'failed', {'reason': 'Missing transcription'})
            return False

        user_prompt = context.get("user_prompt")
        b_roll_data = context.get("b_roll_data")

        # Prepare a comprehensive prompt for the LLM to select clips
        llm_prompt = f"""
        You are an expert video editor. Based on the following comprehensive analysis of a video,
        select the most engaging and coherent clips. Prioritize clips with high engagement scores,
        strong hook potential, and those recommended by the LLM Video Director.

        Consider the following data:

        Transcription:
        {json.dumps(transcription, indent=2)}

        Storyboard Data:
        {json.dumps(storyboard_data, indent=2)}

        Audio Rhythm Data:
        {json.dumps(audio_rhythm_data, indent=2)}

        Engagement Analysis Results:
        {json.dumps(engagement_analysis_results, indent=2)}

        Layout Detection Results:
        {json.dumps(layout_detection_results, indent=2)}

        Speaker Tracking Results:
        {json.dumps(speaker_tracking_results, indent=2)}

        Qwen-VL Vision Analysis Results:
        {json.dumps(qwen_vision_analysis_results, indent=2)}

        Content Alignment Data:
        {json.dumps(content_alignment_data, indent=2)}

        Identified Hooks:
        {json.dumps(identified_hooks, indent=2)}

        LLM Video Director Cut Decisions:
        {json.dumps(llm_cut_decisions, indent=2)}

        User Prompt (if any): {user_prompt}

        B-roll Data (if any): {b_roll_data}

        Select a list of clip segments. Each segment should be a JSON object with:
        - "start": The start time of the clip in seconds.
        - "end": The end time of the clip in seconds.
        - "text": A brief summary of the spoken content in the clip.
        - "reason": Why this clip was selected (e.g., "high engagement", "director recommendation", "key topic").
        - "viral_potential_score": The viral potential score for this clip (0-10).

        Ensure the selected clips are coherent and flow well together.
        """

        try:
            response = llm_interaction.llm_pass(
                llm_interaction.LLM_MODEL,
                [
                    {"role": "system", "content": "You are an expert video clip selection AI."},
                    {"role": "user", "content": llm_prompt.strip()}
                ]
            )
            
            clips = llm_interaction.extract_json_from_text(response)
            if not isinstance(clips, list):
                raise ValueError("LLM did not return a list of clips.")

            context.update({
                "clips": clips,
                "current_stage": "llm_selection_complete" # Update stage after successful clip selection
            })
            
            self.log_info(f"Selected {len(clips)} clips:")
            for i, clip in enumerate(clips, 1):
                duration = clip['end'] - clip['start']
                self.log_info(f"  Clip {i}: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s) - {clip['text'][:70]}...")
            set_stage_status('llm_selection_complete', 'complete', {'num_clips': len(clips)})
            return True

        except Exception as e:
            self.log_error(f"Failed to select clips with LLM: {e}")
            set_stage_status('llm_selection', 'failed', {'reason': str(e)})
            return False

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # This method is kept for compatibility with AgentManager, but the core logic is in run()
        return self.run(context)
