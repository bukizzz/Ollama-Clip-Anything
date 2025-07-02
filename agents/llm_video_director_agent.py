from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status
from llm import llm_interaction
import json

class LLMVideoDirectorAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)

    def run(self, context):
        # Gather all relevant analysis results from the context
        transcription = context.get('transcription')
        storyboard_data = context.get('storyboard_data')
        audio_rhythm_data = context.get('audio_rhythm_data')
        engagement_analysis_results = context.get('engagement_analysis_results')
        layout_detection_results = context.get('layout_detection_results')
        speaker_tracking_results = context.get('speaker_tracking_results')
        qwen_vision_analysis_results = context.get('qwen_vision_analysis_results')
        content_alignment_data = context.get('content_alignment_data')
        identified_hooks = context.get('identified_hooks')

        if not transcription or not qwen_vision_analysis_results or not content_alignment_data:
            self.log_error("Missing essential analysis results for LLM Video Director. Skipping.")
            set_stage_status('llm_video_director', 'failed', {'reason': 'Missing essential data'})
            return False

        self.log_info("Starting LLM Video Director orchestration...")
        set_stage_status('llm_video_director', 'running')

        try:
            # Prepare a comprehensive prompt for the LLM
            llm_prompt = f"""
            You are an expert video director. Your task is to analyze various data streams from a video
            and make intelligent cut decisions to create a coherent and engaging short clip.

            Here is the available analysis data:

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

            Based on this data, provide a detailed plan for video cuts. Focus on:
            1. Identifying optimal start and end points for clips, considering narrative coherence, engagement, and visual flow.
            2. Highlighting significant events and object interactions.
            3. Ensuring smooth transitions and emphasizing key moments.
            4. Suggesting cut decisions that maximize viral potential and viewer retention.

            Output your recommendations as a JSON array of objects. Each object should represent a suggested clip segment
            and include:
            - "start_time": The start time of the clip segment in seconds.
            - "end_time": The end time of the clip segment in seconds.
            - "reason": A brief explanation for the cut decision (e.g., "high engagement moment", "speaker transition", "scene change").
            - "key_elements": A list of key visual or audio elements present in this segment.
            - "viral_potential_score": An estimated score for viral potential (0-10).
            """

            self.log_info("Sending comprehensive data to LLM for video direction...")
            response = llm_interaction.llm_pass(
                llm_interaction.LLM_MODEL,
                [
                    {"role": "system", "content": "You are an expert video director, making intelligent cut decisions."},
                    {"role": "user", "content": llm_prompt.strip()}
                ]
            )
            
            llm_cut_decisions = llm_interaction.extract_json_from_text(response)
            if not isinstance(llm_cut_decisions, list):
                raise ValueError("LLM did not return a list of cut decisions.")

            context['llm_cut_decisions'] = llm_cut_decisions
            self.log_info(f"LLM Video Director orchestration complete. Generated {len(llm_cut_decisions)} cut decisions.")
            set_stage_status('llm_video_director_complete', 'complete', {'num_cut_decisions': len(llm_cut_decisions)})
            return True

        except Exception as e:
            self.log_error(f"Error during LLM Video Director orchestration: {e}")
            set_stage_status('llm_video_director', 'failed', {'reason': str(e)})
            return False
