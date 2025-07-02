from agents.base_agent import BaseAgent
from typing import Dict, Any
from llm import llm_interaction
import json
from core.state_manager import set_stage_status, get_stage_status

class ContentAlignmentAgent(BaseAgent):
    """Agent responsible for synchronizing audio and video elements."""

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

        if not transcription or not storyboard_data:
            self.log_error("Skipping content alignment: transcription or storyboard_data not available.")
            set_stage_status('content_alignment', 'skipped', {'reason': 'Missing transcription or storyboard'})
            context["content_alignment_data"] = "N/A"
            return context

        self.log_info("Starting content alignment...")
        set_stage_status('content_alignment', 'running')

        # Prepare data for LLM
        simplified_transcript = [{
            "start": round(seg['start'], 1),
            "end": round(seg['end'], 1),
            "text": seg['text']
        } for seg in transcription]

        simplified_storyboard = [{
            "timestamp": round(sb['timestamp'], 1),
            "description": sb['description'],
            "content_type": sb.get('content_type'),
            "hook_potential": sb.get('hook_potential')
        } for sb in storyboard_data]

        # Prepare new data for LLM
        simplified_audio_rhythm = audio_rhythm_data if audio_rhythm_data else "N/A"
        simplified_engagement = engagement_analysis_results if engagement_analysis_results else "N/A"
        simplified_layout = layout_detection_results if layout_detection_results else "N/A"
        simplified_speaker_tracking = speaker_tracking_results if speaker_tracking_results else "N/A"
        simplified_qwen_vision = qwen_vision_analysis_results if qwen_vision_analysis_results else "N/A"

        llm_prompt = f"""
        Given the following video transcript, storyboard data, audio rhythm data, engagement analysis results,
        layout detection results, speaker tracking results, and Qwen-VL vision analysis results,
        identify key moments where the spoken content aligns with visual scene changes or significant visual elements.
        Map speakers to face tracking IDs, sync audio rhythm data with visual engagement metrics,
        and correlate scene changes with content topic shifts.

        Transcript:
        {json.dumps(simplified_transcript, indent=2)}

        Storyboard:
        {json.dumps(simplified_storyboard, indent=2)}

        Audio Rhythm Data:
        {json.dumps(simplified_audio_rhythm, indent=2)}

        Engagement Analysis Results:
        {json.dumps(simplified_engagement, indent=2)}

        Layout Detection Results:
        {json.dumps(simplified_layout, indent=2)}

        Speaker Tracking Results:
        {json.dumps(simplified_speaker_tracking, indent=2)}

        Qwen-VL Vision Analysis Results:
        {json.dumps(simplified_qwen_vision, indent=2)}

        Provide a comprehensive content-visual alignment map as a JSON array of objects.
        Each object should include:
        - "timestamp": The time in seconds of the aligned moment.
        - "visual_description": A brief description of the visual content at that moment.
        - "spoken_text": The corresponding spoken text.
        - "speaker_id": The identified speaker (if available).
        - "engagement_score": The engagement score at that moment.
        - "layout_type": The detected layout type.
        - "qwen_features": Relevant Qwen-VL features (e.g., objects detected, scene description).
        - "audio_rhythm_info": Relevant audio rhythm information (e.g., tempo, beat).
        - "content_topic_shift": Indicate if a content topic shift is detected.
        """

        try:
            response = llm_interaction.llm_pass(llm_interaction.LLM_MODEL, [
                {"role": "system", "content": "You are an expert in video content analysis and synchronization."},
                {"role": "user", "content": llm_prompt.strip()}
            ])
            
            alignment_results = llm_interaction.extract_json_from_text(response)
            context["content_alignment_data"] = alignment_results
            self.log_info("Content alignment by LLM complete.")
            set_stage_status('content_alignment_complete', 'complete', {'num_alignments': len(alignment_results)})
        except Exception as e:
            self.log_error(f"Failed to perform content alignment with LLM: {e}")
            set_stage_status('content_alignment', 'failed', {'reason': str(e)})
            context["content_alignment_data"] = "Error during alignment."

        return context

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # This method is kept for compatibility with AgentManager, but the core logic is in run()
        return self.run(context)