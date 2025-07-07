from agents.base_agent import Agent
from typing import Dict, Any, List, Optional
from llm import llm_interaction
from pydantic import BaseModel, Field
from core.state_manager import set_stage_status
import json

class ContentAlignment(BaseModel):
    start_time: float = Field(description="The start time in seconds of the aligned moment.")
    end_time: float = Field(description="The end time in seconds of the aligned moment.")
    speaker: str = Field(description="The ID of the speaker.")
    transcript_segment: str = Field(description="The spoken content during this segment.")
    visual_description: str = Field(description="A brief description of the visual content at that moment.")
    scene_change: bool = Field(description="True if a scene change was detected, false otherwise.")
    topic_shift: bool = Field(description="True if a topic shift was detected, false otherwise.")
    engagement_impact: str = Field(description="Engagement metric for the segment.")
    rhythm_correlation: Optional[str] = Field("N/A", description="Audio rhythm analysis for the segment.")

class ContentAlignments(BaseModel):
    alignments: List[ContentAlignment]

class ContentAlignmentAgent(Agent):
    """Agent responsible for synchronizing audio and video elements."""

    def __init__(self, config, state_manager):
        super().__init__("ContentAlignmentAgent")
        self.config = config
        self.state_manager = state_manager

    # Removed _filter_data_by_cuts and _filter_transcript_by_cuts as they are no longer needed
    # since we are not filtering by llm_cut_decisions in this agent.

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        transcription = context.get("transcription")
        storyboard_data = context.get("storyboard_data")
        audio_rhythm_data = context.get('audio_rhythm_data')
        engagement_analysis_results = context.get('engagement_analysis_results')
        layout_detection_results = context.get('layout_detection_results')
        speaker_tracking_results = context.get('speaker_tracking_results')
        qwen_vision_analysis_results = context.get('qwen_vision_analysis_results')
        # llm_cut_decisions = context.get('llm_cut_decisions') # Removed dependency

        if not transcription or not storyboard_data: # Removed llm_cut_decisions from check
            self.log_error("Skipping content alignment: transcription or storyboard_data not available.")
            set_stage_status('content_alignment', 'skipped', {'reason': 'Missing transcription or storyboard data'})
            context["content_alignment_data"] = "N/A"
            return context

        print("🔄 Starting content alignment...")
        set_stage_status('content_alignment', 'running')

        # No longer filtering by llm_cut_decisions. Pass full data.
        # Ensure data is a list before passing to LLM, or keep as "N/A" if it's already that string
        # Also, ensure data is not None before passing to json.dumps
        
        # Prepare data for LLM
        # Simplified transcript and storyboard are still useful for conciseness, but not filtered by cuts
        simplified_transcript = [{
            "start": round(seg['start'], 1),
            "end": round(seg['end'], 1),
            "text": seg['text']
        } for seg in transcription if transcription is not None] # Ensure transcription is not None

        simplified_storyboard = [{
            "timestamp": round(sb['timestamp'], 1),
            "description": sb['description'],
            "content_type": sb.get('content_type'),
            "hook_potential": sb.get('hook_potential')
        } for sb in storyboard_data if storyboard_data is not None] # Ensure storyboard_data is not None

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
        {json.dumps(audio_rhythm_data if audio_rhythm_data is not None else {}, indent=2)}

        Engagement Analysis Results:
        {json.dumps(engagement_analysis_results if engagement_analysis_results is not None else [], indent=2)}

        Layout Detection Results:
        {json.dumps(layout_detection_results if layout_detection_results is not None else [], indent=2)}

        Speaker Tracking Results:
        {json.dumps(speaker_tracking_results if speaker_tracking_results is not None else {}, indent=2)}

        Qwen-VL Vision Analysis Results:
        {json.dumps(qwen_vision_analysis_results if qwen_vision_analysis_results is not None else [], indent=2)}

        Provide a comprehensive content-visual alignment map as a JSON object with a single key "alignments" which is a list of alignment objects.
        """

        
        print("🧠 Performing content alignment with LLM...")
        try:
            alignment_results_obj = llm_interaction.robust_llm_json_extraction(
                system_prompt="You are an expert in video content analysis and synchronization. Your task is to create a comprehensive content-visual alignment map.",
                user_prompt=llm_prompt.strip(),
                output_schema=ContentAlignments
            )
            
            # Convert the Pydantic model instance to a dictionary for JSON serialization
            context["content_alignment_data"] = alignment_results_obj.model_dump()
            print("✅ Content alignment by LLM complete.")
            set_stage_status('content_alignment_complete', 'complete', {'num_alignments': len(alignment_results_obj.alignments)})
        except Exception as e:
            self.log_error(f"Failed to perform content alignment with LLM: {e}. Stopping pipeline.")
            set_stage_status('content_alignment', 'failed', {'reason': str(e)})
            context["content_alignment_data"] = "Error during alignment."

        return context
