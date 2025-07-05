from agents.base_agent import Agent
from typing import Dict, Any, List, Optional
from llm import llm_interaction
from pydantic import BaseModel, Field
from core.state_manager import set_stage_status

class ContentAlignment(BaseModel):
    start_time: float = Field(description="The start time in seconds of the aligned moment.")
    end_time: float = Field(description="The end time in seconds of the aligned moment.")
    spoken_content: str = Field(description="The spoken content during this segment.")
    speaker_id: str = Field(description="The ID of the speaker.")
    visual_scene_description: str = Field(description="A brief description of the visual content at that moment.")
    content_topic: str = Field(description="The main topic of the content in this segment.")
    visual_elements_present: str = Field(description="A description of visual elements present.")
    scene_change_detected: bool = Field(description="True if a scene change was detected, false otherwise.")
    engagement_metric: Optional[str] = Field("N/A", description="Engagement metric for the segment.")
    audio_rhythm_analysis: Optional[str] = Field("N/A", description="Audio rhythm analysis for the segment.")
    layout_details: Optional[str] = Field("N/A", description="Layout details for the segment.")

class ContentAlignments(BaseModel):
    alignments: List[ContentAlignment]

class ContentAlignmentAgent(Agent):
    """Agent responsible for synchronizing audio and video elements."""

    def __init__(self, config, state_manager):
        super().__init__("ContentAlignmentAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
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

        print("🔄 Starting content alignment...")
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
        {simplified_transcript}

        Storyboard:
        {simplified_storyboard}

        Audio Rhythm Data:
        {simplified_audio_rhythm}

        Engagement Analysis Results:
        {simplified_engagement}

        Layout Detection Results:
        {simplified_layout}

        Speaker Tracking Results:
        {simplified_speaker_tracking}

        Qwen-VL Vision Analysis Results:
        {simplified_qwen_vision}

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
