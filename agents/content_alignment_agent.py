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

    def _filter_data_by_cuts(self, data: Any, cut_decisions: List[Dict[str, Any]], time_key: str = 'timestamp') -> List[Dict[str, Any]]:
        """Filters a list of data points to include only those within the specified cut decisions."""
        if not isinstance(data, list):
            return [] # Return empty list if data is not a list (e.g., "N/A")

        filtered_data = []
        for item in data:
            item_time = item.get(time_key)
            if item_time is None:
                continue
            for cut in cut_decisions:
                if cut['start_time'] <= item_time <= cut['end_time']:
                    filtered_data.append(item)
                    break # Found a match, move to next item
        return filtered_data

    def _filter_transcript_by_cuts(self, transcription: Any, cut_decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters transcription segments to include only those within the specified cut decisions."""
        if not isinstance(transcription, list):
            return [] # Return empty list if transcription is not a list

        filtered_transcript = []
        for seg in transcription:
            seg_start = seg.get('start')
            seg_end = seg.get('end')
            if seg_start is None or seg_end is None:
                continue
            for cut in cut_decisions:
                # Check for overlap between transcript segment and cut decision
                if max(seg_start, cut['start_time']) < min(seg_end, cut['end_time']):
                    filtered_transcript.append(seg)
                    break
        return filtered_transcript

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        transcription = context.get("transcription")
        storyboard_data = context.get("storyboard_data")
        audio_rhythm_data = context.get('audio_rhythm_data')
        engagement_analysis_results = context.get('engagement_analysis_results')
        layout_detection_results = context.get('layout_detection_results')
        speaker_tracking_results = context.get('speaker_tracking_results')
        qwen_vision_analysis_results = context.get('qwen_vision_analysis_results')
        llm_cut_decisions = context.get('llm_cut_decisions')

        if not transcription or not storyboard_data or not llm_cut_decisions:
            self.log_error("Skipping content alignment: transcription, storyboard_data, or llm_cut_decisions not available.")
            set_stage_status('content_alignment', 'skipped', {'reason': 'Missing transcription, storyboard, or cut decisions'})
            context["content_alignment_data"] = "N/A"
            return context

        print("🔄 Starting content alignment...")
        set_stage_status('content_alignment', 'running')

        # Filter all relevant data based on LLM cut decisions
        filtered_transcript = self._filter_transcript_by_cuts(transcription, llm_cut_decisions)
        filtered_storyboard = self._filter_data_by_cuts(storyboard_data, llm_cut_decisions, time_key='timestamp')
        
        # Ensure data is a list before passing to filter, or keep as "N/A" if it's already that string
        filtered_audio_rhythm = self._filter_data_by_cuts(audio_rhythm_data, llm_cut_decisions, time_key='timestamp') if isinstance(audio_rhythm_data, list) else audio_rhythm_data
        filtered_engagement = self._filter_data_by_cuts(engagement_analysis_results, llm_cut_decisions, time_key='timestamp') if isinstance(engagement_analysis_results, list) else engagement_analysis_results
        filtered_layout = self._filter_data_by_cuts(layout_detection_results, llm_cut_decisions, time_key='timestamp') if isinstance(layout_detection_results, list) else layout_detection_results
        filtered_speaker_tracking = self._filter_data_by_cuts(speaker_tracking_results, llm_cut_decisions, time_key='timestamp') if isinstance(speaker_tracking_results, list) else speaker_tracking_results
        filtered_qwen_vision = self._filter_data_by_cuts(qwen_vision_analysis_results, llm_cut_decisions, time_key='timestamp') if isinstance(qwen_vision_analysis_results, list) else qwen_vision_analysis_results

        # Prepare data for LLM
        simplified_transcript = [{
            "start": round(seg['start'], 1),
            "end": round(seg['end'], 1),
            "text": seg['text']
        } for seg in filtered_transcript]

        simplified_storyboard = [{
            "timestamp": round(sb['timestamp'], 1),
            "description": sb['description'],
            "content_type": sb.get('content_type'),
            "hook_potential": sb.get('hook_potential')
        } for sb in filtered_storyboard]

        llm_prompt = f"""
        Given the following video transcript, storyboard data, audio rhythm data, engagement analysis results,
        layout detection results, speaker tracking results, and Qwen-VL vision analysis results,
        all filtered to specific relevant clip segments,
        identify key moments where the spoken content aligns with visual scene changes or significant visual elements.
        Map speakers to face tracking IDs, sync audio rhythm data with visual engagement metrics,
        and correlate scene changes with content topic shifts.

        Transcript:
        {json.dumps(simplified_transcript, indent=2)}

        Storyboard:
        {json.dumps(simplified_storyboard, indent=2)}

        Audio Rhythm Data:
        {json.dumps(filtered_audio_rhythm, indent=2)}

        Engagement Analysis Results:
        {json.dumps(filtered_engagement, indent=2)}

        Layout Detection Results:
        {json.dumps(filtered_layout, indent=2)}

        Speaker Tracking Results:
        {json.dumps(filtered_speaker_tracking, indent=2)}

        Qwen-VL Vision Analysis Results:
        {json.dumps(filtered_qwen_vision, indent=2)}

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
