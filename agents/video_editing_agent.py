from agents.base_agent import Agent
from typing import Dict, Any
from video import video_editing


class VideoEditingAgent(Agent):
    """Agent responsible for creating enhanced video clips."""

    def __init__(self, config, state_manager):
        super().__init__("VideoEditingAgent")
        self.config = config
        self.state_manager = state_manager
        # Added a comment to force file update
        

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_video_path = context.get("processed_video_path")
        clips = context.get("clips")
        transcription = context.get("transcription")
        
        # current_stage = context.get("current_stage")
        processing_settings = context.get("processing_settings")
        video_info = context.get("video_info") # Get from context
        video_analysis = context.get("video_analysis") # Get from context

        # Ensure critical inputs are available
        if processed_video_path is None:
            raise RuntimeError("Processed video path is missing from context. Cannot perform video editing.")
        if clips is None:
            raise RuntimeError("Clips data is missing from context. Cannot perform video editing.")
        if transcription is None:
            raise RuntimeError("Transcription data is missing from context. Cannot perform video editing.")
        if video_info is None:
            raise RuntimeError("Video info is missing from context. Cannot perform video editing.")
        if processing_settings is None:
            raise RuntimeError("Processing settings are missing from context. Cannot perform video editing.")
        if video_analysis is None:
            raise RuntimeError("Video analysis is missing from context. Cannot perform video editing.")


        # Initialize processing_report and created_clips to ensure they are always in context
        created_clips = context.get("created_clips", [])
        processing_report = context.get("processing_report", {
            'results': {'success_rate': 0.0},
            'performance_metrics': {'total_processing_time': 0.0},
            'failed_clip_numbers': []
        })

        print(f"\n✂️ \u001b[94m5. Creating {len(clips)} enhanced video clips...\u001b[0m")
        created_clips, processing_report = video_editing.batch_process_with_analysis(
            processed_video_path, 
            clips, 
            transcription, 
            video_info=video_info, # Pass video_info
            processing_settings=processing_settings, # Pass processing_settings
            video_analysis=video_analysis, # Pass video_analysis
            audio_rhythm_data=context.get("audio_rhythm_data", {}), # Provide empty dict as default
            llm_cut_decisions=context.get("llm_cut_decisions", []), # Provide empty list as default
            speaker_tracking_results=context.get("speaker_tracking_results", {}) # Provide empty dict as default
        )
        context.update({
            "created_clips": created_clips,
            "processing_report": processing_report,
            "current_stage": "video_editing_complete"
        })

        return context
