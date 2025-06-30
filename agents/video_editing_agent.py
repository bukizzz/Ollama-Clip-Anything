from agents.base_agent import Agent
from typing import Dict, Any
from video import video_editing


class VideoEditingAgent(Agent):
    """Agent responsible for creating enhanced video clips."""

    def __init__(self):
        super().__init__("VideoEditingAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_video_path = context.get("processed_video_path")
        clips = context.get("clips")
        transcription = context.get("transcription")
        
        current_stage = context.get("current_stage")
        processing_settings = context.get("processing_settings")

        print(f"\n5. Creating {len(clips)} enhanced video clips...")
        if current_stage == "video_analysis_complete":
            created_clips, processing_report = video_editing.batch_process_with_analysis(
                processed_video_path, clips, transcription, custom_settings=processing_settings
            )
            context.update({
                "created_clips": created_clips,
                "processing_report": processing_report,
                "current_stage": "video_editing_complete"
            })
        else:
            print("⏩ Skipping video editing. Loaded from state.")

        return context