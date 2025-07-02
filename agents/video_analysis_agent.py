from agents.base_agent import Agent
from typing import Dict, Any
from analysis import analysis_and_reporting
from video.tracking_manager import TrackingManager

class VideoAnalysisAgent(Agent):
    """Agent responsible for performing enhanced video analysis and optimizing processing settings."""

    def __init__(self):
        super().__init__("VideoAnalysisAgent")
        tracking_manager = TrackingManager()
        self.face_tracker = tracking_manager.get_face_tracker()
        self.object_tracker = tracking_manager.get_object_tracker()

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        input_source = context.get("input_source") # Use input_source
        current_stage = context.get("current_stage")
        video_analysis = context.get("video_analysis")

        print("\n📊 \u001b[94m4. Performing enhanced video analysis...\u001b[0m")
        
        # Only perform analysis if it hasn't been done or if a fresh run is needed
        if video_analysis is None or current_stage == "llm_selection_complete":
            print("Performing video analysis...")
            # Ensure input_source is a string
            if input_source is None:
                raise ValueError("Input source cannot be None for video analysis.")
            assert isinstance(input_source, str), "Input source must be a string path."
            
            video_analysis = analysis_and_reporting.analyze_video_content(
                input_source # input_source is now guaranteed to be str
            )
            processing_settings = analysis_and_reporting.optimize_processing_settings(video_analysis)
            context.update({
                "video_analysis": video_analysis,
                "processing_settings": processing_settings,
                "current_stage": "video_analysis_complete"
            })
        else:
            print("⏩ Skipping video analysis. Loaded from state.")
        
        return context
