
from agents.base_agent import Agent
from typing import Dict, Any
from analysis import analysis_and_reporting

class VideoAnalysisAgent(Agent):
    """Agent responsible for performing enhanced video analysis and optimizing processing settings."""

    def __init__(self):
        super().__init__("VideoAnalysisAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_video_path = context.get("processed_video_path")
        current_stage = context.get("current_stage")
        video_analysis = context.get("video_analysis")

        print("\n4. Performing enhanced video analysis...")
        if current_stage == "llm_selection_complete":
            # This stage doesn't produce critical intermediate files for resumption, so no state update needed here.
            video_analysis = analysis_and_reporting.analyze_video_content(processed_video_path)
            processing_settings = analysis_and_reporting.optimize_processing_settings(video_analysis)
            context.update({
                "video_analysis": video_analysis,
                "processing_settings": processing_settings,
                "current_stage": "video_analysis_complete"
            })
        else:
            print("⏩ Skipping video analysis. Loaded from state.")
        
        return context

