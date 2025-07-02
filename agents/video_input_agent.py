
from agents.base_agent import Agent
from typing import Dict, Any
from video import video_input
from core import utils
import argparse

class VideoInputAgent(Agent):
    """Agent responsible for handling video input and initial video analysis."""

    def __init__(self):
        super().__init__("VideoInputAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("\n1. Getting input video...")
        
        # Retrieve args from context or create a dummy for initial run
        args_dict = context.get("args", {})
        args = argparse.Namespace(**args_dict)

        processed_video_path = context.get("processed_video_path")
        video_info = context.get("video_info")
        current_stage = context.get("current_stage")

        if current_stage == "start":
            if args.video_path or args.youtube_url:
                input_video = video_input.get_video_input(video_path=args.video_path, youtube_url=args.youtube_url, youtube_quality=args.youtube_quality)
            else:
                input_video = video_input.choose_input_video()
            
            print("\n🔍 [94mAnalyzing input video...[0m")
            video_info, processed_video_path = utils.get_video_info(input_video)
            
            context.update({
                "input_source": input_video,
                "processed_video_path": processed_video_path,
                "video_info": video_info,
                "current_stage": "video_input_complete",
                "temp_files": {"processed_video": processed_video_path}
            })
        else:
            print(f"⏩ Skipping video input. Loaded from state: {processed_video_path}")
            
        print(f"📹 Video info: {video_info['width']}x{video_info['height']}, "
              f"{video_info['duration']:.1f}s, {video_info['fps']:.1f}fps, "
              f"codec: {video_info['codec']}")
        
        return context
