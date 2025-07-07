import argparse
import os
from typing import Dict, Any, Tuple, Optional
from agents.base_agent import Agent
from video import video_input # Keep this for new video_input functions
from core import utils # Import utils for get_video_info

class VideoInputAgent(Agent):
    """Agent responsible for handling video input and initial video analysis."""

    def __init__(self, config, state_manager):
        super().__init__("VideoInputAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("\n1. Getting input video...")
        
        # Retrieve args from context or create a dummy for initial run
        args_dict = context.get("args", {})
        args = argparse.Namespace(**args_dict)

        processed_video_path = context.get('processed_video_path')
        video_info = context.get('metadata', {}).get('video_info') # Get from new hierarchical structure
        current_stage = context.get('pipeline_stages', {}).get('video_input_complete', {}).get('status')

        # Change condition to check for None or "failed" status
        if current_stage is None or current_stage == "failed":
            try:
                input_video_path = None
                arg_video_path = getattr(args, 'video_path', None)
                arg_youtube_url = getattr(args, 'youtube_url', None)
                arg_youtube_quality = getattr(args, 'youtube_quality', None)

                # Ensure empty strings are treated as None for video_path and youtube_url
                if arg_video_path == "":
                    arg_video_path = None
                if arg_youtube_url == "":
                    arg_youtube_url = None

                if arg_video_path or arg_youtube_url:
                    input_video_path = video_input.get_video_input(
                        video_path=arg_video_path,
                        youtube_url=arg_youtube_url,
                        youtube_quality=arg_youtube_quality
                    )
                else:
                    # If neither is provided as an argument, prompt the user
                    input_video_path = video_input.choose_input_video()

                processed_video_path = input_video_path # The returned path is already processed/downloaded
                
                # Get video info using utils.get_video_info after download/selection
                video_info, _ = utils.get_video_info(processed_video_path)

            except Exception as e:
                print(f"❌ Failed to get video input or validate: {e}")
                context["pipeline_stages"]["video_input_complete"] = {"status": "failed", "details": {"reason": str(e)}}
                return context
            
            context['metadata']['video_info'] = video_info
            context['processed_video_path'] = processed_video_path
            context['current_stage'] = "video_input_complete"
            context['temp_files'] = {"processed_video": processed_video_path}
        else:
            print(f"â Š Skipping video input. Loaded from state: {processed_video_path}")
            
        if video_info: # Check if video_info is not None before accessing its keys
            print(f"đŸ“š Video info: {video_info['width']}x{video_info['height']}, "
                  f"{video_info['duration']:.1f}s, {video_info['fps']:.1f}fps, "
                  f"codec: {video_info['codec']}")
        else:
            print("đŸ“š Video info: Not available or video processing failed.")
        
        return context
