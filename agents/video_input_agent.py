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
        """
        Ensures a video is available for processing, either by using a path from the
        context, command-line arguments, or by prompting the user.
        """
        stage_name = self.name
        print(f"\nExecuting stage: {stage_name}")

        try:
            args_dict = context.get("args", {})
            youtube_url_arg = args_dict.get("youtube_url")
            youtube_quality_arg = args_dict.get("youtube_quality")
            video_path_arg = args_dict.get("video_path")

            input_video_path = None

            # If command-line arguments are provided, they take precedence
            if youtube_url_arg:
                input_video_path = video_input.get_video_input(
                    youtube_url=youtube_url_arg,
                    youtube_quality=youtube_quality_arg
                )
            elif video_path_arg:
                input_video_path = video_input.get_video_input(
                    video_path=video_path_arg
                )
            # If no command-line arguments, check for previously processed video
            elif context.get('processed_video_path') and context.get('pipeline_stages', {}).get(stage_name) == 'complete':
                print(f"✅ Skipping {stage_name}: Video already processed from previous session.")
                return context # Exit early if already processed and no new input
            elif context.get('processed_video_path'):
                input_video_path = context.get('processed_video_path')
            else:
                # If no source is provided via args or context, prompt the user
                input_video_path = video_input.choose_input_video()

            if not input_video_path or not os.path.exists(input_video_path):
                raise FileNotFoundError(f"Video file not found at: {input_video_path}")

            # Get video info and update context
            video_info, _ = utils.get_video_info(input_video_path)
            context['processed_video_path'] = input_video_path
            context.setdefault('metadata', {})['video_info'] = video_info
            context['pipeline_stages'][stage_name] = 'complete'
            
            print(f"✅ {stage_name} complete. Video ready for processing at: {input_video_path}")
            print(f"   Video info: {video_info['width']}x{video_info['height']}, {video_info['duration']:.1f}s, {video_info['fps']:.1f}fps, codec: {video_info['codec']}")

        except Exception as e:
            print(f"❌ Error in {stage_name}: {e}")
            context['pipeline_stages'][stage_name] = 'failed'
            context['error_log'] = str(e)
            raise  # Re-raise the exception to halt the pipeline

        return context
