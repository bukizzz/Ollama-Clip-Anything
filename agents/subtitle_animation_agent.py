from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status
from core.config import SUBTITLE_ANIMATION_CONFIG
from audio.subtitle_generation import create_ass_file # Re-use existing subtitle generation

class SubtitleAnimationAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.subtitle_animation_config = SUBTITLE_ANIMATION_CONFIG

    def run(self, context):
        transcription = context.get('transcription')
        video_info = context.get('video_info')
        clips = context.get('clips')

        if not transcription or not video_info or not clips:
            self.log_error("Missing transcription, video info, or clips. Cannot generate animated subtitles.")
            set_stage_status('subtitle_animation', 'failed', {'reason': 'Missing dependencies'})
            return False

        self.log_info("Starting subtitle animation agent...")
        set_stage_status('subtitle_animation', 'running')

        try:
            animated_subtitle_paths = []
            for i, clip_data in enumerate(clips, 1):
                clip_start = clip_data['start']
                clip_end = clip_data['end']

                # Filter transcription segments relevant to the current clip
                clip_transcript = [
                    seg for seg in transcription 
                    if seg['start'] >= clip_start and seg['end'] <= clip_end
                ]

                if not clip_transcript:
                    self.log_warning(f"No transcript for clip {i}. Skipping subtitle generation for this clip.")
                    continue

                ass_path = self.temp_manager.get_temp_path(f"animated_subtitles_clip_{i}.ass")
                
                # Use the existing create_ass_file function, potentially enhancing it
                # to accept animation parameters from self.subtitle_animation_config
                # For now, we'll just pass the basic parameters.
                create_ass_file(
                    clip_transcript, 
                    ass_path, 
                    time_offset=int(clip_start), 
                    video_height=video_info['height'],
                    # Pass animation settings if create_ass_file is updated to handle them
                    # word_by_word_timing=self.subtitle_animation_config.get('word_by_word_timing_enabled', True),
                    # emphasis_effects=self.subtitle_animation_config.get('emphasis_effects_enabled', True),
                    # speaker_color_coding=self.subtitle_animation_config.get('speaker_color_coding_enabled', True)
                )
                animated_subtitle_paths.append(ass_path)

            context['animated_subtitle_paths'] = animated_subtitle_paths
            self.log_info(f"Subtitle animation complete. Generated {len(animated_subtitle_paths)} subtitle files.")
            set_stage_status('subtitle_animation_complete', 'complete', {'num_subtitle_files': len(animated_subtitle_paths)})
            return True

        except Exception as e:
            self.log_error(f"Error during subtitle animation: {e}")
            set_stage_status('subtitle_animation', 'failed', {'reason': str(e)})
            return False
