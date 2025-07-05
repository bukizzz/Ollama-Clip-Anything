from agents.base_agent import Agent
from core.state_manager import set_stage_status
from audio.subtitle_generation import create_ass_file
from core.temp_manager import get_temp_path

class SubtitleAnimationAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("SubtitleAnimationAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.subtitle_animation_config = agent_config.get('subtitle_animation', {})

    def execute(self, context):
        transcription = context.get('transcription', [])
        video_info = context.get('video_info')
        clips = context.get('clips', [])
        speaker_tracking = context.get('speaker_tracking_results', {})
        layout_recommendations = context.get('layout_optimization_recommendations', [])

        if not all([transcription, video_info, clips]):
            self.log_error("Missing data for subtitle animation.")
            set_stage_status('subtitle_animation', 'failed', {'reason': 'Missing dependencies'})
            return context

        print("🎨 Generating animated subtitles...")
        set_stage_status('subtitle_animation', 'running')

        try:
            speaker_colors = {s: f"FF{i*40:02X}{255-i*40:02X}" for i, s in enumerate(speaker_tracking.get('speaker_to_face_map', {}).keys())}

            animated_subtitle_paths = []
            for i, clip in enumerate(clips, 1):
                start, end = clip['start'], clip['end']
                
                clip_transcript = [seg for seg in transcription if start <= seg['start'] < end]
                if not clip_transcript:
                    continue

                layout_rec = next((item for item in layout_recommendations if item['clip_start'] == start), {'recommended_layout': 'default'})
                
                ass_path = get_temp_path(f"animated_subtitles_clip_{i}.ass")
                create_ass_file(
                    clip_transcript,
                    ass_path,
                    time_offset=int(start),
                    video_height=video_info['height'],
                    layout=layout_rec['recommended_layout'],
                    speaker_colors=speaker_colors if self.subtitle_animation_config.get('speaker_color_coding_enabled') else None
                )
                animated_subtitle_paths.append(ass_path)

            context['animated_subtitle_paths'] = animated_subtitle_paths
            print(f"✅ Generated {len(animated_subtitle_paths)} animated subtitle files.")
            set_stage_status('subtitle_animation_complete', 'complete', {'num_files': len(animated_subtitle_paths)})
            return context

        except Exception as e:
            self.log_error(f"Error in SubtitleAnimationAgent: {e}")
            set_stage_status('subtitle_animation', 'failed', {'reason': str(e)})
            return context
