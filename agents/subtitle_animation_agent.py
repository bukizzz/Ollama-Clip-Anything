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
        # Updated paths to retrieve data from the hierarchical context
        transcription = context.get('archived_data', {}).get('full_transcription', [])
        video_info = context.get('metadata', {}).get('video_info')
        clips = context.get('current_analysis', {}).get('clips', [])
        layout_speaker_analysis = context.get('current_analysis', {}).get('layout_speaker_analysis_results', {})
        speaker_tracking = layout_speaker_analysis.get('speaker_tracking', {})
        layout_recommendations = context.get('layout_optimization_recommendations', [])

        if not all([transcription, video_info, clips]):
            self.log_error("Missing data for subtitle animation. Required: transcription, video_info, clips.")
            set_stage_status('subtitle_animation', 'failed', {'reason': 'Missing dependencies'})
            return context

        print("🎨 Generating animated subtitles...")
        set_stage_status('subtitle_animation', 'running')

        try:
            speaker_to_face_map = speaker_tracking.get('speaker_to_face_map')
            # Ensure speaker_to_face_map is a dictionary, default to empty if not
            if not isinstance(speaker_to_face_map, dict):
                speaker_to_face_map = {} 

            speaker_colors = {}
            # Filter out None values from keys before enumeration
            valid_speaker_keys = [s for s in speaker_to_face_map.keys() if s is not None]
            for i, s in enumerate(valid_speaker_keys):
                speaker_colors[s] = f"FF{i*40:02X}{255-i*40:02X}"

            animated_subtitle_paths = []
            for i, clip in enumerate(clips, 1):
                # Clips from ContentDirectorAgent have 'start_time' and 'end_time' directly
                start = clip.get('start_time')
                end = clip.get('end_time')

                if start is None or end is None:
                    self.log_warning(f"Clip {clip.get('clip_description', 'N/A')} has missing start/end times. Skipping subtitle generation for this clip.")
                    continue
                
                clip_transcript = [seg for seg in transcription if start <= seg['start'] < end]
                if not clip_transcript:
                    self.log_warning(f"No transcript segments found for clip from {start:.2f}s to {end:.2f}s. Skipping subtitle generation for this clip.")
                    continue

                # Find the layout recommendation for the current clip
                layout_rec = next((item for item in layout_recommendations if item['clip_start'] == start and item['clip_end'] == end), {'recommended_layout': 'default'})
                
                ass_path = get_temp_path(f"animated_subtitles_clip_{i}.ass")

                # Ensure video_info is a dictionary and has 'height'
                video_height = None
                if isinstance(video_info, dict) and 'height' in video_info:
                    video_height = video_info['height']
                else:
                    self.log_error("video_info is missing or does not contain 'height'. Cannot generate subtitles.")
                    set_stage_status('subtitle_animation', 'failed', {'reason': 'Missing video_info height'})
                    return context

                create_ass_file(
                    clip_transcript,
                    ass_path,
                    time_offset=int(start),
                    video_height=video_height,
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
