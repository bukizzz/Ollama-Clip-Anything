from agents.base_agent import Agent
from core.state_manager import set_stage_status

class LayoutOptimizationAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("LayoutOptimizationAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        stage_name = self.name
        print(f"\nExecuting stage: {stage_name}")

        # --- Pre-flight Check ---
        # If layout optimization recommendations already exist in the context, skip this stage.
        if context.get('layout_optimization_recommendations') and context.get('pipeline_stages', {}).get(stage_name) == 'complete':
            print(f"✅ Skipping {stage_name}: Layout optimization already complete.")
            return context

        layout_detection = context.get('current_analysis', {}).get('layout_detection_results', [])
        speaker_tracking = context.get('current_analysis', {}).get('speaker_tracking_results', {})
        clips = context.get('current_analysis', {}).get('clips', [])

        if not layout_detection or not clips:
            self.log_error("Layout detection or clips data missing. Cannot optimize layout.")
            set_stage_status('layout_optimization', 'failed', {'reason': 'Missing dependencies'})
            return context

        print("📐 Optimizing layout for selected clips...")
        set_stage_status('layout_optimization', 'running')

        try:
            layout_recommendations = []
            for clip in clips:
                scenes = clip.get('scenes')
                if not scenes:
                    self.log_warning(f"Clip '{clip.get('clip_description', 'N/A')}' has no scenes. Skipping.")
                    continue

                # A clip's start is the start of its first scene, and its end is the end of its last scene.
                start = scenes[0].get('start_time')
                end = scenes[-1].get('end_time')

                if start is None or end is None:
                    self.log_warning(f"Clip '{clip.get('clip_description', 'N/A')}' has scenes with missing start/end times. Skipping.")
                    continue
                
                # Analyze layout for this clip's duration
                clip_layouts = [
                    item for item in layout_detection 
                    if item['start_time'] < end and item['end_time'] > start
                ]
                
                num_faces = [item['num_faces'] for item in clip_layouts]
                avg_faces = sum(num_faces) / len(num_faces) if num_faces else 0
                
                # Screen share detection is not available in layout_detection, so this is removed.
                is_presentation = False 
                
                # Determine optimal layout
                if is_presentation:
                    layout = "presentation_with_speaker"
                elif avg_faces > 1.5:
                    layout = "multi_person_grid"
                else:
                    layout = "single_person_focus"
                
                # Dynamic speaker focus
                active_speaker = None
                if 'speaker_transitions' in speaker_tracking:
                    for t in speaker_tracking['speaker_transitions']:
                        if start <= t['timestamp'] <= end:
                            active_speaker = t['to']
                            break
                
                layout_recommendations.append({
                    'clip_start': start,
                    'clip_end': end,
                    'recommended_layout': layout,
                    'active_speaker': active_speaker
                })

            context['layout_optimization_recommendations'] = layout_recommendations
            print("✅ Layout optimization complete.")
            set_stage_status('layout_optimization_complete', 'complete', {'num_recommendations': len(layout_recommendations)})
            return context

        except Exception as e:
            self.log_error(f"Error during layout optimization: {e}")
            set_stage_status('layout_optimization', 'failed', {'reason': str(e)})
            return context
