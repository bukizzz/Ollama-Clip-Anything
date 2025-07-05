from agents.base_agent import Agent
from core.state_manager import set_stage_status

class LayoutOptimizationAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("LayoutOptimizationAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        layout_detection = context.get('layout_detection_results', [])
        speaker_tracking = context.get('speaker_tracking_results', {})
        clips = context.get('clips', [])

        if not layout_detection or not clips:
            self.log_error("Layout detection or clips data missing. Cannot optimize layout.")
            set_stage_status('layout_optimization', 'failed', {'reason': 'Missing dependencies'})
            return context

        print("📐 Optimizing layout for selected clips...")
        set_stage_status('layout_optimization', 'running')

        try:
            layout_recommendations = []
            for clip in clips:
                start, end = clip['start'], clip['end']
                
                # Analyze layout for this clip's duration
                clip_layouts = [item for item in layout_detection if start <= item['timestamp'] <= end]
                
                num_faces = [item['num_faces'] for item in clip_layouts]
                avg_faces = sum(num_faces) / len(num_faces) if num_faces else 0
                
                is_presentation = any(item['is_screen_share'] for item in clip_layouts)
                
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
