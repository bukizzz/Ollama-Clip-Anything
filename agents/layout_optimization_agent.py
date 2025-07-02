import cv2
from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status
from core.config import LAYOUT_DETECTION_CONFIG # Reusing for thresholds

class LayoutOptimizationAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.layout_config = LAYOUT_DETECTION_CONFIG

    def run(self, context):
        layout_detection_results = context.get('layout_detection_results')
        speaker_tracking_results = context.get('speaker_tracking_results')
        clips = context.get('clips')

        if not layout_detection_results or not clips:
            self.log_error("Layout detection results or clips not found. Cannot perform layout optimization.")
            set_stage_status('layout_optimization', 'failed', {'reason': 'Missing layout data or clips'})
            return False

        self.log_info("Starting layout optimization agent...")
        set_stage_status('layout_optimization', 'running')

        try:
            optimized_layouts = []
            for clip in clips:
                clip_start = clip['start']
                clip_end = clip['end']

                # Filter layout detection results for the current clip
                clip_layouts = [l for l in layout_detection_results 
                                if clip_start <= l['timestamp'] <= clip_end]

                # Determine optimal layout based on detected layout types and speaker info
                # This is a simplified logic. A real system would use more advanced rules
                # and potentially machine learning to determine the best layout.
                optimal_layout = "single_person_focus" # Default
                num_multi_person_frames = sum(1 for l in clip_layouts if l['layout_type'] == 'multi_person')
                num_presentation_frames = sum(1 for l in clip_layouts if l['layout_type'] == 'presentation_mode')

                if num_multi_person_frames > len(clip_layouts) * 0.5: # More than half frames are multi-person
                    optimal_layout = "multi_person_grid"
                elif num_presentation_frames > len(clip_layouts) * 0.5:
                    optimal_layout = "presentation_with_speaker"
                
                # Add dynamic speaker focus (conceptual)
                # If speaker tracking is available, identify the active speaker and suggest focusing on them.
                active_speaker_id = None
                if speaker_tracking_results:
                    # Find the most frequent speaker in this clip segment
                    speaker_counts = {}
                    for speaker_data in speaker_tracking_results:
                        if clip_start <= speaker_data['start'] <= clip_end:
                            speaker_counts[speaker_data['speaker_id']] = speaker_counts.get(speaker_data['speaker_id'], 0) + 1
                    if speaker_counts:
                        active_speaker_id = max(speaker_counts, key=speaker_counts.get)

                optimized_layouts.append({
                    'clip_start': clip_start,
                    'clip_end': clip_end,
                    'optimal_layout': optimal_layout,
                    'active_speaker_id': active_speaker_id,
                    'transitions': "smooth_morph" # Placeholder for transition type
                })
            
            context['optimized_layouts'] = optimized_layouts
            self.log_info(f"Layout optimization complete. Generated {len(optimized_layouts)} optimized layouts.")
            set_stage_status('layout_optimization_complete', 'complete', {'num_layouts': len(optimized_layouts)})
            return True

        except Exception as e:
            self.log_error(f"Error during layout optimization: {e}")
            set_stage_status('layout_optimization', 'failed', {'reason': str(e)})
            return False
