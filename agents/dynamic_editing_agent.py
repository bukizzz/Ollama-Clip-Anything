from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status

class DynamicEditingAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)

    def run(self, context):
        clips = context.get('clips')
        audio_rhythm_data = context.get('audio_rhythm_data')
        engagement_analysis_results = context.get('engagement_analysis_results')
        llm_cut_decisions = context.get('llm_cut_decisions')

        if not clips:
            self.log_error("No clips found for dynamic editing.")
            set_stage_status('dynamic_editing', 'failed', {'reason': 'No clips to edit'})
            return False

        self.log_info("Starting dynamic editing agent...")
        set_stage_status('dynamic_editing', 'running')

        try:
            # This agent primarily influences how clips are processed in video_editing.py
            # It generates editing decisions that will be consumed by the video editing module.
            
            # For now, we'll just log the decisions that would be made.
            editing_decisions = []

            for clip in clips:
                clip_start = clip['start']
                clip_end = clip['end']

                # Example: Suggest a zoom effect at high engagement moments
                for engagement_metric in engagement_analysis_results:
                    if clip_start <= engagement_metric['timestamp'] <= clip_end and \
                       engagement_metric['engagement_score'] > 0.8: # High engagement threshold
                        editing_decisions.append({
                            'type': 'zoom_effect',
                            'timestamp': engagement_metric['timestamp'],
                            'intensity': 'high',
                            'reason': 'high engagement'
                        })
                
                # Example: Suggest a cut based on LLM video director recommendations
                for llm_decision in llm_cut_decisions:
                    if clip_start <= llm_decision['start_time'] <= clip_end:
                        editing_decisions.append({
                            'type': 'llm_cut',
                            'start_time': llm_decision['start_time'],
                            'end_time': llm_decision['end_time'],
                            'reason': llm_decision['reason']
                        })

                # Example: Suggest beat-matched transitions (simplified)
                if audio_rhythm_data and 'beat_times' in audio_rhythm_data:
                    for beat_time in audio_rhythm_data['beat_times']:
                        if clip_start <= beat_time <= clip_end and abs(clip_start - beat_time) < 0.5: # Close to clip start
                             editing_decisions.append({
                                'type': 'beat_matched_transition',
                                'timestamp': beat_time,
                                'reason': 'audio beat alignment'
                            })

            context['dynamic_editing_decisions'] = editing_decisions
            self.log_info(f"Dynamic editing decisions generated. {len(editing_decisions)} decisions made.")
            set_stage_status('dynamic_editing_complete', 'complete', {'num_decisions': len(editing_decisions)})
            return True

        except Exception as e:
            self.log_error(f"Error during dynamic editing: {e}")
            set_stage_status('dynamic_editing', 'failed', {'reason': str(e)})
            return False
