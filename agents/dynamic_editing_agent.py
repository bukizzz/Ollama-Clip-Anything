from agents.base_agent import Agent
from core.state_manager import set_stage_status

class DynamicEditingAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("DynamicEditingAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        clips = context.get('clips', [])
        audio_rhythm = context.get('audio_rhythm_data', {})
        engagement = context.get('engagement_analysis_results', [])
        llm_director_cuts = context.get('llm_cut_decisions', [])

        if not clips:
            self.log_error("No clips for dynamic editing.")
            set_stage_status('dynamic_editing', 'skipped', {'reason': 'No clips'})
            return context

        print("✨ Generating dynamic editing decisions...")
        set_stage_status('dynamic_editing', 'running')

        try:
            editing_decisions = []
            for clip in clips:
                # Extract start and end times from the first and last scene of the clip
                if not clip.get('scenes'):
                    self.log_warning(f"Clip {clip.get('clip_description', 'N/A')} has no scenes. Skipping.")
                    continue
                
                start = clip['scenes'][0]['start_time']
                end = clip['scenes'][-1]['end_time']
                
                # Optimal cut points from LLM Director
                for cut in llm_director_cuts:
                    if start <= cut['start_time'] <= end:
                        editing_decisions.append({'time': cut['start_time'], 'type': 'cut', 'reason': cut['reason']})

                # Dynamic effects from engagement and rhythm
                for eng in engagement:
                    if start <= eng['timestamp'] <= end and eng['engagement_score'] > 0.75:
                        editing_decisions.append({'time': eng['timestamp'], 'type': 'zoom_in', 'intensity': 'medium'})
                
                if 'beat_times' in audio_rhythm:
                    for beat in audio_rhythm['beat_times']:
                        if start <= beat <= end:
                            # Add a small effect on every beat
                            editing_decisions.append({'time': beat, 'type': 'pulse', 'intensity': 'low'})

            # Pacing optimization (conceptual)
            # Could involve adjusting clip speed or adding/removing small segments based on engagement flow
            
            context['dynamic_editing_decisions'] = editing_decisions
            print(f"✅ Generated {len(editing_decisions)} dynamic editing decisions.")
            set_stage_status('dynamic_editing_complete', 'complete', {'num_decisions': len(editing_decisions)})
            return context

        except Exception as e:
            self.log_error(f"Error in DynamicEditingAgent: {e}")
            set_stage_status('dynamic_editing', 'failed', {'reason': str(e)})
            return context
