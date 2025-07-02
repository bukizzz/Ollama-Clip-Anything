from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status
from core.config import ENGAGEMENT_ANALYSIS_CONFIG # Reusing engagement config for thresholds

class HookIdentificationAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.engagement_config = ENGAGEMENT_ANALYSIS_CONFIG

    def run(self, context):
        engagement_analysis_results = context.get('engagement_analysis_results')
        audio_analysis_results = context.get('audio_analysis_results')
        transcription = context.get('transcription')

        if not engagement_analysis_results or not transcription:
            self.log_error("Engagement analysis results or transcription not found. Cannot identify hooks.")
            set_stage_status('hook_identification', 'failed', {'reason': 'Missing engagement or transcription'})
            return False

        self.log_info("Starting hook identification...")
        set_stage_status('hook_identification', 'running')

        try:
            hooks = []
            # Iterate through engagement metrics to find high-scoring moments
            for segment in engagement_analysis_results:
                timestamp = segment['timestamp']
                engagement_score = segment['engagement_score']
                viral_potential_score = segment['viral_potential_score']

                # Find corresponding transcript segment
                spoken_text = ""
                for trans_seg in transcription:
                    if trans_seg['start'] <= timestamp <= trans_seg['end']:
                        spoken_text = trans_seg['text']
                        break

                # Simple logic to identify hooks based on high engagement and viral potential
                if engagement_score > self.engagement_config.get('energy_level_threshold', 0.5) and \
                   viral_potential_score > self.engagement_config.get('facial_expression_threshold', 0.7):
                    
                    # Further refine with audio sentiment (if available)
                    sentiment = "unknown"
                    if audio_analysis_results and audio_analysis_results.get('sentiment'):
                        # Find sentiment for the current timestamp range
                        # This is a simplification; ideally, sentiment would be time-aligned
                        sentiment = audio_analysis_results['sentiment']['label']

                    hooks.append({
                        'timestamp': timestamp,
                        'spoken_text': spoken_text,
                        'engagement_score': engagement_score,
                        'viral_potential_score': viral_potential_score,
                        'sentiment': sentiment,
                        'hook_type': 'high_engagement'
                    })
            
            context['identified_hooks'] = hooks
            self.log_info(f"Hook identification complete. Identified {len(hooks)} hooks.")
            set_stage_status('hook_identification_complete', 'complete', {'num_hooks': len(hooks)})
            return True

        except Exception as e:
            self.log_error(f"Error during hook identification: {e}")
            set_stage_status('hook_identification', 'failed', {'reason': str(e)})
            return False