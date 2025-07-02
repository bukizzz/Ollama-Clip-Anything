from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status
from core.config import ENGAGEMENT_ANALYSIS_CONFIG

class EngagementAnalysisAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.engagement_config = ENGAGEMENT_ANALYSIS_CONFIG

    def run(self, context):
        video_analysis_results = context.get('video_analysis_results')
        if not video_analysis_results or not video_analysis_results.get('engagement_metrics'):
            self.log_error("Engagement metrics not found in video analysis results. Cannot perform engagement analysis.")
            set_stage_status('engagement_analysis', 'failed', {'reason': 'Missing engagement metrics'})
            return False

        self.log_info("Starting engagement analysis...")
        set_stage_status('engagement_analysis', 'running')

        try:
            engagement_metrics = video_analysis_results['engagement_metrics']
            facial_expression_threshold = self.engagement_config.get('facial_expression_threshold', 0.7)
            gesture_detection_threshold = self.engagement_config.get('gesture_detection_threshold', 0.6)
            energy_level_threshold = self.engagement_config.get('energy_level_threshold', 0.5)

            scored_segments = []
            for metric in engagement_metrics:
                score = metric['score']
                timestamp = metric['timestamp']
                
                # Simple scoring logic based on thresholds
                # This would be more sophisticated with actual facial expression and gesture data
                viral_potential_score = 0
                if score > facial_expression_threshold: # Placeholder for actual facial expression score
                    viral_potential_score += 0.4
                if score > gesture_detection_threshold: # Placeholder for actual gesture score
                    viral_potential_score += 0.3
                if score > energy_level_threshold: # Placeholder for actual energy level score
                    viral_potential_score += 0.3

                scored_segments.append({
                    'timestamp': timestamp,
                    'engagement_score': score,
                    'viral_potential_score': viral_potential_score
                })
            
            context['engagement_analysis_results'] = scored_segments
            self.log_info("Engagement analysis complete.")
            set_stage_status('engagement_analysis_complete', 'complete', {'num_segments_scored': len(scored_segments)})
            return True

        except Exception as e:
            self.log_error(f"Error during engagement analysis: {e}")
            set_stage_status('engagement_analysis', 'failed', {'reason': str(e)})
            return False
