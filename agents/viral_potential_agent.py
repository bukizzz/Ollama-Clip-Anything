from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status
from core.config import ENGAGEMENT_ANALYSIS_CONFIG # Reusing for thresholds

class ViralPotentialAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.engagement_config = ENGAGEMENT_ANALYSIS_CONFIG

    def run(self, context):
        clips = context.get('clips')
        engagement_analysis_results = context.get('engagement_analysis_results')
        identified_hooks = context.get('identified_hooks')
        audio_analysis_results = context.get('audio_analysis_results')

        if not clips or not engagement_analysis_results:
            self.log_error("Clips or engagement analysis results not found. Cannot score viral potential.")
            set_stage_status('viral_potential_scoring', 'failed', {'reason': 'Missing clips or engagement data'})
            return False

        self.log_info("Starting viral potential scoring...")
        set_stage_status('viral_potential_scoring', 'running')

        try:
            scored_clips = []
            for clip in clips:
                clip_start = clip['start']
                clip_end = clip['end']
                
                # Aggregate engagement scores within the clip duration
                clip_engagement_scores = [
                    seg['engagement_score'] for seg in engagement_analysis_results
                    if clip_start <= seg['timestamp'] <= clip_end
                ]
                avg_engagement = sum(clip_engagement_scores) / len(clip_engagement_scores) if clip_engagement_scores else 0

                # Check for identified hooks within the clip
                has_hook = any(hook['timestamp'] >= clip_start and hook['timestamp'] <= clip_end for hook in identified_hooks)

                # Incorporate audio sentiment (simplified, ideally time-aligned sentiment)
                sentiment_score = 0
                if audio_analysis_results and audio_analysis_results.get('sentiment'):
                    sentiment_label = audio_analysis_results['sentiment']['label']
                    if sentiment_label == 'POSITIVE':
                        sentiment_score = 0.2 # Boost for positive sentiment
                    elif sentiment_label == 'NEGATIVE':
                        sentiment_score = -0.1 # Slight penalty for negative

                # Calculate viral potential score (0-10)
                # This is a simplified formula, can be made more complex with ML models
                viral_score = (avg_engagement * 5) + (2 if has_hook else 0) + (sentiment_score * 10)
                viral_score = max(0, min(10, viral_score)) # Clamp between 0 and 10

                clip['viral_potential_score'] = viral_score
                clip['viral_optimization_recommendations'] = "Consider adding dynamic text effects for key phrases." # Placeholder
                scored_clips.append(clip)

            context['clips'] = scored_clips # Update clips with viral potential scores
            self.log_info("Viral potential scoring complete.")
            set_stage_status('viral_potential_scoring_complete', 'complete', {'num_clips_scored': len(scored_clips)})
            return True

        except Exception as e:
            self.log_error(f"Error during viral potential scoring: {e}")
            set_stage_status('viral_potential_scoring', 'failed', {'reason': str(e)})
            return False
