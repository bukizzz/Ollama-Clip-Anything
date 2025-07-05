from agents.base_agent import Agent
from core.state_manager import set_stage_status
from typing import Dict, Any

class EngagementAnalysisAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("EngagementAnalysisAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.engagement_config = agent_config.get('engagement_analysis', {})

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]: # Added type hint for clarity
        video_analysis = context.get('video_analysis_results')
        if not video_analysis:
            self.log_error("Video analysis results not found. Cannot perform engagement analysis.")
            set_stage_status('engagement_analysis', 'failed', {'reason': 'Missing video analysis results'})
            return context

        print("📈 Starting engagement analysis...")
        set_stage_status('engagement_analysis', 'running')

        try:
            engagement_scores = []
            for i in range(len(video_analysis['facial_expressions'])):
                timestamp = video_analysis['facial_expressions'][i]['timestamp']
                
                # Score facial expressions
                expr = video_analysis['facial_expressions'][i]['expression']
                expr_score = 0.5 if expr == "happy" else 0.1 if expr == "neutral" else 0.8 if expr == "surprise" else 0
                
                # Score gestures
                gestures = video_analysis['gesture_recognition'][i]['gestures']
                gesture_score = len(gestures) * 0.2
                
                # Score energy levels
                energy = video_analysis['energy_levels'][i]['level']
                energy_score = min(energy / 10, 1.0) # Normalize
                
                # Viral potential score
                viral_score = (expr_score * 0.5) + (gesture_score * 0.3) + (energy_score * 0.2)
                
                engagement_scores.append({
                    'timestamp': timestamp,
                    'engagement_score': viral_score,
                    'details': {
                        'expression': expr,
                        'gestures': gestures,
                        'energy': energy
                    }
                })

            # Rank segments by engagement
            ranked_segments = sorted(engagement_scores, key=lambda x: x['engagement_score'], reverse=True)
            
            context['engagement_analysis_results'] = ranked_segments
            print("✅ Engagement analysis complete.")
            set_stage_status('engagement_analysis_complete', 'complete', {'num_segments_scored': len(ranked_segments)})
            return context # Changed from return True

        except Exception as e:
            self.log_error(f"Error during engagement analysis: {e}")
            set_stage_status('engagement_analysis', 'failed', {'reason': str(e)})
            return context
