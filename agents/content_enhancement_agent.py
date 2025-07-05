from agents.base_agent import Agent
from core.state_manager import set_stage_status

class ContentEnhancementAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("ContentEnhancementAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        print("🚀 Coordinating content enhancement pipeline...")
        set_stage_status('content_enhancement', 'running')

        try:
            # This agent orchestrates the final steps, ensuring data is ready for video editing.
            # It can perform quality checks and final optimizations.

            # Example: Check for required data from previous agents
            required_data = ['clips', 'audio_rhythm_data', 'engagement_analysis_results', 'b_roll_suggestions']
            for data_key in required_data:
                if data_key not in context:
                    self.log_error(f"Missing required data for content enhancement: {data_key}")
                    set_stage_status('content_enhancement', 'failed', {'reason': f'Missing {data_key}'})
                    return context

            # Example: Generate a comprehensive processing report (can be expanded)
            report = {
                "num_clips": len(context.get('clips', [])),
                "num_b_roll_suggestions": len(context.get('b_roll_suggestions', [])),
                "avg_engagement": sum(s['engagement_score'] for s in context.get('engagement_analysis_results', [])) / len(context.get('engagement_analysis_results', []))
            }
            context['processing_summary_report'] = report

            print("✅ Content enhancement pipeline coordinated successfully.")
            set_stage_status('content_enhancement_complete', 'complete', report)
            return context

        except Exception as e:
            self.log_error(f"Error in ContentEnhancementAgent: {e}")
            set_stage_status('content_enhancement', 'failed', {'reason': str(e)})
            return context