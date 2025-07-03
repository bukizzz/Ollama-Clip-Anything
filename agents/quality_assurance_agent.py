from agents.base_agent import Agent
from core.state_manager import set_stage_status

class QualityAssuranceAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("QualityAssuranceAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        self.log_info("Performing final quality assurance checks...")
        set_stage_status('quality_assurance', 'running')

        try:
            # This agent performs final checks on the generated data before video rendering.
            # It can validate synchronization, subtitle accuracy, etc.

            report = {
                "audio_video_sync": {"status": "pass", "details": ""},
                "subtitle_accuracy": {"status": "pass", "details": ""},
                "layout_consistency": {"status": "pass", "details": ""},
                "overall_status": "pass"
            }

            # Example check: A/V sync (conceptual)
            # A real check would involve analyzing the output video, but we can infer from data.
            if not context.get('music_sync_results'):
                report['audio_video_sync'] = {"status": "warn", "details": "Music not synced."}

            # Example check: Subtitle validation
            if not context.get('animated_subtitle_paths'):
                report['subtitle_accuracy'] = {"status": "fail", "details": "No subtitles generated."}

            # Update overall status
            if any(v['status'] != 'pass' for v in report.values() if isinstance(v, dict)):
                report['overall_status'] = "fail"

            context['qa_report'] = report
            self.log_info(f"Quality assurance complete. Status: {report['overall_status']}")
            set_stage_status('quality_assurance_complete', 'complete', report)
            return True

        except Exception as e:
            self.log_error(f"Error in QualityAssuranceAgent: {e}")
            set_stage_status('quality_assurance', 'failed', {'reason': str(e)})
            return context