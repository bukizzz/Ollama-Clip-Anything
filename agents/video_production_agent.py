import os
from agents.base_agent import Agent
from typing import Dict, Any
from core.state_manager import set_stage_status
from video import video_editing

class VideoProductionAgent(Agent):
    """
    Agent responsible for coordinating content enhancement, video editing, and final quality assurance.
    Combines logic from ContentEnhancementAgent, VideoEditingAgent, and QualityAssuranceAgent.
    """

    def __init__(self, config, state_manager):
        super().__init__("VideoProductionAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("🚀 Starting Video Production: Content Enhancement, Editing & QA...")
        set_stage_status('video_production', 'running')

        try:
            # --- Content Enhancement Logic (from ContentEnhancementAgent) ---
            # Check for required data from previous agents
            # Note: Some data might now be in 'current_analysis' or 'summaries'
            required_data_keys = [
                ('current_analysis', 'clips'),
                ('current_analysis', 'multimodal_analysis_results'), # For engagement
                ('summaries', 'engagement_summary'),
                ('b_roll_suggestions', None) # B-roll is still top-level for now
            ]
            
            for parent_key, child_key in required_data_keys:
                if child_key is None: # Top-level key
                    if parent_key not in context:
                        self.log_error(f"Missing required data for content enhancement: {parent_key}")
                        set_stage_status('video_production', 'failed', {'reason': f'Missing {parent_key}'})
                        return context
                else:
                    if parent_key not in context or child_key not in context[parent_key]:
                        self.log_error(f"Missing required data for content enhancement: {parent_key}.{child_key}")
                        set_stage_status('video_production', 'failed', {'reason': f'Missing {parent_key}.{child_key}'})
                        return context

            # Generate a comprehensive processing report
            num_clips = len(context.get('current_analysis', {}).get('clips', []))
            num_b_roll_suggestions = len(context.get('b_roll_suggestions', []))
            
            # Safely get average engagement from summary
            avg_engagement = context.get('summaries', {}).get('engagement_summary', {}).get('overall_stats', {}).get('mean', 0.0)

            processing_summary_report = {
                "num_clips": num_clips,
                "num_b_roll_suggestions": num_b_roll_suggestions,
                "avg_engagement": avg_engagement
            }
            context['processing_summary_report'] = processing_summary_report
            self.log_info("Content enhancement coordination complete.")

            # --- Video Editing Logic (from VideoEditingAgent) ---
            processed_video_path = context.get("processed_video_path")
            clips = context.get("current_analysis", {}).get("clips")
            transcription = context.get("archived_data", {}).get("full_transcription")
            video_info = context.get("metadata", {}).get("video_info")
            processing_settings = context.get("metadata", {}).get("processing_settings")
            
            # Use multimodal_analysis_results for video_analysis
            multimodal_analysis = context.get("current_analysis", {}).get("multimodal_analysis_results")
            
            # Use consolidated layout_speaker_analysis_results
            layout_speaker_results = context.get("current_analysis", {}).get("layout_speaker_analysis_results")

            # Ensure critical inputs are available for video editing
            if not all([processed_video_path, clips, transcription, video_info, processing_settings, multimodal_analysis, layout_speaker_results]):
                missing_data = []
                if not processed_video_path:
                    missing_data.append("processed_video_path")
                if not clips:
                    missing_data.append("clips")
                if not transcription:
                    missing_data.append("transcription")
                if not video_info:
                    missing_data.append("video_info")
                if not processing_settings:
                    missing_data.append("processing_settings")
                if not multimodal_analysis:
                    missing_data.append("multimodal_analysis_results")
                if not layout_speaker_results:
                    missing_data.append("layout_speaker_analysis_results")
                raise RuntimeError(f"Missing critical data for video editing: {', '.join(missing_data)}")

            # Initialize processing_report and created_clips
            created_clips = []
            processing_report = {
                'results': {'success_rate': 0.0},
                'performance_metrics': {'total_processing_time': 0.0},
                'failed_clip_numbers': []
            }

            self.log_info(f"Creating {len(clips)} enhanced video clips...")
            created_clips, processing_report = video_editing.batch_process_with_analysis(
                processed_video_path,
                clips,
                transcription,
                video_info=video_info,
                processing_settings=processing_settings,
                video_analysis=multimodal_analysis, # Pass multimodal_analysis_results
                audio_rhythm_data=context.get('current_analysis', {}).get('audio_analysis_results', {}).get('audio_rhythm_data', {}), # From audio_intelligence_agent
                llm_cut_decisions=context.get("current_analysis", {}).get("llm_cut_decisions", []), # From content_director_agent
                speaker_tracking_results=layout_speaker_results.get('speaker_tracking_results', {}), # From layout_speaker_agent
                output_dir=self.config.get('output_dir')
            )
            context['created_clips'] = created_clips
            context['processing_report'] = processing_report
            self.log_info("Video editing complete.")

            # --- Efficient Subtitle Generation ---
            from audio.subtitle_generation import generate_subtitles_efficiently
            animated_subtitle_paths = []
            for clip in created_clips:
                clip_name = os.path.basename(clip).split('.')[0]
                # Assuming transcription for the clip can be extracted or passed
                # For now, using the full transcription, but ideally it should be clip-specific
                subtitle_path = generate_subtitles_efficiently(
                    transcription,
                    self.config.get('output_dir'),
                    clip_name,
                    video_info.get('height')
                )
                animated_subtitle_paths.append(subtitle_path)
            context['animated_subtitle_paths'] = animated_subtitle_paths
            self.log_info("Efficient subtitle generation complete.")

            # --- Quality Assurance Logic (from QualityAssuranceAgent) ---
            self.log_info("Performing final quality assurance checks...")
            qa_report = {
                "audio_video_sync": {"status": "pass", "details": ""},
                "subtitle_accuracy": {"status": "pass", "details": ""},
                "layout_consistency": {"status": "pass", "details": ""},
                "overall_status": "pass"
            }

            # Example check: A/V sync (conceptual)
            audio_analysis_results = context.get('current_analysis', {}).get('audio_analysis_results', {})
            if not audio_analysis_results.get('audio_rhythm_data'):
                qa_report['audio_video_sync'] = {"status": "warn", "details": "Music rhythm data not available."}

            # Example check: Subtitle validation (conceptual, based on presence)
            # This would ideally check actual rendered subtitles
            if not context.get('animated_subtitle_paths'): # Assuming this is still generated by SubtitleAnimationAgent
                qa_report['subtitle_accuracy'] = {"status": "fail", "details": "No subtitles generated."}

            # Example check: Layout consistency (conceptual)
            if not layout_speaker_results.get('layout_detection_results'):
                qa_report['layout_consistency'] = {"status": "warn", "details": "Layout detection data not available."}

            # Update overall status
            if any(v['status'] != 'pass' for v in qa_report.values() if isinstance(v, dict)):
                qa_report['overall_status'] = "fail"

            context['qa_report'] = qa_report
            self.log_info(f"Quality assurance complete. Status: {qa_report['overall_status']}")

            set_stage_status('video_production_complete', 'complete', {
                'num_clips_created': len(created_clips),
                'qa_status': qa_report['overall_status']
            })
            return context

        except Exception as e:
            self.log_error(f"Error during video production: {e}")
            set_stage_status('video_production', 'failed', {'reason': str(e)})
            return context
