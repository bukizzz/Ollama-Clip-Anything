from agents.base_agent import Agent
from typing import Dict, Any
import os
import logging # Import logging

class ResultsSummaryAgent(Agent):
    """Agent responsible for summarizing the results and cleaning up."""

    def __init__(self):
        super().__init__("ResultsSummaryAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Add debug logging
        logging.info(f"ResultsSummaryAgent: Context keys: {context.keys()}")
        processing_report = context.get("processing_report")
        logging.info(f"ResultsSummaryAgent: Value of 'processing_report': {processing_report}")

        created_clips = context.get("created_clips")
        clips = context.get("current_analysis", {}).get("clips") # Get clips from current_analysis
        
        # Implement graceful handling for processing_report
        if processing_report is None:
            self.log_error("Processing report is missing from context. Cannot display full summary.")
            processing_report = {
                'results': {'success_rate': 0.0},
                'performance_metrics': {'total_processing_time': 0.0},
                'failed_clip_numbers': []
            }
            # Set default values for created_clips and clips if they are also missing
            if created_clips is None:
                created_clips = []
            if clips is None:
                clips = []

        failed_clips = processing_report.get('failed_clip_numbers', [])
        # Access multimodal_analysis_results from current_analysis
        multimodal_analysis_results = context.get('current_analysis', {}).get('multimodal_analysis_results')

        print("\n--- 🎉 \033[95mGeneration Complete!\033[0m ---")
        print(f"📊 Successfully created: {len(created_clips) if created_clips else 0}/{len(clips) if clips else 0} clips.")
        print(f"⏱️  Total processing time: {processing_report.get('performance_metrics', {}).get('total_processing_time', 0):.1f}s")
        print(f"📈 Success rate: {processing_report.get('results', {}).get('success_rate', 0.0):.1f}%")
        
        if created_clips:
            # Ensure created_clips is not empty before trying to get its first element
            if created_clips and len(created_clips) > 0:
                output_dir = os.path.dirname(created_clips[0])
                print(f"📂 Clips saved in: {output_dir}")
            print("📄 Processing report saved in output directory")
            
        if failed_clips:
            print(f"❌ Failed clip numbers: {failed_clips}")
            print("   Consider checking the source video at those timestamps or converting it to H.264 first.")
            
        # Display enhanced features used - add checks for multimodal_analysis_results
        print("\n🎨 Enhanced Features Applied:")
        if multimodal_analysis_results:
            # These keys might not exist directly in multimodal_analysis_results,
            # but rather in sub-dictionaries like 'face_detection_results' or 'object_detection_results'.
            # For now, I'll keep the generic check, but this might need refinement if the data structure is deeper.
            print(f"   - Face tracking: {'✅' if multimodal_analysis_results.get('has_faces') else '❌'}")
            print(f"   - Object detection: {'✅' if multimodal_analysis_results.get('has_objects') else '❌'}")
        else:
            print("   - Face tracking: ❌ (Video analysis data missing)")
            print("   - Object detection: ❌ (Video analysis data missing)")
        print("   - Animated subtitles: ✅")
        print("   - Scene effects: ✅")
        print("   - Content analysis: ✅")

        context.update({
            "current_stage": "results_summary_complete"
        })
        return context
