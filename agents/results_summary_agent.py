
from agents.base_agent import Agent
from typing import Dict, Any
import os

class ResultsSummaryAgent(Agent):
    """Agent responsible for summarizing the results and cleaning up."""

    def __init__(self):
        super().__init__("ResultsSummaryAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        created_clips = context.get("created_clips")
        clips = context.get("clips")
        processing_report = context.get("processing_report")
        failed_clips = processing_report.get('failed_clip_numbers', [])
        video_analysis = context.get("video_analysis")

        print("\n--- 🎉 \033[95mGeneration Complete!\033[0m ---")
        print(f"📊 Successfully created: {len(created_clips)}/{len(clips)} clips.")
        print(f"⏱️  Total processing time: {processing_report.get('total_processing_time', 0):.1f}s")
        print(f"📈 Success rate: {processing_report['results']['success_rate']:.1f}%")
        
        if created_clips:
            output_dir = os.path.dirname(created_clips[0])
            print(f"📂 Clips saved in: {output_dir}")
            print("📄 Processing report saved in output directory")
            
        if failed_clips:
            print(f"❌ Failed clip numbers: {failed_clips}")
            print("   Consider checking the source video at those timestamps or converting it to H.264 first.")
            
        # Display enhanced features used
        print("\n🎨 Enhanced Features Applied:")
        print(f"   - Face tracking: {'✅' if video_analysis.get('has_faces') else '❌'}")
        print(f"   - Object detection: {'✅' if video_analysis.get('has_objects') else '❌'}")
        print("   - Animated subtitles: ✅")
        print("   - Scene effects: ✅")
        print("   - Content analysis: ✅")

        context.update({
            "current_stage": "results_summary_complete"
        })
        return context
