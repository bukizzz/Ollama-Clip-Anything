from agents.base_agent import Agent
from core.state_manager import set_stage_status
from llm.llm_interaction import get_viral_recommendations_batch
import numpy as np

class ViralPotentialAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("ViralPotentialAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        # Updated paths to retrieve data from the hierarchical context
        clips = context.get('current_analysis', {}).get('clips', [])
        engagement_results = context.get('current_analysis', {}).get('multimodal_analysis_results', {}).get('engagement_metrics', [])
        hooks = context.get('identified_hooks', [])
        audio_analysis = context.get('current_analysis', {}).get('audio_analysis_results', {}) # Assuming audio analysis results are here

        if not clips:
            self.log_error("No clips found. Cannot assess viral potential.")
            set_stage_status('viral_potential', 'skipped', {'reason': 'No clips provided'})
            return context

        print("🧐 Assessing viral potential of selected clips...")
        set_stage_status('viral_potential', 'running')

        try:
            scored_clips = []
            recommendation_contexts = []

            for clip in clips:
                scenes = clip.get('scenes')
                if not scenes:
                    self.log_warning(f"Clip '{clip.get('clip_description', 'N/A')}' has no scenes. Skipping.")
                    continue

                # A clip's start is the start of its first scene, and its end is the end of its last scene.
                start = scenes[0].get('start_time')
                end = scenes[-1].get('end_time')

                if start is None or end is None:
                    self.log_warning(f"Clip '{clip.get('clip_description', 'N/A')}' has scenes with missing start/end times. Skipping.")
                    continue
                
                # Use 'score' key for engagement results as per MultimodalAnalysisAgent's output
                clip_engagement = [s['score'] for s in engagement_results if start <= s['timestamp'] <= end]
                avg_engagement = np.mean(clip_engagement) if clip_engagement else 0
                
                clip_hooks = [h for h in hooks if start <= h['timestamp'] <= end]
                hook_score = sum(h.get('quotability_score', 0) for h in clip_hooks)
                
                # Access sentiment from the correct hierarchical path
                sentiment = audio_analysis.get('sentiment', {}).get('label', 'NEUTRAL')
                sentiment_score = 0.2 if sentiment == 'POSITIVE' else -0.1 if sentiment == 'NEGATIVE' else 0
                
                viral_score = (avg_engagement * 0.5) + (hook_score * 0.3) + (sentiment_score * 0.2)
                viral_score = np.clip(viral_score * 10, 0, 10) # Scale to 0-10

                recommendation_contexts.append({
                    "start": start,
                    "end": end,
                    "viral_score": viral_score,
                    "clip_description": clip.get('reason', 'N/A'), # Use reason as description
                    "avg_engagement": avg_engagement,
                    "hook_score": hook_score,
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "original_clip": clip # Store original clip to merge later
                })
            
            if recommendation_contexts:
                print(f"🧠 Generating viral potential recommendations with LLM for {len(recommendation_contexts)} clips in batch...")
                batch_recommendations = get_viral_recommendations_batch(recommendation_contexts)

                for i, rec_context in enumerate(recommendation_contexts):
                    original_clip = rec_context['original_clip']
                    recommendation = batch_recommendations[i]
                    # Merge the original clip data with new viral potential score and recommendation
                    scored_clips.append({
                        **original_clip,
                        'viral_potential_score': rec_context['viral_score'],
                        'recommendation': recommendation
                    })
            else:
                self.log_warning("No valid clips to generate recommendations for.")

            context['current_analysis']['clips'] = sorted(scored_clips, key=lambda x: x['viral_potential_score'], reverse=True)
            print("✅ Viral potential assessment complete.")
            set_stage_status('viral_potential_scoring_complete', 'complete', {'num_clips_scored': len(scored_clips)})
            return context

        except Exception as e:
            self.log_error(f"Error during viral potential assessment: {e}")
            set_stage_status('viral_potential', 'failed', {'reason': str(e)})
            return context
