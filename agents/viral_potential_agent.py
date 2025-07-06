from agents.base_agent import Agent
from core.state_manager import set_stage_status
from llm import llm_interaction
import numpy as np
from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    recommendation: str = Field(description="A brief, actionable recommendation to enhance the clip's virality.")

class ViralPotentialAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("ViralPotentialAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        clips = context.get('clips', [])
        engagement_results = context.get('engagement_analysis_results', [])
        hooks = context.get('identified_hooks', [])
        audio_analysis = context.get('audio_analysis_results', {})

        if not clips:
            self.log_error("No clips found. Cannot assess viral potential.")
            set_stage_status('viral_potential', 'skipped', {'reason': 'No clips provided'})
            return context

        print("🧐 Assessing viral potential of selected clips...")
        set_stage_status('viral_potential', 'running')

        try:
            scored_clips = []
            for clip in clips:
                # Extract start and end times from the first and last scene of the clip
                if not clip.get('scenes'):
                    self.log_warning(f"Clip {clip.get('clip_description', 'N/A')} has no scenes. Skipping.")
                    continue
                
                start = clip['scenes'][0]['start_time']
                end = clip['scenes'][-1]['end_time']
                
                # Engagement score for the clip
                clip_engagement = [s['engagement_score'] for s in engagement_results if start <= s['timestamp'] <= end]
                avg_engagement = np.mean(clip_engagement) if clip_engagement else 0
                
                # Hook and quotability score
                clip_hooks = [h for h in hooks if start <= h['timestamp'] <= end]
                hook_score = sum(h.get('quotability_score', 0) for h in clip_hooks)
                
                # Emotional impact from audio
                sentiment = audio_analysis.get('sentiment', {}).get('label', 'NEUTRAL')
                sentiment_score = 0.2 if sentiment == 'POSITIVE' else -0.1 if sentiment == 'NEGATIVE' else 0
                
                # Final viral score
                viral_score = (avg_engagement * 0.5) + (hook_score * 0.3) + (sentiment_score * 0.2)
                viral_score = np.clip(viral_score * 10, 0, 10) # Scale to 0-10

                # LLM-based optimization recommendations
                print(f"🧠 Generating viral potential recommendations with LLM for clip {start:.2f}s - {end:.2f}s...")
                prompt = f"""
                A video clip from {start:.2f}s to {end:.2f}s has a viral potential score of {viral_score:.2f}/10.
                Content: "{clip['clip_description']}"
                Provide a brief, actionable recommendation to enhance its virality.
                """
                try:
                    recommendation_obj = llm_interaction.robust_llm_json_extraction(
                        system_prompt="You are an expert in generating actionable recommendations for video virality.",
                        user_prompt=prompt,
                        output_schema=Recommendation
                    )
                    recommendation = recommendation_obj.recommendation
                except Exception:
                    recommendation = "N/A"

                scored_clips.append({**clip, 'viral_potential_score': viral_score, 'recommendation': recommendation, 'start': start, 'end': end})

            context['clips'] = sorted(scored_clips, key=lambda x: x['viral_potential_score'], reverse=True)
            print("✅ Viral potential assessment complete.")
            set_stage_status('viral_potential_scoring_complete', 'complete', {'num_clips_scored': len(scored_clips)})
            return context

        except Exception as e:
            self.log_error(f"Error during viral potential assessment: {e}")
            set_stage_status('viral_potential', 'failed', {'reason': str(e)})
            return context
