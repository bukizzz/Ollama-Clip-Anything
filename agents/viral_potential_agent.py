from agents.base_agent import Agent
from core.state_manager import set_stage_status
from llm import llm_interaction
import numpy as np

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

        self.log_info("Assessing viral potential of selected clips...")
        set_stage_status('viral_potential', 'running')

        try:
            scored_clips = []
            for clip in clips:
                start, end = clip['start'], clip['end']
                
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
                self.log_info(f"🧠 \u001b[94mGenerating viral potential recommendations with LLM for clip {start:.2f}s - {end:.2f}s...\u001b[0m")
                prompt = f"""
                A video clip from {start:.2f}s to {end:.2f}s has a viral potential score of {viral_score:.2f}/10.
                Content: "{clip['text']}"
                Provide a brief, actionable recommendation to enhance its virality (e.g., "Add a text overlay for the key question").
                Output JSON: {{"recommendation": "..."}}
                """
                try:
                    response = llm_interaction.llm_pass(llm_interaction.LLM_MODEL, [{"role": "user", "content": prompt}])
                    recommendation = llm_interaction.extract_json_from_text(response).get('recommendation', '')
                except Exception:
                    recommendation = "N/A"

                scored_clips.append({**clip, 'viral_potential_score': viral_score, 'recommendation': recommendation})

            context['clips'] = sorted(scored_clips, key=lambda x: x['viral_potential_score'], reverse=True)
            self.log_info("Viral potential assessment complete.")
            set_stage_status('viral_potential_scoring_complete', 'complete', {'num_clips_scored': len(scored_clips)})
            return True

        except Exception as e:
            self.log_error(f"Error during viral potential assessment: {e}")
            set_stage_status('viral_potential', 'failed', {'reason': str(e)})
            return context
