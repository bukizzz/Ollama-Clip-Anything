from agents.base_agent import Agent
from core.state_manager import set_stage_status
from llm import llm_interaction
import numpy as np

class HookIdentificationAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("HookIdentificationAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        engagement_results = context.get('engagement_analysis_results', [])
        transcription = context.get('transcription', [])
        
        if not engagement_results or not transcription:
            self.log_error("Engagement or transcription data missing. Cannot identify hooks.")
            set_stage_status('hook_identification', 'failed', {'reason': 'Missing dependencies'})
            return context

        self.log_info("Starting hook identification...")
        set_stage_status('hook_identification', 'running')

        try:
            # Identify top engagement peaks
            scores = [s['engagement_score'] for s in engagement_results]
            peak_threshold = np.mean(scores) + 1.5 * np.std(scores)
            potential_hooks = [s for s in engagement_results if s['engagement_score'] > peak_threshold]

            hooks = []
            for hook in potential_hooks:
                timestamp = hook['timestamp']
                
                # Find corresponding transcript text
                text = ""
                for seg in transcription:
                    if seg['start'] <= timestamp <= seg['end']:
                        text = seg['text']
                        break
                
                if text:
                    # Use LLM to evaluate quotability and narrative potential
                    prompt = f"""
                    Analyze this text from a video transcript at {timestamp:.2f}s: "{text}"
                    Is this a strong narrative hook or a highly quotable moment?
                    Provide a JSON response with: {{"is_hook": boolean, "quotability_score": float (0-1), "reason": "..."}}
                    """
                    self.log_info(f"🧠 \u001b[94mAnalyzing potential hook at {timestamp:.2f}s with LLM...\u001b[0m")
                    try:
                        response = llm_interaction.llm_pass(llm_interaction.LLM_MODEL, [{"role": "user", "content": prompt}])
                        analysis = llm_interaction.extract_json_from_text(response)
                        
                        if analysis.get('is_hook') or analysis.get('quotability_score', 0) > 0.7:
                            hooks.append({
                                'timestamp': timestamp,
                                'text': text,
                                'engagement_score': hook['engagement_score'],
                                'quotability_score': analysis.get('quotability_score', 0),
                                'reason': analysis.get('reason', '')
                            })
                    except Exception as e:
                        self.log_error(f"LLM analysis for hook at {timestamp:.2f}s failed: {e}")

            context['identified_hooks'] = sorted(hooks, key=lambda x: x['engagement_score'], reverse=True)
            self.log_info(f"Hook identification complete. Identified {len(hooks)} potential hooks.")
            set_stage_status('hook_identification_complete', 'complete', {'num_hooks': len(hooks)})
            return True

        except Exception as e:
            self.log_error(f"Error during hook identification: {e}")
            set_stage_status('hook_identification', 'failed', {'reason': str(e)})
            return context
