from agents.base_agent import Agent
from core.state_manager import set_stage_status
from llm import llm_interaction
import numpy as np
from pydantic import BaseModel, Field

class HookIdentificationAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("HookIdentificationAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        # Updated paths to retrieve data from the hierarchical context
        engagement_results = context.get('current_analysis', {}).get('multimodal_analysis_results', {}).get('engagement_metrics', [])
        transcription = context.get('archived_data', {}).get('full_transcription', [])
        
        if not engagement_results or not transcription:
            self.log_error("Engagement or transcription data missing. Cannot identify hooks.")
            set_stage_status('hook_identification', 'failed', {'reason': 'Missing dependencies'})
            return context

        print("🪝 Starting hook identification...")
        set_stage_status('hook_identification', 'running')

        try:
            # Identify top engagement peaks
            # Use 'score' key as per MultimodalAnalysisAgent's output
            scores = [s['score'] for s in engagement_results]
            
            if not scores:
                self.log_info("No engagement scores found to identify hooks.")
                context['identified_hooks'] = []
                set_stage_status('hook_identification_complete', 'complete', {'num_hooks': 0})
                return context

            peak_threshold = np.mean(scores) + 1.5 * np.std(scores)
            potential_hooks = [s for s in engagement_results if s['score'] > peak_threshold]

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
                    class HookAnalysis(BaseModel):
                        is_hook: bool = Field(description="True if the text is a strong narrative hook, False otherwise.")
                        quotability_score: float = Field(description="A score from 0 to 1 indicating how quotable the moment is.", ge=0, le=1)
                        reason: str = Field(description="Explanation for the assessment.")

                    # Updated system prompt to enforce JSON output
                    system_prompt_for_hook_analysis = """
                    You are an expert in identifying narrative hooks and quotable moments from text.
                    You MUST respond with ONLY a valid JSON object that strictly adheres to the following schema:

                    {
                        "is_hook": boolean,
                        "quotability_score": float (between 0 and 1),
                        "reason": string
                    }

                    Example:
                    {
                        "is_hook": true,
                        "quotability_score": 0.85,
                        "reason": "This phrase is highly impactful and sets up a clear conflict."
                    }

                    Do NOT include any other text, explanations, or markdown fences (e.g., ```json).
                    """

                    prompt = f"""
                    Analyze this text from a video transcript at {timestamp:.2f}s: "{text}"
                    Is this a strong narrative hook or a highly quotable moment?
                    """
                    print(f"🧠 Analyzing potential hook at {timestamp:.2f}s with LLM...")
                    try:
                        analysis_obj = llm_interaction.robust_llm_json_extraction(
                            system_prompt=system_prompt_for_hook_analysis, # Use the new system prompt
                            user_prompt=prompt,
                            output_schema=HookAnalysis
                        )
                        
                        if analysis_obj.is_hook or analysis_obj.quotability_score > 0.7:
                            hooks.append({
                                'timestamp': timestamp,
                                'text': text,
                                'engagement_score': hook['score'], # Changed to 'score'
                                'quotability_score': analysis_obj.quotability_score,
                                'reason': analysis_obj.reason
                            })
                    except Exception as e:
                        self.log_error(f"LLM analysis for hook at {timestamp:.2f}s failed: {e}")

            context['identified_hooks'] = sorted(hooks, key=lambda x: x['engagement_score'], reverse=True)
            print(f"✅ Hook identification complete. Identified {len(hooks)} potential hooks.")
            set_stage_status('hook_identification_complete', 'complete', {'num_hooks': len(hooks)})
            return context

        except Exception as e:
            self.log_error(f"Error during hook identification: {e}")
            set_stage_status('hook_identification', 'failed', {'reason': str(e)})
            return context
