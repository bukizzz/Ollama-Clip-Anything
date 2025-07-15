import time
from agents.base_agent import Agent
from core.state_manager import set_stage_status
from llm import llm_interaction
from llm.llm_interaction import get_model_details, switch_active_model
import numpy as np
from pydantic import BaseModel, Field
import concurrent.futures # Import concurrent.futures

class HookAnalysis(BaseModel):
    is_hook: bool = Field(description="True if the text is a strong narrative hook, False otherwise.")
    quotability_score: float = Field(description="A score from 0 to 1 indicating how quotable the moment is.", ge=0, le=1)
    reason: str = Field(description="Explanation for the assessment.")

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

Do NOT include any other text, explanations, or markdown fences (e.g.,
"""

class HookIdentificationAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("HookIdentificationAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context):
        stage_name = self.name
        print(f"\nExecuting stage: {stage_name}")

        # --- Pre-flight Check ---
        if context.get('identified_hooks') and context.get('pipeline_stages', {}).get(stage_name) == 'complete':
            print(f"✅ Skipping {stage_name}: Hooks already identified.")
            return context

        engagement_results = context.get('current_analysis', {}).get('multimodal_analysis_results', {}).get('engagement_metrics', [])
        transcription = context.get('archived_data', {}).get('full_transcription', [])
        
        if not engagement_results or not transcription:
            self.log_error("Engagement or transcription data missing. Cannot identify hooks.")
            set_stage_status('hook_identification', 'failed', {'reason': 'Missing dependencies'})
            return context

        print("🪝 Starting hook identification...")
        set_stage_status('hook_identification', 'running')

        try:
            scores = [s['score'] for s in engagement_results]
            
            if not scores:
                self.log_info("No engagement scores found to identify hooks.")
                context['identified_hooks'] = []
                set_stage_status('hook_identification_complete', 'complete', {'num_hooks': 0})
                return context

            peak_threshold = np.mean(scores) + 1.5 * np.std(scores)
            potential_hooks = [s for s in engagement_results if s['score'] > peak_threshold]

            llm_tasks = []
            for hook in potential_hooks:
                timestamp = hook['timestamp']
                text = ""
                for seg in transcription:
                    if seg['start'] <= timestamp <= seg['end']:
                        text = seg['text']
                        break
                
                if text:
                    prompt = f"""
                    Analyze this text from a video transcript at {timestamp:.2f}s: "{text}"
                    Is this a strong narrative hook or a highly quotable moment?
                    """
                    llm_tasks.append((timestamp, text, hook['score'], prompt))

            hooks = []
            if llm_tasks:
                print(f"🧠 Analyzing {len(llm_tasks)} potential hooks with LLM using batching and model rotation...")
                
                remaining_tasks = list(llm_tasks)
                
                while remaining_tasks:
                    current_model_name = self.config.get('llm.current_active_llm_model')
                    model_details = get_model_details(current_model_name)
                    
                    if not model_details:
                        self.log_error(f"Could not get details for current active model: {current_model_name}. Cannot proceed with LLM tasks.")
                        break # Exit loop if no model details

                    requests_per_minute = model_details.get('requests_per_minute', float('inf'))
                    
                    batch_size = int(requests_per_minute) if requests_per_minute != float('inf') else len(remaining_tasks)
                    
                    if batch_size == 0: # Avoid infinite loop if RPM is very low or 0
                        self.log_warning(f"Requests per minute for {current_model_name} is 0. Skipping this model.")
                        switch_active_model('llm_model') # Try next model
                        time.sleep(1) # Small delay before next iteration
                        continue

                    current_batch = remaining_tasks[:batch_size]
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                        future_to_hook = {executor.submit(llm_interaction.robust_llm_json_extraction,
                                                           system_prompt=system_prompt_for_hook_analysis,
                                                           user_prompt=task_prompt,
                                                           output_schema=HookAnalysis): (timestamp, text, engagement_score)
                                          for timestamp, text, engagement_score, task_prompt in current_batch}

                        for future in concurrent.futures.as_completed(future_to_hook):
                            timestamp, text, engagement_score = future_to_hook[future]
                            try:
                                analysis_obj = future.result()
                                if analysis_obj.is_hook or analysis_obj.quotability_score > 0.7:
                                    hooks.append({
                                        'timestamp': timestamp,
                                        'text': text,
                                        'engagement_score': engagement_score,
                                        'quotability_score': analysis_obj.quotability_score,
                                        'reason': analysis_obj.reason
                                    })
                            except Exception as e:
                                self.log_error(f"LLM analysis for hook at {timestamp:.2f}s failed: {e}")
                                # If an error occurs, especially quota exhaustion, switch model
                                if "quota exhausted" in str(e).lower():
                                    self.log_warning(f"Quota exhausted for {current_model_name}. Switching model.")
                                    switch_active_model('llm_model')
                                    # Break from inner loop to restart with new model
                                    break 
                        else: # This else block executes if the inner for loop completes without a 'break'
                            # If the batch completed successfully, remove processed tasks
                                remaining_tasks = remaining_tasks[batch_size:]
                                if remaining_tasks: # If there are more tasks, switch to the next model
                                    self.log_info(f"Batch of {len(current_batch)} requests completed for {current_model_name}. Switching to next model for remaining tasks.")
                                    switch_active_model('llm_model')
                                    # The while loop will continue, picking up the new active model
                                    
            context['identified_hooks'] = sorted(hooks, key=lambda x: x['engagement_score'], reverse=True)
            print(f"✅ Hook identification complete. Identified {len(hooks)} potential hooks.")
            set_stage_status('hook_identification_complete', 'complete', {'num_hooks': len(hooks)})
            return context

        except Exception as e:
            self.log_error(f"Error during hook identification: {e}")
            set_stage_status('hook_identification', 'failed', {'reason': str(e)})
            return context
