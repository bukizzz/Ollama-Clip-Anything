from agents.base_agent import Agent
from core.state_manager import set_stage_status
from llm import llm_interaction
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional
import time # Import time for sleep
import numpy as np # For statistical calculations
import json # For verbose printing

class CutDecision(BaseModel):
    start_time: float = Field(description="The start time of the clip segment in seconds.")
    end_time: float = Field(description="The end time of the clip segment in seconds.")
    reason: str = Field(description="A brief explanation for the cut decision (e.g., \"high engagement moment\", \"speaker transition\", \"scene change\").")
    key_elements: Optional[List[str]] = Field(default_factory=list, description="A list of key visual or audio elements present in this segment.")
    viral_potential_score: Optional[int] = Field(default=0, description="An estimated score for viral potential (0-10).", ge=0, le=10)

class CutDecisions(BaseModel):
    cut_decisions: List[CutDecision]

class LLMVideoDirectorAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("LLMVideoDirectorAgent")
        self.config = config
        self.state_manager = state_manager

    def _summarize_audio_rhythm(self, audio_rhythm_data: Dict[str, Any]) -> Dict[str, Any]:
        # This function will now return the full data for debugging purposes
        return audio_rhythm_data if audio_rhythm_data is not None else {}

    def _summarize_engagement_analysis(self, engagement_analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This function will now return the full data for debugging purposes
        return {"engagement_analysis_results": engagement_analysis_results} if engagement_analysis_results is not None else {}

    def _summarize_layout_detection(self, layout_detection_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This function will now return the full data for debugging purposes
        return {"layout_detection_results": layout_detection_results} if layout_detection_results is not None else {}

    def _summarize_speaker_tracking(self, speaker_tracking_results: Dict[str, Any]) -> Dict[str, Any]:
        # This function will now return the full data for debugging purposes
        return speaker_tracking_results if speaker_tracking_results is not None else {}

    def _summarize_initial_cut_decisions(self, initial_cut_decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # This function will now return the full data for debugging purposes
        return initial_cut_decisions if initial_cut_decisions is not None else []

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Gather all relevant analysis results from the context
        storyboard_data = context.get('storyboard_data')
        audio_rhythm_data = context.get('audio_rhythm_data')
        engagement_analysis_results = context.get('engagement_analysis_results')
        layout_detection_results = context.get('layout_detection_results')
        speaker_tracking_results = context.get('speaker_tracking_results')
        content_alignment_data = context.get('content_alignment_data')
        identified_hooks = context.get('identified_hooks')

        if not storyboard_data or not content_alignment_data:
            self.log_error("Missing essential analysis results for LLM Video Director. Skipping.")
            set_stage_status('llm_video_director', 'failed', {'reason': 'Missing essential data'})
            return context

        print("🎬 Starting LLM Video Director orchestration...")
        set_stage_status('llm_video_director', 'running')

        max_agent_retries = self.config.get('llm_agent_max_retries', 3) # Get from config or default to 3
        
        # --- First LLM Pass: Core Narrative and Content Cuts ---
        initial_cut_decisions_obj = None
        for attempt in range(max_agent_retries):
            try:
                self.log_info(f"Attempting LLM Video Director initial pass (Agent Retry {attempt + 1}/{max_agent_retries})...")
                
                llm_prompt_initial = f"""
                You are an expert video director. Your task is to analyze core video data
                and make intelligent *initial* cut decisions to create a coherent and engaging short clip.
                These decisions will be refined in a subsequent step.

                Here is the available analysis data:

                Storyboard Data:
                {storyboard_data}

                Content Alignment Data:
                {content_alignment_data}

                Identified Hooks:
                {identified_hooks}

                Based on this data, provide a detailed plan for *initial* video cuts. Focus on:
                1. Identifying optimal start and end points for clips, considering narrative coherence and main content flow.
                2. Highlighting significant events and key narrative beats.
                3. Suggesting preliminary cut decisions that capture the essence of the video.

                Provide your response as a JSON object with a single key "cut_decisions" which is a list of cut decision objects.
                Ensure 'key_elements' is a JSON array of strings.
                EXAMPLE OUTPUT: 
{{
  "cut_decisions": [
    {{
      "start_time": 0.0,
      "end_time": 2.0,
      "reason": "Strong opening with Joe Rogan Experience backdrop and hearty laughter from the middle-aged man. High hook potential.",
      "key_elements": ["Intense group laughter"],
      "viral_potential_score": 0
    }},
    {{
      "start_time": 21.1211,
      "end_time": 26.0,
      "reason": "Joe Rogan speaking into the microphone with the 'Joe Rogan Experience' backdrop. High hook potential.",
      "key_elements": ["Joe Rogan speaking"],
      "viral_potential_score": 0
    }}
  ]
}}               
                """
                print("🧠 Sending core data to LLM for initial video direction...")
                initial_cut_decisions_obj = llm_interaction.robust_llm_json_extraction(
                    system_prompt="You are an expert video director, making intelligent initial cut decisions.",
                    user_prompt=llm_prompt_initial.strip(),
                    output_schema=CutDecisions
                )
                break # Break if successful
            except RuntimeError as e:
                if "Token quota exhausted" in str(e) or "Daily request quota exhausted" in str(e) or "limit_hits" in str(e):
                    self.log_warning(f"LLM quota limit hit for initial pass (Attempt {attempt + 1}): {e}")
                    if attempt < max_agent_retries - 1:
                        self.log_info("Attempting to switch LLM model and retry initial pass...")
                        from core.config import config as global_config
                        global_config.update_llm_active_model('llm_model')
                        time.sleep(self.config.get('llm_retry_delay', 2))
                    else:
                        self.log_error(f"Max agent retries ({max_agent_retries}) exhausted for initial pass due to quota limits.")
                        set_stage_status('llm_video_director', 'failed', {'reason': f"Max retries exhausted for initial pass due to LLM quota: {e}"})
                        return context
                else:
                    self.log_error(f"Error during LLM Video Director initial pass: {e}")
                    set_stage_status('llm_video_director', 'failed', {'reason': str(e)})
                    return context
            except Exception as e:
                self.log_error(f"Error during LLM Video Director initial pass: {e}")
                set_stage_status('llm_video_director', 'failed', {'reason': str(e)})
                return context
        
        if not initial_cut_decisions_obj:
            self.log_error("Initial LLM pass failed to generate cut decisions.")
            set_stage_status('llm_video_director', 'failed', {'reason': 'Initial LLM pass failed'})
            return context

        # Convert Pydantic models to dictionaries for JSON serialization for the next pass
        initial_cut_decisions_dicts = [decision.model_dump() for decision in initial_cut_decisions_obj.cut_decisions]

        # Add a 63-second delay as requested
        self.log_info("Pausing for 63 seconds before the refinement pass to respect API limits.")
        time.sleep(63)

        # For debugging, pass original data and print it
        # Summarize auxiliary data for the second pass, handling potential None values
        # Temporarily use original data for verbose output
        verbose_audio_rhythm = audio_rhythm_data if audio_rhythm_data is not None else {}
        verbose_engagement_analysis = engagement_analysis_results if engagement_analysis_results is not None else []
        verbose_layout_detection = layout_detection_results if layout_detection_results is not None else [] 
        verbose_speaker_tracking = speaker_tracking_results if speaker_tracking_results is not None else {}
        verbose_initial_cut_decisions = initial_cut_decisions_dicts # Use the full list

        print("\n--- VERBOSE INPUT FOR SECOND LLM PASS (DEBUGGING) ---")
        print("Initial Cut Decisions (Full):")
        print(json.dumps(verbose_initial_cut_decisions, indent=2))
        print("\nAudio Rhythm Data (Full):")
        print(json.dumps(verbose_audio_rhythm, indent=2))
        print("\nEngagement Analysis Results (Full):")
        print(json.dumps(verbose_engagement_analysis, indent=2))
        print("\nLayout Detection Results (Full):")
        print(json.dumps(verbose_layout_detection, indent=2))
        print("\nSpeaker Tracking Results (Full):")
        print(json.dumps(verbose_speaker_tracking, indent=2))
        print("--- END VERBOSE INPUT ---")


        # --- Second LLM Pass: Refine Cuts with Auxiliary Data ---
        final_cut_decisions_obj = None
        for attempt in range(max_agent_retries):
            try:
                self.log_info(f"Attempting LLM Video Director refinement pass (Agent Retry {attempt + 1}/{max_agent_retries})...")
                
                llm_prompt_refinement = f"""
                You are an expert video director. Your task is to refine a set of preliminary video cut decisions
                based on additional detailed analysis data.

                Here are the preliminary cut decisions you generated:
                {json.dumps(verbose_initial_cut_decisions, indent=2)}

                Here is the additional analysis data for refinement:

                Audio Rhythm Data:
                {json.dumps(verbose_audio_rhythm, indent=2)}

                Engagement Analysis Results:
                {json.dumps(verbose_engagement_analysis, indent=2)}

                Layout Detection Results:
                {json.dumps(verbose_layout_detection, indent=2)}

                Speaker Tracking Results:
                {json.dumps(verbose_speaker_tracking, indent=2)}

                Based on these preliminary decisions and the new data, refine the video cuts. Focus on:
                1. Adjusting start and end points for clips to align with audio rhythm, engagement peaks, and speaker transitions.
                2. Incorporating insights from layout detection to ensure visually dynamic cuts.
                3. Maximizing viral potential and viewer retention by emphasizing key moments identified in the new data.
                4. You may add new cut decisions or modify existing ones as needed.

                Provide your response as a JSON object with a single key "cut_decisions" which is a list of refined cut decision objects.
                Ensure the 'viral_potential_score' is updated and 'key_elements' is a JSON array of strings based on the new information.
                EXAMPLE OUTPUT: 
{{
  "cut_decisions": [
    {{
      "start_time": 0.0,
      "end_time": 2.0,
      "reason": "Strong opening with Joe Rogan Experience backdrop and hearty laughter from the middle-aged man. High hook potential and good layout with many faces. Layout detection shows a lot of layout changes, indicating dynamic visual interest. Engagement score is also high at the beginning.",
      "key_elements": ["Intense group laughter"],
      "viral_potential_score": 8
    }},
    {{
      "start_time": 21.1211,
      "end_time": 26.0,
      "reason": "Joe Rogan speaking into the microphone with the 'Joe Rogan Experience' backdrop. High hook potential. Layout detection shows a lot of faces, indicating dynamic visual interest. Engagement score is also high at the beginning.",
      "key_elements": ["Highly dynamic scene"],
      "viral_potential_score": 9
    }}
  ]
}}               
                """
                print("🧠 Sending preliminary cuts and auxiliary data to LLM for refinement...")
                final_cut_decisions_obj = llm_interaction.robust_llm_json_extraction(
                    system_prompt="You are an expert video director, refining preliminary cut decisions.",
                    user_prompt=llm_prompt_refinement.strip(),
                    output_schema=CutDecisions
                )
                break # Break if successful
            except RuntimeError as e:
                if "Token quota exhausted" in str(e) or "Daily request quota exhausted" in str(e) or "limit_hits" in str(e):
                    self.log_warning(f"LLM quota limit hit for refinement pass (Attempt {attempt + 1}): {e}")
                    if attempt < max_agent_retries - 1:
                        self.log_info("Attempting to switch LLM model and retry refinement pass...")
                        from core.config import config as global_config
                        global_config.update_llm_active_model('llm_model')
                        time.sleep(self.config.get('llm_retry_delay', 2))
                    else:
                        self.log_error(f"Max agent retries ({max_agent_retries}) exhausted for refinement pass due to quota limits.")
                        set_stage_status('llm_video_director', 'failed', {'reason': f"Max retries exhausted for refinement pass due to LLM quota: {e}"})
                        return context
                else:
                    self.log_error(f"Error during LLM Video Director refinement pass: {e}")
                    set_stage_status('llm_video_director', 'failed', {'reason': str(e)})
                    return context
            except Exception as e:
                self.log_error(f"Error during LLM Video Director refinement pass: {e}")
                set_stage_status('llm_video_director', 'failed', {'reason': str(e)})
                return context

        if not final_cut_decisions_obj:
            self.log_error("Refinement LLM pass failed to generate cut decisions.")
            set_stage_status('llm_video_director', 'failed', {'reason': 'Refinement LLM pass failed'})
            return context

        final_cut_decisions_dicts = [decision.model_dump() for decision in final_cut_decisions_obj.cut_decisions]

        context['llm_cut_decisions'] = final_cut_decisions_dicts
        print(f"✅ LLM Video Director orchestration complete. Generated {len(final_cut_decisions_dicts)} refined cut decisions.")
        set_stage_status('llm_video_director_complete', 'complete', {'num_cut_decisions': len(final_cut_decisions_dicts)})
        return context
