from agents.base_agent import Agent
from typing import Dict, Any, List, Optional
from llm import llm_interaction
from llm.prompt_utils import build_adaptive_prompt
from pydantic import BaseModel, Field
from core.state_manager import set_stage_status
import time

class ContentAlignment(BaseModel):
    start_time: float = Field(description="The start time in seconds of the aligned moment.")
    end_time: float = Field(description="The end time in seconds of the aligned moment.")
    speaker: str = Field(description="The ID of the speaker.")
    transcript_segment: str = Field(description="The spoken content during this segment.")
    visual_description: str = Field(description="A brief description of the visual content at that moment.")
    scene_change: bool = Field(description="True if a scene change was detected, false otherwise.")
    topic_shift: Optional[bool] = Field(None, description="True if a topic shift was detected, false otherwise. (Optional)")
    engagement_impact: str = Field(description="Engagement metric for the segment.")
    rhythm_correlation: Optional[str] = Field(None, description="Audio rhythm analysis for the segment. (Optional)")

class ContentAlignments(BaseModel):
    alignments: List[ContentAlignment]

class CutDecision(BaseModel):
    start_time: float = Field(description="The start time of the clip segment in seconds.")
    end_time: float = Field(description="The end time of the clip segment in seconds.")
    reason: str = Field(description="A brief explanation for the cut decision (e.g., \"high engagement moment\", \"speaker transition\", \"scene change\").")
    key_elements: Optional[List[str]] = Field(default_factory=list, description="A list of key visual or audio elements present in this segment.")
    viral_potential_score: Optional[int] = Field(default=0, description="An estimated score for viral potential (0-10).", ge=0, le=10)

class CutDecisions(BaseModel):
    cut_decisions: List[CutDecision]

class ContentDirectorAgent(Agent):
    """
    Agent responsible for content alignment and video direction, combining logic from:
    - ContentAlignmentAgent
    - LLMVideoDirectorAgent
    """

    def __init__(self, config, state_manager):
        super().__init__("ContentDirectorAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        full_transcription = context.get("archived_data", {}).get("full_transcription")
        storyboard_data = context.get("storyboard_data")
        audio_analysis_results = context.get('current_analysis', {}).get('audio_analysis_results')
        engagement_summary = context.get('summaries', {}).get('engagement_summary')
        layout_detection_results = context.get('current_analysis', {}).get('layout_detection_results')
        speaker_tracking_results = context.get('current_analysis', {}).get('speaker_tracking_results')
        multimodal_analysis_results = context.get('current_analysis', {}).get('multimodal_analysis_results')
        identified_hooks = context.get('identified_hooks')

        if not full_transcription or not storyboard_data:
            self.log_error("Skipping content director: transcription or storyboard_data not available.")
            set_stage_status('content_director', 'skipped', {'reason': 'Missing transcription or storyboard data'})
            context["current_analysis"]["content_alignment_data"] = "N/A"
            context["current_analysis"]["llm_cut_decisions"] = []
            return context

        print("🔄🎬 Starting Content Director orchestration (alignment & video direction)...")
        set_stage_status('content_director', 'running')

        simplified_storyboard = [{
            "timestamp": round(sb['timestamp'], 1),
            "description": sb['description'],
            "content_type": sb.get('content_type'),
            "hook_potential": sb.get('hook_potential')
        } for sb in storyboard_data if storyboard_data is not None]

        audio_rhythm_summary = {
            "beats_count": len(audio_analysis_results.get('audio_rhythm_data', {}).get('beats', [])),
            "tempo": audio_analysis_results.get('audio_rhythm_data', {}).get('tempo')
        } if audio_analysis_results and audio_analysis_results.get('audio_rhythm_data') else {}

        engagement_summary_data = engagement_summary if engagement_summary is not None else {}

        layout_summary = {
            "num_segments": len(layout_detection_results) if layout_detection_results else 0,
            "common_layouts": list(set([s['layout_type'] for s in layout_detection_results])) if layout_detection_results else []
        }
        speaker_summary = {
            "num_speakers": len(speaker_tracking_results.get('speaker_to_face_map', {})) if speaker_tracking_results else 0,
            "num_transitions": len(speaker_tracking_results.get('speaker_transitions', [])) if speaker_tracking_results else 0
        }

        qwen_summary = {
            "num_frames_analyzed": len(multimodal_analysis_results.get('qwen_features', [])) if multimodal_analysis_results else 0,
            "sample_qwen_descriptions": [f['scene_description'] for f in multimodal_analysis_results.get('qwen_features', [])[:3]] if multimodal_analysis_results else []
        }

        alignment_context_data = {
            "transcript_summary": context.get('summaries', {}).get('full_transcription_summary'),
            "storyboard_summary": simplified_storyboard,
            "audio_rhythm_summary": audio_rhythm_summary,
            "engagement_summary": engagement_summary_data,
            "layout_summary": layout_summary,
            "speaker_summary": speaker_summary,
            "qwen_vision_summary": qwen_summary
        }

        # Explicitly define the expected JSON schema in the system prompt
        alignment_system_prompt = """
        You are an expert in video content analysis and synchronization. Your task is to create a comprehensive content-visual alignment map.
        You MUST respond with ONLY a valid JSON object with a single key "alignments" which is a list of alignment objects.
        Each alignment object MUST strictly adhere to the following schema:

        {
            "start_time": float,
            "end_time": float,
            "speaker": "string",
            "transcript_segment": "string",
            "visual_description": "string",
            "scene_change": boolean,
            "topic_shift": boolean (optional),
            "engagement_impact": "string",
            "rhythm_correlation": "string" (optional)
        }

        Example:
        {
            "alignments": [
                {
                    "start_time": 10.5,
                    "end_time": 15.2,
                    "speaker": "Speaker A",
                    "transcript_segment": "This is a key point about the new feature.",
                    "visual_description": "Close-up of product interface.",
                    "scene_change": true,
                    "topic_shift": false,
                    "engagement_impact": "High",
                    "rhythm_correlation": "On beat"
                },
                {
                    "start_time": 16.0,
                    "end_time": 20.0,
                    "speaker": "Speaker B",
                    "transcript_segment": "Let's look at the data.",
                    "visual_description": "Graph showing sales trends.",
                    "scene_change": false,
                    "topic_shift": true,
                    "engagement_impact": "Medium",
                    "rhythm_correlation": "Off beat"
                }
            ]
        }
        """

        base_alignment_user_prompt = """
        Given the following summarized video data, identify key moments where spoken content aligns with visual changes, speaker transitions, and engagement peaks.
        Provide a comprehensive content-visual alignment map as a JSON object with a single key "alignments" which is a list of alignment objects.
        """

        alignment_prompt = build_adaptive_prompt(
            base_prompt=base_alignment_user_prompt,
            context_data=alignment_context_data,
            max_tokens=self.config.get('resources.max_tokens_per_llm_call', 40000),
            model_name=self.config.get('llm.current_active_llm_model', 'gemma'),
            prioritize_keys=['transcript_summary', 'storyboard_summary', 'engagement_summary'],
            compress_strategies={
                'transcript_summary': {'strategy': 'summarize'},
                'storyboard_summary': {'strategy': 'first_n', 'n': 5},
                'qwen_vision_summary': {'strategy': 'first_n', 'n': 3}
            }
        )
        
        print("🧠 Performing content alignment with LLM (Stage 1/2)...")
        try:
            alignment_results_obj = llm_interaction.robust_llm_json_extraction(
                system_prompt=alignment_system_prompt.strip(), # Use the detailed system prompt
                user_prompt=alignment_prompt.strip(),
                output_schema=ContentAlignments,
                max_attempts=3
            )
            context["current_analysis"]["content_alignment_data"] = alignment_results_obj.model_dump()
            print("✅ Content alignment by LLM complete.")
        except Exception as e:
            self.log_error(f"Failed to perform content alignment with LLM: {e}. Setting to N/A.")
            set_stage_status('content_director', 'failed', {'reason': f"Content alignment failed: {e}"})
            context["current_analysis"]["content_alignment_data"] = {"alignments": []}
            # Continue to next stage even if alignment fails, but log it

        max_agent_retries = self.config.get('llm_agent_max_retries', 3)
        
        current_content_alignment_data = context['current_analysis']['content_alignment_data']

        initial_cut_decisions_obj = None
        for attempt in range(max_agent_retries):
            try:
                self.log_info(f"Attempting LLM Video Director initial pass (Agent Retry {attempt + 1}/{max_agent_retries})...")
                
                initial_cut_context_data = {
                    "storyboard_data": simplified_storyboard,
                    "content_alignment_data": current_content_alignment_data,
                    "identified_hooks": identified_hooks
                }

                base_initial_cut_prompt = """
                You are an expert video director. Your task is to analyze core video data
                and make intelligent *initial* cut decisions to create a coherent and engaging short clip.
                These decisions will be refined in a subsequent step.

                Based on this data, provide a detailed plan for *initial* video cuts. Focus on:
                1. Identifying optimal start and end points for clips, considering narrative coherence and main content flow.
                2. Highlighting significant events and key narrative beats.
                3. Suggesting preliminary cut decisions that capture the essence of the video.
                4. For each cut decision, you MUST provide a 'reason' explaining why this segment was chosen (e.g., "high engagement moment", "speaker transition", "scene change").

                Provide your response as a JSON object with a single key "cut_decisions" which is a list of cut decision objects.
                Ensure 'key_elements' is a JSON array of strings.

                Example:
                {
                    "cut_decisions": [
                        {
                            "start_time": 10.0,
                            "end_time": 25.0,
                            "reason": "Introduction of main topic and high speaker energy.",
                            "key_elements": ["topic introduction", "speaker energy spike"],
                            "viral_potential_score": 7
                        }
                    ]
                }
                """

                initial_cut_prompt = build_adaptive_prompt(
                    base_prompt=base_initial_cut_prompt,
                    context_data=initial_cut_context_data,
                    max_tokens=self.config.get('resources.max_tokens_per_llm_call', 40000),
                    model_name=self.config.get('llm.current_active_llm_model', 'gemma'),
                    prioritize_keys=['content_alignment_data', 'storyboard_data', 'identified_hooks'],
                    compress_strategies={
                        'storyboard_data': {'strategy': 'first_n', 'n': 5},
                        'identified_hooks': {'strategy': 'first_n', 'n': 3}
                    }
                )
                print("🧠 Sending core data to LLM for initial video direction (Stage 2/2 - Pass 1)...")
                initial_cut_decisions_obj = llm_interaction.robust_llm_json_extraction(
                    system_prompt="You are an expert video director, making intelligent initial cut decisions.",
                    user_prompt=initial_cut_prompt.strip(),
                    output_schema=CutDecisions,
                    max_attempts=1
                )
                break
            except Exception as e:
                self.log_error(f"Error during LLM Video Director initial pass: {e}")
                if attempt < max_agent_retries - 1:
                    self.log_info(f"Retrying initial pass in {self.config.get('llm_retry_delay', 2)} seconds...")
                    time.sleep(self.config.get('llm_retry_delay', 2))
                else:
                    self.log_error(f"Max agent retries ({max_agent_retries}) exhausted for initial pass.")
                    set_stage_status('content_director', 'failed', {'reason': f"Initial cut decisions failed: {e}"})
                    context["current_analysis"]["llm_cut_decisions"] = []
                    return context
        
        if not initial_cut_decisions_obj:
            self.log_error("Initial LLM pass failed to generate cut decisions.")
            set_stage_status('content_director', 'failed', {'reason': 'Initial LLM pass failed'})
            context["current_analysis"]["llm_cut_decisions"] = []
            return context

        initial_cut_decisions_dicts = [decision.model_dump() for decision in initial_cut_decisions_obj.cut_decisions]

        final_cut_decisions_obj = None
        for attempt in range(max_agent_retries):
            try:
                self.log_info(f"Attempting LLM Video Director refinement pass (Agent Retry {attempt + 1}/{max_agent_retries})...")
                
                refinement_context_data = {
                    "initial_cut_decisions": initial_cut_decisions_dicts,
                    "audio_rhythm_summary": audio_rhythm_summary,
                    "engagement_summary": engagement_summary_data,
                    "layout_summary": layout_summary,
                    "speaker_summary": speaker_summary
                }

                base_refinement_prompt = """
                You are an expert video director. Your task is to refine a set of preliminary video cut decisions
                based on additional detailed analysis data.

                Based on these preliminary decisions and the new data, refine the video cuts. Focus on:
                1. Adjusting start and end points for clips to align with audio rhythm, engagement peaks, and speaker transitions.
                2. Incorporating insights from layout detection to ensure visually dynamic cuts.
                3. Maximizing viral potential and viewer retention by emphasizing key moments identified in the new data.
                4. You may add new cut decisions or modify existing ones as needed.
                5. For each cut decision, you MUST provide a 'reason' explaining why this segment was chosen (e.g., "high engagement moment", "speaker transition", "scene change").

                Provide your response as a JSON object with a single key "cut_decisions" which is a list of refined cut decision objects.
                Ensure the 'viral_potential_score' is updated and 'key_elements' is a JSON array of strings based on the new information.
                """

                refinement_prompt = build_adaptive_prompt(
                    base_prompt=base_refinement_prompt,
                    context_data=refinement_context_data,
                    max_tokens=self.config.get('resources.max_tokens_per_llm_call', 40000),
                    model_name=self.config.get('llm.current_active_llm_model', 'gemma'),
                    prioritize_keys=['initial_cut_decisions', 'engagement_summary', 'audio_rhythm_summary'],
                    compress_strategies={
                        'initial_cut_decisions': {'strategy': 'first_n', 'n': 5}
                    }
                )
                print("🧠 Sending preliminary cuts and auxiliary data to LLM for refinement (Stage 2/2 - Pass 2)...")
                final_cut_decisions_obj = llm_interaction.robust_llm_json_extraction(
                    system_prompt="You are an expert video director, refining preliminary cut decisions.",
                    user_prompt=refinement_prompt.strip(),
                    output_schema=CutDecisions,
                    max_attempts=1
                )
                break
            except Exception as e:
                self.log_error(f"Error during LLM Video Director refinement pass: {e}")
                if attempt < max_agent_retries - 1:
                    self.log_info(f"Retrying refinement pass in {self.config.get('llm_retry_delay', 2)} seconds...")
                    time.sleep(self.config.get('llm_retry_delay', 2))
                else:
                    self.log_error(f"Max agent retries ({max_agent_retries}) exhausted for refinement pass.")
                    set_stage_status('content_director', 'failed', {'reason': f"Refinement cut decisions failed: {e}"})
                    context["current_analysis"]["llm_cut_decisions"] = []
                    return context

        if not final_cut_decisions_obj:
            self.log_error("Refinement LLM pass failed to generate cut decisions.")
            set_stage_status('content_director', 'failed', {'reason': 'Refinement LLM pass failed'})
            context["current_analysis"]["llm_cut_decisions"] = []
            return context

        final_cut_decisions_dicts = [decision.model_dump() for decision in final_cut_decisions_obj.cut_decisions]

        context['current_analysis']['llm_cut_decisions'] = final_cut_decisions_dicts
        print(f"✅ Content Director orchestration complete. Generated {len(final_cut_decisions_dicts)} refined cut decisions.")
        set_stage_status('content_director_complete', 'complete', {'num_cut_decisions': len(final_cut_decisions_dicts)})
        return context
