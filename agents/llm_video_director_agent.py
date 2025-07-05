from agents.base_agent import Agent
from core.state_manager import set_stage_status
from llm import llm_interaction
from pydantic import BaseModel, Field
from typing import List, Any, Dict

class CutDecision(BaseModel):
    start_time: float = Field(description="The start time of the clip segment in seconds.")
    end_time: float = Field(description="The end time of the clip segment in seconds.")
    reason: str = Field(description="A brief explanation for the cut decision (e.g., \"high engagement moment\", \"speaker transition\", \"scene change\").")
    key_elements: List[str] = Field(description="A list of key visual or audio elements present in this segment.")
    viral_potential_score: int = Field(description="An estimated score for viral potential (0-10).", ge=0, le=10)

class CutDecisions(BaseModel):
    cut_decisions: List[CutDecision]

class LLMVideoDirectorAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("LLMVideoDirectorAgent")
        self.config = config
        self.state_manager = state_manager

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Gather all relevant analysis results from the context
        transcription = context.get('transcription')
        storyboard_data = context.get('storyboard_data')
        audio_rhythm_data = context.get('audio_rhythm_data')
        engagement_analysis_results = context.get('engagement_analysis_results')
        layout_detection_results = context.get('layout_detection_results')
        speaker_tracking_results = context.get('speaker_tracking_results')
        qwen_vision_analysis_results = context.get('qwen_vision_analysis_results')
        content_alignment_data = context.get('content_alignment_data')
        identified_hooks = context.get('identified_hooks')

        if not transcription or not qwen_vision_analysis_results or not content_alignment_data:
            self.log_error("Missing essential analysis results for LLM Video Director. Skipping.")
            set_stage_status('llm_video_director', 'failed', {'reason': 'Missing essential data'})
            return context

        print("🎬 Starting LLM Video Director orchestration...")
        set_stage_status('llm_video_director', 'running')

        try:
            # Prepare a comprehensive prompt for the LLM
            llm_prompt = f"""
            You are an expert video director. Your task is to analyze various data streams from a video
            and make intelligent cut decisions to create a coherent and engaging short clip.

            Here is the available analysis data:

            Transcription:
            {transcription}

            Storyboard Data:
            {storyboard_data}

            Audio Rhythm Data:
            {audio_rhythm_data}

            Engagement Analysis Results:
            {engagement_analysis_results}

            Layout Detection Results:
            {layout_detection_results}

            Speaker Tracking Results:
            {speaker_tracking_results}

            Qwen-VL Vision Analysis Results:
            {qwen_vision_analysis_results}

            Content Alignment Data:
            {content_alignment_data}

            Identified Hooks:
            {identified_hooks}

            Based on this data, provide a detailed plan for video cuts. Focus on:
            1. Identifying optimal start and end points for clips, considering narrative coherence, engagement, and visual flow.
            2. Highlighting significant events and object interactions.
            3. Ensuring smooth transitions and emphasizing key moments.
            4. Suggesting cut decisions that maximize viral potential and viewer retention.

            Provide your response as a JSON object with a single key "cut_decisions" which is a list of cut decision objects.
            """

            print("🧠 Sending comprehensive data to LLM for video direction...")
            
            llm_cut_decisions_obj = llm_interaction.robust_llm_json_extraction(
                system_prompt="You are an expert video director, making intelligent cut decisions.",
                user_prompt=llm_prompt.strip(),
                output_schema=CutDecisions
            )
            
            llm_cut_decisions = llm_cut_decisions_obj.cut_decisions

            context['llm_cut_decisions'] = llm_cut_decisions
            print(f"✅ LLM Video Director orchestration complete. Generated {len(llm_cut_decisions)} cut decisions.")
            set_stage_status('llm_video_director_complete', 'complete', {'num_cut_decisions': len(llm_cut_decisions)})
            return context

        except Exception as e:
            self.log_error(f"Error during LLM Video Director orchestration: {e}")
            set_stage_status('llm_video_director', 'failed', {'reason': str(e)})
            return context