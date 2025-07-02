from core.base_agent import BaseAgent
from typing import Dict, Any
import os
from PIL import Image
from llm.image_analysis import describe_image
from llm import llm_interaction
from core.state_manager import set_stage_status, get_stage_status
import json

class BrollAnalysisAgent(BaseAgent):
    """Agent responsible for analyzing B-roll assets and generating descriptions and suggestions."""

    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.b_roll_assets_dir = "b_roll_assets" # Defined in CFC.md

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        clips = context.get('clips')
        transcription = context.get('transcription')
        audio_rhythm_data = context.get('audio_rhythm_data')
        llm_cut_decisions = context.get('llm_cut_decisions')

        self.log_info("Scanning and analyzing B-roll assets...")
        set_stage_status('broll_analysis', 'running')

        b_roll_assets = []
        if not os.path.exists(self.b_roll_assets_dir):
            self.log_warning(f"B-roll assets directory '{self.b_roll_assets_dir}' not found. Skipping B-roll analysis.")
            context["b_roll_data"] = []
            set_stage_status('broll_analysis', 'skipped', {'reason': 'B-roll directory not found'})
            return context

        for filename in os.listdir(self.b_roll_assets_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.b_roll_assets_dir, filename)
                try:
                    with Image.open(image_path) as img:
                        description_prompt = "Describe this image in detail, focusing on key objects, subjects, and the overall scene."
                        image_description = describe_image(img, description_prompt)
                        
                        b_roll_assets.append({
                            "path": image_path,
                            "description": image_description
                        })
                        self.log_info(f"Analyzed B-roll asset: {filename}")
                except Exception as e:
                    self.log_error(f"Error processing B-roll image {filename}: {e}")
        
        context["b_roll_assets"] = b_roll_assets
        self.log_info(f"B-roll asset analysis complete. Found {len(b_roll_assets)} assets.")

        # Generate contextual B-roll suggestions based on selected clips
        b_roll_suggestions = []
        if clips and b_roll_assets:
            self.log_info("Generating contextual B-roll suggestions for selected clips...")
            for clip in clips:
                clip_start = clip['start']
                clip_end = clip['end']
                clip_text = clip['text']

                # Find relevant transcript segments for the clip
                relevant_transcript = [seg for seg in transcription if seg['start'] >= clip_start and seg['end'] <= clip_end]
                transcript_summary = " ".join([seg['text'] for seg in relevant_transcript])

                llm_prompt = f"""
                Given the following video clip content and available B-roll assets, suggest suitable B-roll footage.
                Focus on B-roll that enhances the narrative, provides visual context, or adds emotional depth.

                Video Clip Content (transcript):
                {transcript_summary}

                Available B-roll Assets (description and path):
                {json.dumps(b_roll_assets, indent=2)}

                Audio Rhythm Data (for timing suggestions):
                {json.dumps(audio_rhythm_data, indent=2)}

                LLM Video Director Cut Decisions (for visual cuts and transitions):
                {json.dumps(llm_cut_decisions, indent=2)}

                Provide suggestions as a JSON array of objects. Each object should include:
                - "clip_start": The start time of the video clip.
                - "clip_end": The end time of the video clip.
                - "b_roll_path": The path to the suggested B-roll asset.
                - "suggestion_reason": Why this B-roll is suitable for the clip.
                - "timing_suggestion": Suggested start and end times for the B-roll within the clip, considering audio rhythm and visual cuts.
                - "transition_suggestion": Suggested transition type (e.g., "fade", "cut", "wipe").
                """
                try:
                    response = llm_interaction.llm_pass(
                        llm_interaction.LLM_MODEL,
                        [
                            {"role": "system", "content": "You are an expert B-roll selection AI."},
                            {"role": "user", "content": llm_prompt.strip()}
                        ]
                    )
                    suggestions = llm_interaction.extract_json_from_text(response)
                    if isinstance(suggestions, list):
                        b_roll_suggestions.extend(suggestions)
                except Exception as e:
                    self.log_error(f"Failed to generate B-roll suggestions for clip {clip_start}-{clip_end}: {e}")

        context["b_roll_suggestions"] = b_roll_suggestions
        self.log_info(f"B-roll suggestion complete. Generated {len(b_roll_suggestions)} suggestions.")
        set_stage_status('broll_analysis_complete', 'complete', {'num_suggestions': len(b_roll_suggestions)})
        return context

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # This method is kept for compatibility with AgentManager, but the core logic is in run()
        return self.run(context)
