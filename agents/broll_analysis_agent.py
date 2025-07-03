from agents.base_agent import Agent
from typing import Dict, Any
import os
from llm import llm_interaction
from core.state_manager import set_stage_status
import json

class BrollAnalysisAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("BrollAnalysisAgent")
        self.config = config
        self.state_manager = state_manager
        self.b_roll_assets_dir = config.get('b_roll_assets_dir')

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        clips = context.get('clips', [])
        audio_rhythm = context.get('audio_rhythm_data', {})

        if not clips:
            self.log_info("No clips selected, skipping B-roll analysis.")
            set_stage_status('broll_analysis', 'skipped', {'reason': 'No clips'})
            return context

        self.log_info("Analyzing B-roll assets for selected clips...")
        set_stage_status('broll_analysis', 'running')

        b_roll_assets = self._scan_b_roll_directory()
        if not b_roll_assets:
            self.log_warning("No B-roll assets found.")
            context['b_roll_suggestions'] = []
            set_stage_status('broll_analysis', 'complete', {'num_suggestions': 0})
            return context

        suggestions = []
        for clip in clips:
            suggestions.extend(self._generate_suggestions_for_clip(clip, b_roll_assets, audio_rhythm))
        
        context['b_roll_suggestions'] = suggestions
        self.log_info(f"B-roll analysis complete. Generated {len(suggestions)} suggestions.")
        set_stage_status('broll_analysis_complete', 'complete', {'num_suggestions': len(suggestions)})
        return context

    def _scan_b_roll_directory(self):
        assets = []
        if not os.path.exists(self.b_roll_assets_dir):
            return assets
        for filename in os.listdir(self.b_roll_assets_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(self.b_roll_assets_dir, filename)
                # In a real scenario, you might generate a description here
                assets.append({"path": path, "description": "Generic asset"})
        return assets

    def _generate_suggestions_for_clip(self, clip, b_roll_assets, audio_rhythm):
        prompt = f"""
        For a video clip about \"{clip['text']}\", suggest relevant B-roll from the list below.
        Consider the audio rhythm for timing.
        B-roll assets: {json.dumps(b_roll_assets)}
        Audio beats at: {json.dumps(audio_rhythm.get('beat_times', []))}
        Provide a JSON list of suggestions, each with 'b_roll_path', 'start_time' (in clip), 'duration', and 'reason'.
        """
        self.log_info(f"🧠 \u001b[94mGenerating B-roll suggestions for clip: {clip['text'][:50]}...\u001b[0m")
        try:
            response = llm_interaction.llm_pass(self.config.get('llm_model'), [{"role": "user", "content": prompt}])
            return llm_interaction.extract_json_from_text(response)
        except Exception as e:
            self.log_error(f"Failed to get B-roll suggestions for clip: {e}")
            return []

    
