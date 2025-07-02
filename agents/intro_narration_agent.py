import os
from core.base_agent import BaseAgent
from core.state_manager import set_stage_status, get_stage_status
from core.temp_manager import get_temp_path
from llm import llm_interaction
from audio import voice_cloning
from core.config import INTRO_NARRATION_CONFIG

class IntroNarrationAgent(BaseAgent):
    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)
        self.intro_narration_config = INTRO_NARRATION_CONFIG

    def run(self, context):
        selected_clips_info = context.get('selected_clips_info')
        if not selected_clips_info:
            self.log_error("Selected clips information not found in context. Cannot generate intro narration.")
            set_stage_status('intro_narration_generated', 'failed', {'reason': 'Missing selected clips info'})
            return False

        self.log_info("Starting intro narration generation...")
        set_stage_status('intro_narration_generated', 'running')

        try:
            # 1. Analyze selected clips to identify key themes and hooks
            # For simplicity, we'll concatenate transcriptions of selected clips.
            # In a more advanced scenario, this would involve deeper analysis of clip content.
            combined_transcript = " ".join([clip['transcription'] for clip in selected_clips_info if 'transcription' in clip])

            if not combined_transcript:
                self.log_warning("No transcript available for selected clips. Skipping intro narration generation.")
                set_stage_status('intro_narration_generated', 'skipped', {'reason': 'No transcript for selected clips'})
                return True

            # 2. Generate compelling intro narration using LLM
            llm_prompt = f"""
            Based on the following content, generate a compelling, non-clickbait, and honest intro narration.
            The narration should be concise, maximum {self.intro_narration_config.get('duration_limit_seconds', 5)} seconds long,
            and match the overall tone and mood of the content.

            Content Summary (from selected clips):
            {combined_transcript}

            Generate only the narration text, without any additional commentary or formatting.
            """
            
            self.log_info("Generating intro narration text with LLM...")
            narration_text = llm_interaction.llm_pass(
                llm_interaction.LLM_MODEL,
                [
                    {"role": "system", "content": "You are a creative assistant that generates concise and honest video intro narrations."},
                    {"role": "user", "content": llm_prompt.strip()}
                ]
            )
            self.log_info(f"Generated narration text: {narration_text}")

            if not narration_text:
                self.log_error("LLM failed to generate narration text.")
                set_stage_status('intro_narration_generated', 'failed', {'reason': 'LLM failed to generate text'})
                return False

            # 3. Integrate with voice cloning for speaker consistency
            # Assuming voice_cloning.generate_voice_from_text exists and takes text and an output path
            narration_audio_path = get_temp_path("intro_narration.wav")
            
            if self.intro_narration_config.get('voice_cloning_enabled', False):
                self.log_info("Generating intro narration audio using voice cloning...")
                # In a real scenario, you'd pass a speaker reference audio to voice_cloning.generate_voice_from_text
                # For now, we'll assume it can generate a default voice or use a placeholder.
                success = voice_cloning.generate_voice_from_text(narration_text, narration_audio_path)
                if not success:
                    self.log_error("Voice cloning failed for intro narration.")
                    set_stage_status('intro_narration_generated', 'failed', {'reason': 'Voice cloning failed'})
                    return False
            else:
                self.log_info("Voice cloning disabled. Skipping audio generation for intro narration.")
                # If voice cloning is disabled, you might want to use a default TTS or skip audio generation
                # For now, we'll mark it as successful if voice cloning is not enabled.
                narration_audio_path = None # No audio generated

            context['intro_narration'] = {
                'text': narration_text,
                'audio_path': narration_audio_path
            }
            self.log_info("Intro narration generation complete.")
            set_stage_status('intro_narration_generated', 'complete', {'text': narration_text, 'audio_path': narration_audio_path})
            return True

        except Exception as e:
            self.log_error(f"Error during intro narration generation: {e}")
            set_stage_status('intro_narration_generated', 'failed', {'reason': str(e)})
            return False
