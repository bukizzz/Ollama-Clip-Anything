from agents.base_agent import Agent
from core.state_manager import set_stage_status
from core.temp_manager import get_temp_path
from llm import llm_interaction
from audio import voice_cloning

class IntroNarrationAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("IntroNarrationAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.intro_narration_config = agent_config.get('intro_narration')

    def execute(self, context):
        clips = context.get('clips')
        audio_analysis_results = context.get('audio_analysis_results', {})
        
        if not clips:
            self.log_error("Selected clips not found in context. Cannot generate intro narration.")
            set_stage_status('intro_narration_generated', 'failed', {'reason': 'Missing selected clips info'})
            return context

        print("🎤 Starting intro narration generation...")
        set_stage_status('intro_narration_generated', 'running')

        try:
            # 1. Analyze selected clips to identify key themes and hooks
            clips_summary = " ".join([clip.get('text', '') for clip in clips])
            main_theme = audio_analysis_results.get('audio_themes', {}).get('theme_0', 'general')
            sentiment = audio_analysis_results.get('sentiment', {'label': 'NEUTRAL'})['label']

            # 2. Generate compelling intro narration using LLM
            llm_prompt = f"""
            Based on the following video summary, main theme, and sentiment, generate a compelling, non-clickbait, and honest intro narration.
            The narration should be concise (max {self.intro_narration_config.get('duration_limit_seconds', 5)} seconds), engaging, and match the content's tone.

            - Video Summary: {clips_summary[:500]}...
            - Main Theme: {main_theme}
            - Overall Sentiment: {sentiment}

            Generate only the narration text.
            """
            
            print("🧠 Generating intro narration text with LLM...")
            narration_text = llm_interaction.llm_pass(
                self.config.get('llm_model'),
                [
                    {"role": "system", "content": "You are a creative writer specializing in concise and honest video intros."},
                    {"role": "user", "content": llm_prompt.strip()}
                ]
            )
            print(f"🤖 Generated narration text: {narration_text}")
            llm_interaction.cleanup() # Clear VRAM after LLM text generation

            if not narration_text:
                self.log_error("LLM failed to generate narration text.")
                set_stage_status('intro_narration_generated', 'failed', {'reason': 'LLM failed to generate text'})
                return context

            # 3. Integrate with voice cloning for speaker consistency
            narration_audio_path = get_temp_path("intro_narration.wav")
            
            if self.intro_narration_config.get('voice_cloning_enabled', False):
                print("🗣️ Generating intro narration audio using voice cloning...")
                
                # Extract dominant speaker's audio for voice cloning
                speaker_diarization = audio_analysis_results.get('speaker_diarization', [])
                dominant_speaker = max(set(s['speaker'] for s in speaker_diarization), key=speaker_diarization.count) if speaker_diarization else None
                
                speaker_wav_path = None
                if dominant_speaker:
                    # This part is conceptual - would need to extract the audio segment for the dominant speaker
                    # For now, we'll pass a placeholder path to the voice cloning function
                    speaker_wav_path = context.get('audio_path') # Use the full audio as a proxy
                    print(f"🗣️ Using dominant speaker '{dominant_speaker}' for voice cloning.")

                success = voice_cloning.generate_voice_from_text(narration_text, narration_audio_path, speaker_wav_path=speaker_wav_path)
                if not success:
                    self.log_error("Voice cloning failed for intro narration.")
                    set_stage_status('intro_narration_generated', 'failed', {'reason': 'Voice cloning failed'})
                    return context
            else:
                print("🔇 Voice cloning disabled. Skipping audio generation for intro narration.")
                narration_audio_path = None

            context['intro_narration'] = {
                'text': narration_text,
                'audio_path': narration_audio_path
            }
            print("✅ Intro narration generation complete.")
            set_stage_status('intro_narration_generated', 'complete', {'text': narration_text, 'audio_path': narration_audio_path})
            return context

        except Exception as e:
            self.log_error(f"Error during intro narration generation: {e}")
            set_stage_status('intro_narration_generated', 'failed', {'reason': str(e)})
            return context
