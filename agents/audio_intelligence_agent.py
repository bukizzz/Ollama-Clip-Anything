from agents.base_agent import Agent
from typing import Dict, Any
from audio import audio_processing
from core.temp_manager import get_temp_path
from core.cache_manager import cache_manager
from llm.llm_interaction import summarize_transcript_with_llm
from core.state_manager import set_stage_status # Import the function

import os
import torch
from transformers.pipelines import pipeline
from core.gpu_manager import gpu_manager
from core.resource_manager import resource_manager
from pyannote.audio import Pipeline as PyannotePipeline
import librosa
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Define constants for better readability and maintainability
MAX_TOKEN_LENGTH = 510
SENTIMENT_POSITIVE_THRESHOLD = 0.2
SENTIMENT_NEGATIVE_THRESHOLD = -0.2
DEFAULT_NUM_THEME_CLUSTERS = 5

class AudioIntelligenceAgent(Agent):
    """Agent responsible for transcribing video audio, performing rhythm analysis, and advanced audio analysis."""

    def __init__(self, agent_config, state_manager):
        super().__init__("AudioIntelligenceAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.sentiment_analyzer = None
        self.speaker_diarization_pipeline = None
        self.theme_model = None

    def _load_model(self, model_name, model_loader_func, priority, error_message, hf_token_key=None):
        """Helper to load a single model."""
        if getattr(self, model_name) is None:
            self.log_info(f"Loading {model_name.replace('_', ' ')} model...")
            try:
                if hf_token_key:
                    hf_token = self.config.get(hf_token_key)
                    if not hf_token:
                        self.log_error(f"Hugging Face token for {hf_token_key} not found in config.yaml. Please add it.")
                        return False
                    model_instance = model_loader_func(hf_token)
                else:
                    model_instance = model_loader_func()

                if model_instance is None:
                    self.log_error(f"{model_name.replace('_', ' ')} failed to load.")
                    return False

                setattr(self, model_name, model_instance)
                if torch.cuda.is_available() and hasattr(model_instance, 'to'):
                    model_instance.to(torch.device("cuda"))
                gpu_manager.load_model(model_name, model_instance, priority=priority)
                self.log_info(f"{model_name.replace('_', ' ')} model loaded.")
                return True
            except Exception as e:
                self.log_error(f"{error_message}: {e}")
                resource_manager.unload_all_models()
                return False
        return True

    def _load_models(self):
        self._load_model(
            "sentiment_analyzer",
            lambda: pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1),
            priority=2,
            error_message="Failed to load sentiment analysis model"
        )

        self._load_model(
            "speaker_diarization_pipeline",
            lambda token: PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token),
            priority=3,
            error_message="Failed to load speaker diarization model",
            hf_token_key='huggingface_tokens.pyannote_audio'
        )

        self._load_model(
            "theme_model",
            lambda: SentenceTransformer('all-MiniLM-L6-v2'),
            priority=1,
            error_message="Failed to load theme identification model"
        )

    def _perform_speaker_diarization(self, audio_path, audio_analysis_results):
        if self.speaker_diarization_pipeline:
            try:
                diarization = self.speaker_diarization_pipeline(audio_path)
                self.log_info(f"DEBUG: Type of diarization object: {type(diarization)}")
                self.log_info(f"DEBUG: Has itertracks attribute: {hasattr(diarization, 'itertracks')}")

                if diarization and hasattr(diarization, 'itertracks'):
                    speakers = []
                    # Ensure itertracks returns an iterable
                    itertracks_result = diarization.itertracks(yield_label=True)
                    if itertracks_result is None:
                        self.log_warning("diarization.itertracks returned None. Skipping diarization.")
                        return
                    
                    for turn, _, speaker in itertracks_result:
                        speakers.append({'speaker': speaker, 'start': turn.start, 'end': turn.end})
                    audio_analysis_results['speaker_diarization'] = speakers
                    self.log_info(f"Speaker diarization complete. Detected {len(set(s['speaker'] for s in speakers))} speakers.")
                else:
                    self.log_warning("Speaker diarization pipeline returned no valid results or an unexpected object. Skipping diarization.")
            except Exception as e:
                self.log_error(f"Failed to run speaker diarization model: {e}")
        else:
            self.log_warning("Speaker diarization model not loaded. Skipping diarization.")

    def _perform_sentiment_analysis(self, transcription, audio_analysis_results):
        if self.sentiment_analyzer:
            try:
                full_transcript = " ".join([seg['text'] for seg in transcription])
                if full_transcript:
                    tokenizer = self.sentiment_analyzer.tokenizer
                    tokenized_input = tokenizer(full_transcript, add_special_tokens=False, truncation=True)
                    input_ids = tokenized_input['input_ids']
                    chunks_input_ids = [input_ids[i:i + MAX_TOKEN_LENGTH] for i in range(0, len(input_ids), MAX_TOKEN_LENGTH)]
                    
                    sentiments = []
                    for chunk_input_ids in chunks_input_ids:
                        chunk_text = tokenizer.decode(chunk_input_ids, skip_special_tokens=True)
                        try:
                            sentiment_result = self.sentiment_analyzer(chunk_text)
                            sentiments.append(sentiment_result[0]['score'] if sentiment_result[0]['label'] == 'POSITIVE' else -sentiment_result[0]['score'])
                        except Exception as e:
                            self.log_warning(f"Sentiment analysis failed for chunk: {e}")
                            sentiments.append(0)
                    
                    if sentiments:
                        avg_sentiment_score = sum(sentiments) / len(sentiments)
                        if avg_sentiment_score > SENTIMENT_POSITIVE_THRESHOLD:
                            label = 'POSITIVE'
                        elif avg_sentiment_score < SENTIMENT_NEGATIVE_THRESHOLD:
                            label = 'NEGATIVE'
                        else:
                            label = 'NEUTRAL'
                        audio_analysis_results['sentiment'] = {'label': label, 'score': avg_sentiment_score}
                        self.log_info(f"Audio sentiment: {label} (score: {avg_sentiment_score:.2f})")
            except Exception as e:
                self.log_error(f"Failed to run sentiment analysis model: {e}")
        else:
            self.log_warning("Sentiment analysis model not loaded. Skipping sentiment analysis.")

    def _analyze_speech_energy_vocal_emphasis(self, y, sr, audio_analysis_results):
        rms = librosa.feature.rms(y=y)[0]
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        pitch = np.array([])

        if magnitudes.size > 0 and magnitudes.shape[1] > 0:
            max_magnitude_indices = np.argmax(magnitudes, axis=0)
            pitch = np.take_along_axis(pitches, max_magnitude_indices[np.newaxis, :], axis=0).squeeze()
        else:
            self.log_warning("No valid magnitudes detected for pitch tracking. Skipping vocal emphasis calculation.")
        
        audio_analysis_results['speech_energy'] = {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms))
        }
        valid_pitches = pitch[pitch > 0]
        if valid_pitches.size > 0:
            audio_analysis_results['vocal_emphasis'] = {
                'pitch_mean': float(np.mean(valid_pitches)),
                'pitch_std': float(np.std(valid_pitches))
            }
        else:
            audio_analysis_results['vocal_emphasis'] = {
                'pitch_mean': 0.0,
                'pitch_std': 0.0
            }
            self.log_warning("No valid pitches detected for vocal emphasis analysis.")

        self.log_info("Speech energy and vocal emphasis analysis complete.")

    def _identify_audio_themes(self, transcription, audio_analysis_results):
        if self.theme_model:
            try:
                sentences = [seg['text'] for seg in transcription]
                if not sentences:
                    self.log_warning("No sentences found for theme identification. Skipping theme analysis.")
                    return

                embeddings = self.theme_model.encode(sentences)
                num_clusters = min(DEFAULT_NUM_THEME_CLUSTERS, len(sentences))
                
                if num_clusters > 0:
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(embeddings) # Changed n_init to 10
                    themes = {}
                    for i, label in enumerate(kmeans.labels_):
                        if label not in themes:
                            themes[label] = []
                        themes[label].append(sentences[i])
                    audio_analysis_results['audio_themes'] = {f'theme_{k}': " ".join(v) for k, v in themes.items()}
                    self.log_info(f"Identified {num_clusters} audio themes.")
                else:
                    self.log_warning("Not enough sentences to form clusters for theme identification. Skipping theme analysis.")
            except Exception as e:
                self.log_error(f"Failed to run theme identification model: {e}")
        else:
            self.log_warning("Theme identification model not loaded. Skipping theme analysis.")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_video_path = context.get("processed_video_path")
        
        context.setdefault('archived_data', {})
        context.setdefault('summaries', {})
        context.setdefault('current_analysis', {})

        transcription = context['archived_data'].get("full_transcription")

        print("\n🎤 \u001b[94mStarting Audio Intelligence Analysis...\u001b[0m")
        
        if processed_video_path is None:
            raise RuntimeError("Processed video path is missing from context. Cannot perform audio intelligence.")

        cache_key = f"audio_intelligence_{os.path.basename(processed_video_path)}"
        cached_results = cache_manager.get(cache_key, level="disk")

        if cached_results:
            print("⏩ Skipping audio intelligence. Loaded from cache.")
            context.update(cached_results)
            set_stage_status('audio_analysis_complete', 'complete', {'loaded_from_cache': True})
            return context

        audio_path = get_temp_path("temp_audio_normalized.wav")
        if not context.get("audio_path"):
            audio_processing.extract_audio(processed_video_path, audio_path)
            context["audio_path"] = audio_path

        if transcription is None:
            transcription = audio_processing.transcribe_video(processed_video_path)
            if not transcription:
                raise RuntimeError("Transcription failed. Video may have no audio.")
            context["archived_data"]["full_transcription"] = transcription
            
            print("🧠 \u001b[94mSummarizing transcript with LLM...\u001b[0m")
            transcript_summary = summarize_transcript_with_llm(transcription)
            context["summaries"]["full_transcription_summary"] = transcript_summary.model_dump()
            
            if "transcription" in context:
                del context["transcription"]
        else:
            print("⏩ Skipping transcription. Loaded from state.")
            
        print(f"✅ Transcription complete: {len(transcription)} segments found.")

        self.log_info(f"Starting advanced audio analysis for {audio_path}")
        set_stage_status('audio_analysis', 'running')

        self._load_models()

        try:
            y, sr = librosa.load(audio_path)
            audio_analysis_results = {}

            self._perform_speaker_diarization(audio_path, audio_analysis_results)
            self._perform_sentiment_analysis(transcription, audio_analysis_results)
            self._analyze_speech_energy_vocal_emphasis(y, sr, audio_analysis_results)
            self._identify_audio_themes(transcription, audio_analysis_results)

            context['current_analysis']['audio_analysis_results'] = audio_analysis_results
            self.log_info("Advanced audio analysis complete.")
            set_stage_status('audio_analysis_complete', 'complete', audio_analysis_results)
            
            cache_manager.set(cache_key, {
                'audio_path': audio_path,
                'archived_data': context.get('archived_data'),
                'summaries': context.get('summaries'),
                'current_analysis': context.get('current_analysis')
            }, level="disk")

            return context

        except Exception as e:
            self.log_error(f"Error during advanced audio analysis: {e}")
            set_stage_status('audio_analysis', 'failed', {'reason': str(e)})
            return context
