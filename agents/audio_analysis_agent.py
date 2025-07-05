import os
import torch
from transformers.pipelines import pipeline # Corrected import
from agents.base_agent import Agent
from core.state_manager import set_stage_status
from core.gpu_manager import gpu_manager
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

class AudioAnalysisAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("AudioAnalysisAgent")
        self.config = config
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
                return False
        return True # Model already loaded

    def _load_models(self):
        # Sentiment Analyzer
        self._load_model(
            "sentiment_analyzer",
            lambda: pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1),
            priority=2,
            error_message="Failed to load sentiment analysis model"
        )

        # Speaker Diarization
        self._load_model(
            "speaker_diarization_pipeline",
            lambda token: PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token),
            priority=3,
            error_message="Failed to load speaker diarization model",
            hf_token_key='huggingface_tokens.pyannote_audio'
        )

        # Theme Model
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
                speakers = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speakers.append({'speaker': speaker, 'start': turn.start, 'end': turn.end})
                audio_analysis_results['speaker_diarization'] = speakers
                self.log_info(f"Speaker diarization complete. Detected {len(set(s['speaker'] for s in speakers))} speakers.")
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
        
        pitch = np.array([]) # Initialize pitch as an empty numpy array

        # Check if magnitudes has data before attempting pitch calculation
        if magnitudes.size > 0 and magnitudes.shape[1] > 0:
            # Optimized pitch calculation using numpy argmax
            max_magnitude_indices = np.argmax(magnitudes, axis=0)
            # Use np.take_along_axis for potentially better Pylance inference
            # The indices need to be broadcastable to the shape of pitches along axis 0
            pitch = np.take_along_axis(pitches, max_magnitude_indices[np.newaxis, :], axis=0).squeeze()
        else:
            self.log_warning("No valid magnitudes detected for pitch tracking. Skipping vocal emphasis calculation.")
        
        audio_analysis_results['speech_energy'] = {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms))
        }
        # Ensure pitch > 0 to avoid division by zero or issues with silent parts
        valid_pitches = pitch[pitch > 0]
        if valid_pitches.size > 0:
            audio_analysis_results['vocal_emphasis'] = {
                'pitch_mean': float(np.mean(valid_pitches)),
                'pitch_std': float(np.std(valid_pitches))
            }
        else:
            audio_analysis_results['vocal_emphasis'] = {
                'pitch_mean': 0.0, # Or np.nan, depending on desired behavior for silent audio
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
                    # Added n_init='auto' for KMeans to suppress future warnings
                    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(embeddings)
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

    def execute(self, context):
        print("🎵 Starting audio analysis.")
        audio_path = context.get('audio_path')
        transcription = context.get('transcription')

        if not audio_path or not os.path.exists(audio_path):
            print("❌ Audio path not found in context.")
            return context
        
        if not transcription:
            print("❌ Transcription not found in context.")
            return context

        self.log_info(f"Starting advanced audio analysis for {audio_path}")
        set_stage_status('audio_analysis', 'running')

        self._load_models() # Ensure models are loaded

        try:
            # Consider specifying sr=None if original sample rate is preferred, or a fixed sr if resampling is desired.
            # For now, keeping default behavior.
            y, sr = librosa.load(audio_path)
            audio_analysis_results = {}

            self._perform_speaker_diarization(audio_path, audio_analysis_results)
            self._perform_sentiment_analysis(transcription, audio_analysis_results)
            self._analyze_speech_energy_vocal_emphasis(y, sr, audio_analysis_results)
            self._identify_audio_themes(transcription, audio_analysis_results)

            context['audio_analysis_results'] = audio_analysis_results
            self.log_info("Advanced audio analysis complete.")
            set_stage_status('audio_analysis', 'complete', audio_analysis_results)
            return context

        except Exception as e:
            self.log_error(f"Error during advanced audio analysis: {e}")
            set_stage_status('audio_analysis', 'failed', {'reason': str(e)})
            return context
