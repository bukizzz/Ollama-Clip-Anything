import torch
from transformers import pipeline
from agents.base_agent import Agent
from core.state_manager import set_stage_status
from core.gpu_manager import gpu_manager
from pyannote.audio import Pipeline as PyannotePipeline
import librosa
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class AudioAnalysisAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("AudioAnalysisAgent")
        self.config = config
        self.state_manager = state_manager
        self.sentiment_analyzer = None
        self.speaker_diarization_pipeline = None
        self.theme_model = None

    def _load_models(self):
        
        if self.sentiment_analyzer is None:
            self.log_info("Loading audio sentiment analysis model...")
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1
                )
                gpu_manager.load_model("sentiment_analyzer", self.sentiment_analyzer, priority=2)
                self.log_info("Audio sentiment analysis model loaded.")
            except Exception as e:
                self.log_error(f"Failed to load sentiment analysis model: {e}")

        if self.speaker_diarization_pipeline is None:
            self.log_info("Loading speaker diarization model...")
            try:
                hf_token = self.config.get('huggingface_tokens.pyannote_audio')
                
                if not hf_token:
                    self.log_error("Hugging Face token for pyannote.audio not found in config.yaml. Please add it.")
                    return
                self.speaker_diarization_pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
                if self.speaker_diarization_pipeline is None: # Added check
                    self.log_error("Pyannote pipeline failed to load.")
                    return
                if torch.cuda.is_available():
                    self.speaker_diarization_pipeline.to(torch.device("cuda"))
                gpu_manager.load_model("speaker_diarization", self.speaker_diarization_pipeline, priority=3)
                self.log_info("Speaker diarization model loaded.")
            except Exception as e:
                self.log_error(f"Failed to load speaker diarization model: {e}")

        if self.theme_model is None:
            self.log_info("Loading theme identification model...")
            try:
                self.theme_model = SentenceTransformer('all-MiniLM-L6-v2')
                gpu_manager.load_model("theme_model", self.theme_model, priority=1)
                self.log_info("Theme identification model loaded.")
            except Exception as e:
                self.log_error(f"Failed to load theme identification model: {e}")

    def execute(self, context):
        audio_path = context.get('audio_path')
        transcription = context.get('transcription')

        if not audio_path or not transcription:
            self.log_error("Audio path or transcription not found in context.")
            set_stage_status('audio_analysis', 'failed', {'reason': 'Missing audio or transcription'})
            return context

        self.log_info(f"Starting advanced audio analysis for {audio_path}")
        set_stage_status('audio_analysis', 'running')

        self._load_models() # Ensure models are loaded

        try:
            y, sr = librosa.load(audio_path)
            audio_analysis_results = {}

            # 1. Speaker Diarization
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

            # 2. Audio Sentiment
            if self.sentiment_analyzer:
                try:
                    full_transcript = " ".join([seg['text'] for seg in transcription])
                    if full_transcript:
                        tokenizer = self.sentiment_analyzer.tokenizer
                        max_len = 510
                        tokenized_input = tokenizer(full_transcript, add_special_tokens=False, truncation=True)
                        input_ids = tokenized_input['input_ids']
                        chunks_input_ids = [input_ids[i:i + max_len] for i in range(0, len(input_ids), max_len)]
                        
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
                            if avg_sentiment_score > 0.2:
                                label = 'POSITIVE'
                            elif avg_sentiment_score < -0.2:
                                label = 'NEGATIVE'
                            else:
                                label = 'NEUTRAL'
                            audio_analysis_results['sentiment'] = {'label': label, 'score': avg_sentiment_score}
                            self.log_info(f"Audio sentiment: {label} (score: {avg_sentiment_score:.2f})")
                except Exception as e:
                    self.log_error(f"Failed to run sentiment analysis model: {e}")
            else:
                self.log_warning("Sentiment analysis model not loaded. Skipping sentiment analysis.")

            # 3. Speech Energy and Vocal Emphasis
            rms = librosa.feature.rms(y=y)[0]
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch.append(pitches[index, t])
            pitch = np.array(pitch)
            
            audio_analysis_results['speech_energy'] = {
                'rms_mean': float(np.mean(rms)),
                'rms_std': float(np.std(rms))
            }
            audio_analysis_results['vocal_emphasis'] = {
                'pitch_mean': float(np.mean(pitch[pitch > 0])),
                'pitch_std': float(np.std(pitch[pitch > 0]))
            }
            self.log_info("Speech energy and vocal emphasis analysis complete.")

            # 4. Audio Themes
            if self.theme_model:
                try:
                    sentences = [seg['text'] for seg in transcription]
                    embeddings = self.theme_model.encode(sentences)
                    num_clusters = min(5, len(sentences))
                    if num_clusters > 0:
                        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
                        themes = {}
                        for i, label in enumerate(kmeans.labels_):
                            if label not in themes:
                                themes[label] = []
                            themes[label].append(sentences[i])
                        audio_analysis_results['audio_themes'] = {f'theme_{k}': " ".join(v) for k, v in themes.items()}
                        self.log_info(f"Identified {num_clusters} audio themes.")
                except Exception as e:
                    self.log_error(f"Failed to run theme identification model: {e}")
            else:
                self.log_warning("Theme identification model not loaded. Skipping theme analysis.")

            context['audio_analysis_results'] = audio_analysis_results
            self.log_info("Advanced audio analysis complete.")
            set_stage_status('audio_analysis', 'complete', audio_analysis_results)
            return context

        except Exception as e:
            self.log_error(f"Error during advanced audio analysis: {e}")
            set_stage_status('audio_analysis', 'failed', {'reason': str(e)})
            return context
