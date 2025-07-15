import os
from agents.base_agent import Agent
from typing import Dict, Any, List, Optional, Tuple
import cv2
from mediapipe.python.solutions import holistic as mp_holistic_solutions
import numpy as np
from core.state_manager import set_stage_status
from core.cache_manager import cache_manager # Import cache_manager
from core.resource_manager import resource_manager # Import resource_manager
from deepface import DeepFace
import time
from llm.llm_interaction import robust_llm_json_extraction
from llm.image_analysis import ImageAnalysisResult
from video.frame_processor import FrameProcessor # Import FrameProcessor

class MultimodalAnalysisAgent(Agent):
    """
    Agent for comprehensive multimodal video analysis, combining:
    - Video analysis (facial expressions, gestures, visual complexity, energy levels)
    - Qwen-VL vision analysis (scene description, content type, hook potential)
    - Engagement analysis (calculating and aggregating engagement scores)
    """

    # Constants for configuration and thresholds (from VideoAnalysisAgent)
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    HOLISTIC_MODEL_COMPLEXITY = 1 # Lighter model for faster processing
    PROGRESS_LOG_INTERVAL = 100 # Log progress every N frames

    # Engagement score weights (from VideoAnalysisAgent)
    FACE_WEIGHT = 0.4
    GESTURE_WEIGHT = 0.2
    ENERGY_WEIGHT = 0.3
    COMPLEXITY_WEIGHT = 0.1
    COMPLEXITY_NORMALIZATION_FACTOR = 1000.0 # Factor to normalize complexity score

    def __init__(self, agent_config, state_manager):
        super().__init__("MultimodalAnalysisAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.engagement_config = agent_config.get('engagement_analysis', {})
        self.qwen_vision_config = self.config.get('qwen_vision', {})
        self.ollama_image_model_name = self.config.get('llm.image_model', "llava:latest")
        self.mp_holistic = mp_holistic_solutions.Holistic

    def _process_frame_data(self, frame_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Reads a frame from path and performs basic validation."""
        current_frame = cv2.imread(frame_path)
        if current_frame is None or current_frame.size == 0:
            self.log_warning(f"Invalid frame at {timestamp}s from path {frame_path}")
            return None
        return current_frame

    def _perform_deepface_analysis(self, frame: np.ndarray, timestamp: float) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """Performs facial emotion analysis using DeepFace."""
        dominant_emotion = "neutral"
        deepface_results = None
        try:
            deepface_results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='mediapipe',
                silent=True
            )
            if deepface_results and len(deepface_results) > 0:
                dominant_emotion = deepface_results[0]['dominant_emotion']
        except Exception as e:
            self.log_warning(f"DeepFace analysis failed for frame at {timestamp}s: {e}")
        return dominant_emotion, deepface_results # Return deepface_results for face count

    def _perform_gesture_recognition(self, frame: np.ndarray, holistic_processor: mp_holistic_solutions.Holistic) -> List[str]:
        """Performs gesture recognition using MediaPipe Holistic."""
        gestures = []
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_processor.process(rgb_frame)
            if results.pose_landmarks: # type: ignore
                gestures.append("pose")
            if results.left_hand_landmarks: # type: ignore
                gestures.append("left_hand")
            if results.right_hand_landmarks: # type: ignore
                gestures.append("right_hand")
        except Exception as e:
            self.log_warning(f"Gesture recognition failed: {e}")
        return gestures

    def _calculate_visual_complexity(self, gray_frame: np.ndarray, timestamp: float) -> float:
        """Calculates visual complexity using Laplacian variance."""
        complexity = 0.0
        try:
            complexity = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        except Exception as e:
            self.log_warning(f"Complexity calculation failed at {timestamp}s: {e}")
        return float(complexity)

    def _calculate_energy_level(self, prev_gray_frame: Optional[np.ndarray], current_gray_frame: np.ndarray, timestamp: float) -> float:
        """Calculates energy level using optical flow."""
        energy = 0.0
        if prev_gray_frame is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, current_gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0) # type: ignore
                energy = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            except Exception as e:
                self.log_warning(f"Optical flow calculation failed at {timestamp}s: {e}")
        return float(energy)

    def _calculate_engagement_score(self, faces_detected_count: int, gestures: List[str], energy: float, complexity: float) -> float:
        """Calculates an engagement score based on various metrics."""
        engagement_score = 0.0
        try:
            normalized_complexity = complexity / self.COMPLEXITY_NORMALIZATION_FACTOR
            engagement_score = (
                (faces_detected_count * self.FACE_WEIGHT) +
                (len(gestures) * self.GESTURE_WEIGHT) +
                (energy * self.ENERGY_WEIGHT) +
                (normalized_complexity * self.COMPLEXITY_WEIGHT)
            )
        except Exception as e:
            self.log_warning(f"Engagement calculation failed: {e}")
        return float(engagement_score)

    def _aggregate_engagement_ranges(self, engagement_scores: list) -> dict:
        """
        Aggregates per-frame engagement scores into high-engagement ranges.
        """
        if not engagement_scores:
            return {'high_engagement_ranges': [], 'overall_stats': {}}

        scores = [s['engagement_score'] for s in engagement_scores]
        overall_mean = float(np.mean(scores))
        overall_std = float(np.std(scores))
        
        high_engagement_threshold = overall_mean + (overall_std * self.engagement_config.get('high_engagement_std_multiplier', 1.0))
        
        high_engagement_ranges = []
        current_range = None

        for score_data in engagement_scores:
            timestamp = score_data['timestamp']
            score = score_data['engagement_score']

            if score >= high_engagement_threshold:
                if current_range is None:
                    current_range = {
                        'start': timestamp,
                        'end': timestamp,
                        'scores': [score]
                    }
                else:
                    current_range['end'] = timestamp
                    current_range['scores'].append(score)
            else:
                if current_range is not None:
                    high_engagement_ranges.append({
                        'start': current_range['start'],
                        'end': current_range['end'],
                        'avg_score': float(np.mean(current_range['scores'])),
                        'peak_score': float(np.max(current_range['scores']))
                    })
                    current_range = None
        
        if current_range is not None:
            high_engagement_ranges.append({
                'start': current_range['start'],
                'end': current_range['end'],
                'avg_score': float(np.mean(current_range['scores'])),
                'peak_score': float(np.max(current_range['scores']))
            })

        return {
            'high_engagement_ranges': high_engagement_ranges,
            'overall_stats': {
                'mean': overall_mean,
                'std': overall_std,
                'peak_count': len(high_engagement_ranges)
            }
        }

    def _perform_qwen_vision_analysis(self, frame_path: str, timestamp_sec: float) -> Dict[str, Any]:
        """Performs Qwen-VL analysis on a single frame."""
        system_prompt = """
        You are an expert image analysis AI. Your task is to describe the provided image concisely,
        identify the content type, and assess its hook potential. Focus on people and their clothing, features.

        You MUST output ONLY a JSON object that adheres to the ImageAnalysisResult Pydantic schema.
        DO NOT include any other text, explanations, or markdown outside of the JSON block.
        DO NOT include any ```json``` markdown blocks!!!!

        Example output:
        
        {
          "scene_description": "Man with a black hat wearing a red checkered shirt speaking into the microphone.",
          "content_type": "educational/entertainment/promotional/tutorial/discussion/other",
          "hook_potential": 7
        }

        """
        llm_user_prompt = "Generate the image analysis JSON for the provided image."

        try:
            analysis_result: ImageAnalysisResult = robust_llm_json_extraction(
                system_prompt=system_prompt,
                user_prompt=llm_user_prompt,
                output_schema=ImageAnalysisResult,
                image_path=frame_path,
                max_attempts=5
            )
            return {
                "timestamp": timestamp_sec,
                "content_type": analysis_result.content_type,
                "hook_potential": analysis_result.hook_potential,
                "scene_description": analysis_result.scene_description
            }
        except Exception as e:
            self.log_error(f"Error processing frame {frame_path} with Ollama: {e}")
            # Attempt to unload models if an error occurs during LLM analysis
            resource_manager.unload_all_models()
            return {
                "timestamp": timestamp_sec,
                "description": "Error during analysis",
                "error_details": str(e)
            }

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_name = self.name
        print(f"\nExecuting stage: {stage_name}")

        # --- Pre-flight Check ---
        # If multimodal analysis results already exist in the context, skip this stage.
        if context.get('current_analysis', {}).get('multimodal_analysis_results') and context.get('pipeline_stages', {}).get(stage_name) == 'complete':
            print(f"✅ Skipping {stage_name}: Multimodal analysis already complete.")
            return context

        # Ensure 'current_analysis' and 'summaries' are dictionaries
        context.setdefault('current_analysis', {})
        context.setdefault('summaries', {})

        # Ensure 'current_analysis' and 'summaries' are dictionaries
        context.setdefault('current_analysis', {})
        context.setdefault('summaries', {})

        processed_video_path = context.get("processed_video_path")
        video_info = context['metadata'].get("video_info")
        processing_settings = context.get('metadata', {}).get('processing_settings', {})
        frame_analysis_rate = processing_settings.get('frame_analysis_rate', 1) # Default to 1 FPS

        if not processed_video_path or not video_info:
            self.log_error("Video path or info missing. Skipping multimodal analysis.")
            set_stage_status('multimodal_analysis', 'skipped', {'reason': 'Missing video path or info'})
            return context

        # Initialize FrameProcessor with actual video dimensions
        # This is needed for frame extraction within this agent
        original_w = video_info.get('width')
        original_h = video_info.get('height')
        output_w = self.config.get('qwen_vision.output_width', 1280) # Example default
        output_h = self.config.get('qwen_vision.output_height', 720) # Example default
        
        if original_w is None or original_h is None:
            self.log_error("Original video dimensions missing. Cannot initialize FrameProcessor.")
            set_stage_status('multimodal_analysis', 'failed', {'reason': 'Missing video dimensions'})
            return context

        frame_processor = FrameProcessor(original_w, original_h, output_w, output_h)

        print(f"👁️ Starting comprehensive multimodal analysis at {frame_analysis_rate} FPS...")
        set_stage_status('multimodal_analysis', 'running')

        cache_key = f"multimodal_analysis_{os.path.basename(processed_video_path)}_{frame_analysis_rate}"
        cached_results = cache_manager.get(cache_key, level="disk")

        if cached_results:
            print("⏩ Skipping multimodal analysis. Loaded from cache.")
            context.update(cached_results)
            set_stage_status('multimodal_analysis_complete', 'complete', {'loaded_from_cache': True})
            return context

        multimodal_analysis_results = {
            "facial_expressions": [], "gesture_recognition": [],
            "visual_complexity": [], "energy_levels": [], "engagement_metrics": []
        }
        
        all_engagement_scores = [] # To collect all scores for aggregation

        processed_frame_count = 0
        prev_gray_frame = None
        start_time = time.time()

        try:
            with self.mp_holistic(
                min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE,
                model_complexity=self.HOLISTIC_MODEL_COMPLEXITY
            ) as holistic:
                
                # Extract frames at the specified rate
                extracted_frames_info = frame_processor.extract_frames_at_rate(
                    video_path=processed_video_path,
                    fps=frame_analysis_rate
                )
                
                total_frames_to_process = len(extracted_frames_info)
                if total_frames_to_process == 0:
                    self.log_warning("No frames extracted for multimodal analysis. Skipping.")
                    set_stage_status('multimodal_analysis', 'skipped', {'reason': 'No frames extracted'})
                    return context

                for frame_info in extracted_frames_info:
                    frame_path = frame_info['frame_path']
                    timestamp = frame_info['timestamp_sec']
                    
                    current_frame = self._process_frame_data(frame_path, timestamp)
                    if current_frame is None:
                        continue
                    
                    if processed_frame_count % self.PROGRESS_LOG_INTERVAL == 0:
                        progress = (processed_frame_count / total_frames_to_process) * 100
                        print(f"🎞️ Processing frame {processed_frame_count}/{total_frames_to_process} ({progress:.1f}%)")

                    # 1. DeepFace analysis
                    dominant_emotion, deepface_results = self._perform_deepface_analysis(current_frame, timestamp)
                    multimodal_analysis_results["facial_expressions"].append({
                        "timestamp": timestamp,
                        "expression": dominant_emotion
                    })

                    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                    # 2. Gesture recognition
                    gestures = self._perform_gesture_recognition(current_frame, holistic)
                    multimodal_analysis_results["gesture_recognition"].append({
                        "timestamp": timestamp,
                        "gestures": gestures
                    })

                    # 3. Visual complexity
                    complexity = self._calculate_visual_complexity(gray_frame, timestamp)
                    multimodal_analysis_results["visual_complexity"].append({
                        "timestamp": timestamp,
                        "score": complexity
                    })

                    # 4. Energy levels
                    energy = self._calculate_energy_level(prev_gray_frame, gray_frame, timestamp)
                    multimodal_analysis_results["energy_levels"].append({
                        "timestamp": timestamp,
                        "level": energy
                    })
                    prev_gray_frame = gray_frame.copy()

                    # 5. Engagement score
                    faces_detected_count = len(deepface_results) if deepface_results else 0
                    engagement_score = self._calculate_engagement_score(faces_detected_count, gestures, energy, complexity)
                    multimodal_analysis_results["engagement_metrics"].append({
                        "timestamp": timestamp,
                        "score": engagement_score
                    })
                    all_engagement_scores.append({
                        'timestamp': timestamp,
                        'engagement_score': engagement_score,
                        'details': {
                            'expression': dominant_emotion,
                            'gestures': gestures,
                            'energy': energy
                        }
                    })
                    
                    processed_frame_count += 1
                        
        except Exception as e:
            self.log_error(f"Critical error during multimodal analysis: {e}")
            set_stage_status('multimodal_analysis', 'failed', {'reason': str(e)})
            return context

        processing_time = time.time() - start_time
        print(f"✅ Comprehensive multimodal analysis complete. Processed {processed_frame_count} frames in {processing_time:.2f}s")
        
        # Store raw analysis results in current_analysis
        context['current_analysis']['multimodal_analysis_results'] = multimodal_analysis_results
        
        # Aggregate engagement scores and store in summaries
        engagement_summary = self._aggregate_engagement_ranges(all_engagement_scores)
        context['summaries']['engagement_summary'] = engagement_summary

        set_stage_status('multimodal_analysis_complete', 'complete', {
            'metrics_collected': True,
            'frames_processed': processed_frame_count,
            'processing_time': processing_time,
            'num_high_engagement_ranges': len(engagement_summary['high_engagement_ranges'])
        })
        
        # Cache the results before returning
        cache_manager.set(cache_key, {
            'current_analysis': context.get('current_analysis'),
            'summaries': context.get('summaries')
        }, level="disk")

        return context
