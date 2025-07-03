from agents.base_agent import Agent
from typing import Dict, Any
import cv2

from core.state_manager import set_stage_status
from llm import llm_interaction
from video.scene_detection import SceneDetector

class StoryboardingAgent(Agent):
    """Agent for generating a detailed storyboard with multimodal analysis."""

    def __init__(self, config, state_manager):
        super().__init__("StoryboardingAgent")
        self.config = config
        self.state_manager = state_manager
        self.scene_detector = SceneDetector()

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        video_path = context.get("processed_video_path")
        video_info = context.get("video_info")
        qwen_results = context.get('qwen_vision_analysis_results', [])

        if not video_path or not video_info:
            self.log_error("Video path or info missing. Skipping storyboarding.")
            set_stage_status('storyboarding', 'skipped', {'reason': 'Missing video path or info'})
            return context

        self.log_info("Starting enhanced storyboarding...")
        set_stage_status('storyboarding', 'running')

        storyboard = []
        cap = cv2.VideoCapture(video_path)
        # fps = video_info['fps']
        scene_boundaries = self.scene_detector.detect_scene_changes(video_path)

        for i, start_time in enumerate(scene_boundaries):
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            # pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Get Qwen-VL data for this scene
            qwen_data = next((r for r in qwen_results if r['timestamp'] >= start_time), None)
            
            # Multimodal LLM analysis
            self.log_info(f"🧠 \033[94mPerforming LLM analysis for scene at {start_time:.2f}s...\033[0m")
            prompt = f"""
            Analyze this video frame at {start_time:.2f}s, which marks a new scene.
            Qwen-VL analysis: {qwen_data.get('description', 'N/A') if qwen_data else 'N/A'}
            Identify content type (e.g., discussion, demo), hook potential, and a concise scene description.
            Output as JSON: {{"content_type": "...", "hook_potential": "...", "scene_description": "..."}}
            """
            
            analysis = {} # Initialize analysis to an empty dictionary
            try:
                response = llm_interaction.llm_pass(
                self.config.get('llm_model'),
                [{"role": "user", "content": prompt}],
                
            )
                extracted_analysis = llm_interaction.extract_json_from_text(response)
                if extracted_analysis is not None:
                    analysis = extracted_analysis
            except llm_interaction.InvalidJsonError as e:
                self.log_error(f"LLM analysis failed for scene at {start_time:.2f}s: {e}. Appending empty analysis and continuing.")
                analysis = {}
            except Exception as e:
                self.log_error(f"An unexpected error occurred during LLM analysis for scene at {start_time:.2f}s: {e}. Appending empty analysis and continuing.")
                analysis = {}

            storyboard.append({
                "scene_index": i,
                "timestamp": start_time,
                "description": analysis.get("scene_description", "N/A"),
                "content_type": analysis.get("content_type", "unknown"),
                "hook_potential": analysis.get("hook_potential", "low"),
                "qwen_data": qwen_data
            })

        cap.release()
        context["storyboard_data"] = storyboard
        self.log_info(f"Enhanced storyboarding complete. Generated {len(storyboard)} scenes.")
        set_stage_status('storyboarding_complete', 'complete', {'num_scenes': len(storyboard)})
        return context
