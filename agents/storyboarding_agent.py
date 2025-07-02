from agents.base_agent import BaseAgent
from typing import Dict, Any
import cv2
from PIL import Image
from core import llm_models
from llm.image_analysis import describe_image
from core.state_manager import set_stage_status, get_stage_status
from llm import llm_interaction

class StoryboardingAgent(BaseAgent):
    """Agent responsible for analyzing video frames and generating a storyboard."""

    def __init__(self, config, state_manager):
        super().__init__(config, state_manager)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_video_path = context.get("processed_video_path")
        video_info = context.get("video_info")
        qwen_vision_analysis_results = context.get('qwen_vision_analysis_results')

        if not processed_video_path or not video_info:
            self.log_error("Skipping storyboarding: processed_video_path or video_info not available.")
            set_stage_status('storyboarding', 'skipped', {'reason': 'Missing video path or info'})
            context["storyboard_data"] = "N/A"
            return context

        self.log_info("Starting enhanced storyboarding...")
        set_stage_status('storyboarding', 'running')

        storyboard_data = []
        cap = cv2.VideoCapture(processed_video_path)
        fps = video_info['fps']

        # Use scene boundaries if available, otherwise sample keyframes
        scene_boundaries = context.get('scene_boundaries') # Assuming scene detection agent will provide this
        frames_to_analyze = []

        if scene_boundaries:
            self.log_info(f"Using {len(scene_boundaries)} detected scene boundaries for storyboarding.")
            for boundary_time in scene_boundaries:
                frame_idx = int(boundary_time * fps)
                frames_to_analyze.append(frame_idx)
        else:
            self.log_warning("No scene boundaries detected. Sampling keyframes for storyboarding.")
            duration = video_info['duration']
            frames_to_analyze = [
                0,
                int(duration / 2 * fps),
                int((duration - 1) * fps) # Last second
            ]
        
        for frame_idx in frames_to_analyze:
            if frame_idx < 0: # Handle very short videos
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp_sec = frame_idx / fps

                # Integrate Qwen2.5-VL structured data for enhanced scene understanding
                qwen_data_for_frame = next((item for item in qwen_vision_analysis_results if item['frame_number'] == frame_idx), None) if qwen_vision_analysis_results else None
                qwen_description = f"Qwen-VL analysis: {qwen_data_for_frame}" if qwen_data_for_frame else "No Qwen-VL data for this frame."

                # Multimodal LLM analysis at scene boundaries
                llm_prompt = f"""
                Analyze the following video frame at {timestamp_sec:.2f} seconds.
                Consider the visual content and any provided Qwen-VL analysis:
                {qwen_description}

                Identify:
                1. Main content type (e.g., discussion, presentation, demo, reaction).
                2. Potential hook moments or viral potential (e.g., surprising reactions, strong statements).
                3. A concise scene-level content description.

                Provide the output in a structured JSON format with keys: "content_type", "hook_potential", "scene_description".
                """
                
                self.log_info(f"Performing multimodal LLM analysis for frame at {timestamp_sec:.2f}s...")
                try:
                    llm_response = llm_interaction.llm_pass(
                        llm_interaction.LLM_MODEL,
                        [
                            {"role": "system", "content": "You are an expert in video content analysis."},
                            {"role": "user", "content": llm_prompt.strip()}
                        ],
                        image=pil_image # Pass the image for multimodal analysis
                    )
                    llm_analysis = llm_interaction.extract_json_from_text(llm_response)
                except Exception as e:
                    self.log_error(f"LLM analysis failed for frame at {timestamp_sec:.2f}s: {e}")
                    llm_analysis = {"content_type": "unknown", "hook_potential": "none", "scene_description": "Error during analysis."} # Fallback

                storyboard_data.append({
                    "timestamp": timestamp_sec,
                    "description": llm_analysis.get("scene_description"),
                    "content_type": llm_analysis.get("content_type"),
                    "hook_potential": llm_analysis.get("hook_potential"),
                    "frame_base64": llm_models.image_to_base64(pil_image) # Store base64 for potential later use
                })
            else:
                self.log_warning(f"Could not read frame at index {frame_idx}")

        cap.release()
        context["storyboard_data"] = storyboard_data
        self.log_info(f"Enhanced storyboarding complete. Generated {len(storyboard_data)} storyboard entries.\033[0m")
        set_stage_status('storyboarding_complete', 'complete', {'num_entries': len(storyboard_data)})
        return context
