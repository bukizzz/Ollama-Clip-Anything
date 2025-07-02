from agents.base_agent import Agent
from typing import Dict, Any
import cv2
from PIL import Image
from core import llm_models
from llm.image_analysis import describe_image

class StoryboardingAgent(Agent):
    """Agent responsible for analyzing video frames and generating a storyboard."""

    def __init__(self):
        super().__init__("StoryboardingAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        processed_video_path = context.get("processed_video_path")
        video_info = context.get("video_info")

        print("StoryboardingAgent: Analyzing frames and generating storyboard...")

        if not processed_video_path or not video_info:
            print("Skipping storyboarding: processed_video_path or video_info not available.")
            context["storyboard_data"] = "N/A"
            return context

        storyboard_data = []
        cap = cv2.VideoCapture(processed_video_path)
        fps = video_info['fps']
        duration = video_info['duration']

        # Sample 3 frames: beginning, middle, end
        frames_to_sample = [
            0,
            int(duration / 2 * fps),
            int((duration - 1) * fps) # Last second
        ]
        
        for i, frame_idx in enumerate(frames_to_sample):
            if frame_idx < 0: # Handle very short videos
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert OpenCV image to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Use LLM to describe the frame
                description_prompt = f"Describe the scene in this video frame at {frame_idx/fps:.2f} seconds."
                llm_description = describe_image(pil_image, description_prompt)
                print(f"""
🖼️ LLM Description for frame at {frame_idx/fps:.2f}s:
{llm_description}
""")
                
                storyboard_data.append({
                    "timestamp": frame_idx / fps,
                    "description": llm_description,
                    "frame_base64": llm_models.image_to_base64(pil_image) # Store base64 for potential later use
                })
            else:
                print(f"Could not read frame at index {frame_idx}")

        cap.release()
        context["storyboard_data"] = storyboard_data
        print(f"✅ \033[92mStoryboarding complete. Generated {len(storyboard_data)} storyboard entries.\033[0m")
        return context
