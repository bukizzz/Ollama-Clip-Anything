from agents.base_agent import Agent
from typing import Dict, Any

from core.state_manager import set_stage_status
from llm.image_analysis import describe_image, ImageAnalysisResult
from video.scene_detection import SceneDetector
from video.frame_processor import FrameProcessor
from pydantic import BaseModel, Field
import os

class StoryboardAnalysis(BaseModel):
    content_type: str = Field(description="The type of content in the scene (e.g., discussion, demo).")
    hook_potential: int = Field(description="The hook potential of the scene (from 1 to 10).")
    scene_description: str = Field(description="A concise description of the scene.")

class StoryboardingAgent(Agent):
    """Agent for generating a detailed storyboard with multimodal analysis."""

    def __init__(self, config, state_manager):
        super().__init__("StoryboardingAgent")
        self.config = config
        self.state_manager = state_manager
        self.scene_detector = SceneDetector(self.config)
        # FrameProcessor needs video dimensions, which are available in context['video_info']
        # Initialize with dummy values for now, and re-initialize in execute if needed
        self.frame_processor = None 

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        video_path = context.get("processed_video_path")
        video_info = context.get("video_info")
        # qwen_results = context.get('qwen_vision_analysis_results', []) # No longer directly used in storyboard entry

        if not video_path or not video_info:
            self.log_error("Video path or info missing. Skipping storyboarding.")
            set_stage_status('storyboarding', 'skipped', {'reason': 'Missing video path or info'})
            return context

        # Initialize FrameProcessor with actual video dimensions
        if self.frame_processor is None:
            original_w = video_info.get('width')
            original_h = video_info.get('height')
            # Assuming output dimensions are fixed or can be derived from config
            output_w = self.config.get('qwen_vision.output_width', 1280) # Example default
            output_h = self.config.get('qwen_vision.output_height', 720) # Example default
            
            if original_w is None or original_h is None:
                self.log_error("Original video dimensions missing. Cannot initialize FrameProcessor.")
                set_stage_status('storyboarding', 'failed', {'reason': 'Missing video dimensions'})
                return context

            self.frame_processor = FrameProcessor(original_w, original_h, output_w, output_h)


        print("📝 Starting enhanced storyboarding...")
        set_stage_status('storyboarding', 'running')

        extracted_frames_info = context.get("extracted_frames_info")
        if not extracted_frames_info:
            self.log_error("Extracted frames info missing. Skipping storyboarding.")
            set_stage_status('storyboarding', 'skipped', {'reason': 'Missing extracted_frames_info'})
            return context

        scene_boundaries = self.scene_detector.detect_scene_changes(video_path)
        print(f"DEBUG: Detected scene boundaries: {scene_boundaries}")

        # Create a set of all relevant timestamps to analyze: start, end, and all scene boundaries
        timestamps_to_analyze = {0.0, video_info['duration']}
        for boundary in scene_boundaries:
            timestamps_to_analyze.add(boundary)

        # Create a mapping of existing extracted frames for quick lookup
        existing_frames_map = {}
        for frame_info in extracted_frames_info:
            # Use a rounded timestamp as key for approximate matching
            sec_rounded = round(frame_info['timestamp_sec'], 1) # Round to 1 decimal for better matching
            if sec_rounded not in existing_frames_map:
                existing_frames_map[sec_rounded] = []
            existing_frames_map[sec_rounded].append(frame_info)

        frames_for_storyboard_analysis = []
        for ts in sorted(list(timestamps_to_analyze)):
            found_existing = False
            # Check if a frame already exists very close to this timestamp
            # Check current rounded second, and +/- 0.1 seconds for precision
            for offset in [-0.1, 0.0, 0.1]: 
                rounded_ts_check = round(ts + offset, 1)
                if rounded_ts_check in existing_frames_map:
                    for frame_info in existing_frames_map[rounded_ts_check]:
                        if abs(frame_info['timestamp_sec'] - ts) < 0.2: # Within 0.2 seconds
                            frames_for_storyboard_analysis.append(frame_info)
                            found_existing = True
                            break
                if found_existing:
                    break
            
            if not found_existing:
                # If no sufficiently close frame exists, extract a new one
                self.log_info(f"Extracting new frame at scene boundary: {ts:.2f}s")
                new_frame_path = self.frame_processor.extract_frame_at_timestamp(video_path, ts)
                if new_frame_path and os.path.exists(new_frame_path): # Ensure file was actually created
                    frames_for_storyboard_analysis.append({
                        'frame_path': new_frame_path,
                        'timestamp_sec': ts
                    })
                else:
                    self.log_warning(f"Failed to extract frame at {ts:.2f}s for storyboarding or file not found.")

        # Filter out duplicates based on frame_path and sort by timestamp
        seen_frame_paths = set()
        final_frames_to_analyze = []
        for frame_info in frames_for_storyboard_analysis:
            if frame_info['frame_path'] not in seen_frame_paths:
                seen_frame_paths.add(frame_info['frame_path'])
                final_frames_to_analyze.append(frame_info)
        
        final_frames_to_analyze = sorted(final_frames_to_analyze, key=lambda x: x['timestamp_sec'])

        print(f"DEBUG: Final list of unique frames selected for LLM analysis (paths): {[f['frame_path'] for f in final_frames_to_analyze]}")
        print(f"DEBUG: Final list of unique frames selected for LLM analysis (timestamps): {[f['timestamp_sec'] for f in final_frames_to_analyze]}")

        storyboard = []
        for frame_info in final_frames_to_analyze: # Use the new list
            timestamp = frame_info['timestamp_sec']
            temp_frame_path = frame_info['frame_path']

            # Multimodal LLM analysis
            print(f"🧠 Performing LLM analysis for frame at {timestamp:.2f}s. Frame path: {temp_frame_path}...\n")
            # Refined prompt for conciseness: Emphasize very short descriptions
            prompt = f"Describe the image at {timestamp:.2f}s in 5-10 words. Focus on key visual elements, people, and their actions. Identify the content type (e.g., discussion, demo, presentation) and hook potential (1-10)."
            
            analysis_result: ImageAnalysisResult # Declare type hint
            try:
                analysis_result = describe_image(temp_frame_path, prompt)
                print(f"✨ Extracted structured analysis for frame at {timestamp:.2f}s: {analysis_result.model_dump_json(indent=2)}\n")
            except Exception as e:
                self.log_error(f"LLM analysis failed for frame at {timestamp:.2f}s: {e}. Appending empty analysis and continuing.")
                analysis_result = ImageAnalysisResult(scene_description="N/A", content_type="unknown", hook_potential=0) # Fallback

            storyboard.append({
                "timestamp": timestamp,
                "description": analysis_result.scene_description,
                "content_type": analysis_result.content_type,
                "hook_potential": analysis_result.hook_potential,
                # Removed qwen_data to reduce token count
            })

        # Sort the storyboard by timestamp
        storyboard = sorted(storyboard, key=lambda x: x['timestamp'])
        context["storyboard_data"] = storyboard
        print(f"✅ Enhanced storyboarding complete. Generated {len(storyboard)} scenes.")
        set_stage_status('storyboarding_complete', 'complete', {'num_scenes': len(storyboard)})
        return context
