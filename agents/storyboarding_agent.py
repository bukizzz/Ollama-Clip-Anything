from agents.base_agent import Agent
from typing import Dict, Any

from core.state_manager import set_stage_status
from core.cache_manager import cache_manager # Import cache_manager
from llm.image_analysis import describe_images_batch
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
        stage_name = self.name
        print(f"\nExecuting stage: {stage_name}")

        # --- Pre-flight Check ---
        # If storyboard data already exists in the context, skip this stage.
        if context.get('storyboard_data') and context.get('pipeline_stages', {}).get(stage_name) == 'complete':
            print(f"✅ Skipping {stage_name}: Storyboard data already complete.")
            return context

        # Ensure 'metadata' is a dictionary and then get video_info
        context.setdefault('metadata', {})
        video_info = context['metadata'].get("video_info")
        processed_video_path = context.get("processed_video_path")

        if not processed_video_path or not video_info:
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

        cache_key = f"storyboard_{os.path.basename(processed_video_path)}"
        cached_storyboard = cache_manager.get(cache_key, level="disk")

        if cached_storyboard:
            print("⏩ Skipping storyboarding. Loaded from cache.")
            context["storyboard_data"] = cached_storyboard
            set_stage_status('storyboarding_complete', 'complete', {'loaded_from_cache': True})
            return context

        scene_boundaries = self.scene_detector.detect_scene_changes(processed_video_path)
        print(f"DEBUG: Detected scene boundaries: {scene_boundaries}")

        # Create a set of all relevant timestamps to analyze: start, end, and all scene boundaries
        timestamps_to_analyze = {0.0}
        for boundary in scene_boundaries:
            timestamps_to_analyze.add(boundary)
        
        # Add the end of the video, slightly adjusted to avoid off-by-one errors
        # Subtract a small epsilon (e.g., 0.01 seconds) from the total duration
        # to ensure the frame is extracted just before the very end.
        end_timestamp_adjusted = max(0.0, video_info['duration'] - 0.01)
        timestamps_to_analyze.add(end_timestamp_adjusted)

        frames_for_storyboard_analysis = []
        for ts in sorted(list(timestamps_to_analyze)):
            # Extract a new frame at each relevant timestamp
            self.log_info(f"Extracting frame at timestamp: {ts:.2f}s for storyboarding.")
            new_frame_path = self.frame_processor.extract_frame_at_timestamp(processed_video_path, ts)
            if new_frame_path and os.path.exists(new_frame_path):
                frames_for_storyboard_analysis.append({
                    'frame_path': new_frame_path,
                    'timestamp_sec': ts
                })
            else:
                self.log_warning(f"Failed to extract frame at {ts:.2f}s for storyboarding or file not found.")
                # --- NEW FALLBACK LOGIC ---
                # If it's the very last frame attempt and it failed, try a slightly earlier timestamp
                if abs(ts - video_info['duration']) < 0.05: # Check if it's close to the end
                    self.log_warning(f"Attempting fallback for last frame extraction at {ts:.2f}s.")
                    fallback_ts = max(0.0, video_info['duration'] - 0.1) # Try 0.1 seconds before end
                    fallback_frame_path = self.frame_processor.extract_frame_at_timestamp(processed_video_path, fallback_ts)
                    if fallback_frame_path and os.path.exists(fallback_frame_path):
                        self.log_info(f"Successfully extracted fallback frame at {fallback_ts:.2f}s for storyboarding.")
                        frames_for_storyboard_analysis.append({
                            'frame_path': fallback_frame_path,
                            'timestamp_sec': fallback_ts # Use fallback timestamp
                        })
                    else:
                        self.log_error(f"Fallback extraction for last frame also failed at {fallback_ts:.2f}s.")
                # --- END NEW FALLBACK LOGIC ---

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

        # Prepare frames for batch LLM analysis
        frames_for_batch_analysis = []
        for frame_info in final_frames_to_analyze:
            timestamp = frame_info['timestamp_sec']
            temp_frame_path = frame_info['frame_path']
            # Refined prompt for conciseness: Emphasize very short descriptions
            prompt = f"Describe the image at {timestamp:.2f}s in 5-10 words. Focus on key visual elements, people, and their actions. Identify the content type (e.g., discussion, demo, presentation) and hook potential (1-10)."
            frames_for_batch_analysis.append((temp_frame_path, prompt))

        print(f"🧠 Performing batch LLM analysis for {len(frames_for_batch_analysis)} frames...")
        batch_analysis_results = describe_images_batch(frames_for_batch_analysis)

        storyboard = []
        for i, analysis_result in enumerate(batch_analysis_results):
            frame_info = final_frames_to_analyze[i]
            timestamp = frame_info['timestamp_sec']
            
            storyboard.append({
                "timestamp": timestamp,
                "description": analysis_result.scene_description,
                "content_type": analysis_result.content_type,
                "hook_potential": analysis_result.hook_potential,
            })

        # Sort the storyboard by timestamp
        storyboard = sorted(storyboard, key=lambda x: x['timestamp'])
        context['storyboard_data'] = storyboard
        print(f"✅ Enhanced storyboarding complete. Generated {len(storyboard)} scenes.")
        set_stage_status('storyboarding_complete', 'complete', {'num_scenes': len(storyboard)})
        
        # Cache the results before returning
        cache_manager.set(cache_key, storyboard, level="disk")

        return context
