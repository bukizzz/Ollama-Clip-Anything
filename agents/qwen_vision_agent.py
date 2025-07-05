import os
from agents.base_agent import Agent
from core.state_manager import set_stage_status
import json
import ollama
import time
from llm.llm_interaction import robust_llm_json_extraction
from llm.image_analysis import ImageAnalysisResult

class QwenVisionAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("QwenVisionAgent")
        self.config = agent_config
        self.state_manager = state_manager
        # Use the global image model from the main config
        self.ollama_model_name = self.config.get('llm.image_model', "llava:latest")

    def execute(self, context):
        extracted_frames_info = context.get('extracted_frames_info')
        if not extracted_frames_info:
            self.log_error("Extracted frames information not found in context.")
            set_stage_status('qwen_vision_analysis', 'failed', {'reason': 'Missing extracted frames info'})
            context['qwen_vision_analysis_status'] = 'failed'
            context['qwen_vision_analysis_error'] = 'Missing extracted frames info'
            return context

        print("👁️ Starting Qwen-VL integration for advanced video understanding using Ollama...")
        set_stage_status('qwen_vision_analysis', 'running')

        try:
            clips = context.get('clips')
            if not clips:
                self.log_warning("No clips found in context. QwenVisionAgent will not analyze any frames.")
                set_stage_status('qwen_vision_analysis', 'skipped', {'reason': 'No clips to analyze'})
                context['qwen_vision_analysis_status'] = 'skipped'
                return context

            # Filter frames to only include those within selected clip ranges
            filtered_frames_info = []
            for frame_info in extracted_frames_info:
                timestamp_sec = frame_info['timestamp_sec']
                for clip in clips: # 'clip' here is a dictionary representing a Clip object
                    for scene in clip.get('scenes', []): # Iterate through the scenes within each clip
                        if scene['start_time'] <= timestamp_sec <= scene['end_time']:
                            filtered_frames_info.append(frame_info)
                            break # Move to the next frame once it's found in a scene
                    if frame_info in filtered_frames_info: # If frame was added, break outer loop too
                        break

            if not filtered_frames_info:
                self.log_warning("No frames found within the selected clip ranges. QwenVisionAgent will not analyze any frames.")
                set_stage_status('qwen_vision_analysis', 'skipped', {'reason': 'No frames within clip ranges'})
                context['qwen_vision_analysis_status'] = 'skipped'
                return context

            print(f"🖼️ Processing {len(filtered_frames_info)} frames relevant to selected clips.")

            qwen_analysis_results = []
            # Access batch_size directly from the main config
            batch_size = self.config.get('qwen_vision.batch_size', 5) # Default to 5 if not specified

            # Define system and user prompts for robust_llm_json_extraction
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


            for i in range(0, len(filtered_frames_info), batch_size):
                batch_frames = filtered_frames_info[i:i + batch_size]
                print(f"📦 Processing batch {i // batch_size + 1}/{(len(filtered_frames_info) + batch_size - 1) // batch_size} with {len(batch_frames)} frames.")

                for frame_info in batch_frames:
                    frame_path = frame_info['frame_path']
                    timestamp_sec = frame_info['timestamp_sec']

                    try:
                        print(f"🧠 Sending frame {frame_path} (actual path: {os.path.abspath(frame_path)}) at {timestamp_sec:.2f}s to Ollama ({self.ollama_model_name}) for analysis. Type of frame_path: {type(frame_path)}")
                        
                        # Use robust_llm_json_extraction for parsing
                        analysis_result: ImageAnalysisResult = robust_llm_json_extraction(
                            system_prompt=system_prompt,
                            user_prompt=llm_user_prompt,
                            output_schema=ImageAnalysisResult,
                            image_path=frame_path,
                            max_attempts=5 # Allow retries for robustness
                        )
                        
                        qwen_analysis_results.append({
                            "frame_path": frame_path,
                            "timestamp": timestamp_sec,
                            "content_type": analysis_result.content_type,
                            "hook_potential": analysis_result.hook_potential,
                            "scene_description": analysis_result.scene_description
                        })
                        print(f"✅ Processed frame at {timestamp_sec:.2f}s with Ollama.")

                    except Exception as e:
                        self.log_error(f"Error processing frame {frame_path} with Ollama: {e}")
                        qwen_analysis_results.append({
                            "frame_path": frame_path,
                            "timestamp": timestamp_sec,
                            "description": "Error during analysis",
                            "error_details": str(e)
                        })

            context['qwen_vision_analysis_results'] = qwen_analysis_results
            print("✅ Qwen-VL integration with Ollama complete.")
            set_stage_status('qwen_vision_analysis_complete', 'complete', {'num_frames_analyzed': len(qwen_analysis_results)})
            return context

        except Exception as e:
            self.log_error(f"Error during Qwen-VL integration with Ollama: {e}")
            set_stage_status('qwen_vision_analysis', 'failed', {'reason': str(e)})
            context['qwen_vision_analysis_status'] = 'failed'
            context['qwen_vision_analysis_error'] = str(e)
            return context
