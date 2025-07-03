from agents.base_agent import Agent
from core.state_manager import set_stage_status
from core.llm_models import image_to_base64
import json
import ollama
from PIL import Image

class QwenVisionAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("QwenVisionAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.qwen_vision_config = agent_config.get('qwen_vision')
        self.ollama_model_name = self.qwen_vision_config.get('ollama_qwen_vl_model_name', "qwen2.5vl:3b")

    def execute(self, context):
        extracted_frames_info = context.get('extracted_frames_info')
        if not extracted_frames_info:
            self.log_error("Extracted frames information not found in context.")
            set_stage_status('qwen_vision_analysis', 'failed', {'reason': 'Missing extracted frames info'})
            context['qwen_vision_analysis_status'] = 'failed'
            context['qwen_vision_analysis_error'] = 'Missing extracted frames info'
            return context

        self.log_info("Starting Qwen-VL integration for advanced video understanding using Ollama...")
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
                for clip in clips:
                    if clip['start_time'] <= timestamp_sec <= clip['end_time']:
                        filtered_frames_info.append(frame_info)
                        break # Move to the next frame once it's found in a clip

            if not filtered_frames_info:
                self.log_warning("No frames found within the selected clip ranges. QwenVisionAgent will not analyze any frames.")
                set_stage_status('qwen_vision_analysis', 'skipped', {'reason': 'No frames within clip ranges'})
                context['qwen_vision_analysis_status'] = 'skipped'
                return context

            self.log_info(f"Processing {len(filtered_frames_info)} frames relevant to selected clips.")

            qwen_analysis_results = []
            batch_size = self.qwen_vision_config.get('batch_size', 5) # Default to 5 if not specified

            for i in range(0, len(filtered_frames_info), batch_size):
                batch_frames = filtered_frames_info[i:i + batch_size]
                self.log_info(f"Processing batch {i // batch_size + 1}/{(len(filtered_frames_info) + batch_size - 1) // batch_size} with {len(batch_frames)} frames.")

                for frame_info in batch_frames:
                    frame_path = frame_info['frame_path']
                    timestamp_sec = frame_info['timestamp_sec']

                    try:
                        # Load image and convert to base64
                        with Image.open(frame_path) as img:
                            base64_image = image_to_base64(img)

                        # Prompt for structured JSON output for each frame
                        messages = [
                            {
                                'role': 'user',
                                'content': f'Describe the scene at {timestamp_sec:.2f}s, identify all objects with their bounding boxes, and note any significant events or interactions. Output as JSON: {{"timestamp": {timestamp_sec:.2f}, "description": "...", "objects": [...], "events": [...]}}',
                                'images': [base64_image]
                            }
                        ]

                        self.log_info(f"🧠 \u001b[94mSending frame {frame_path} at {timestamp_sec:.2f}s to Ollama ({self.ollama_model_name}) for analysis.\u001b[0m")
                        response = ollama.chat(model=self.ollama_model_name, messages=messages)
                        
                        if 'message' in response and 'content' in response['message']:
                            response_content = response['message']['content']
                            self.log_info(f"Received response for frame {frame_path} at {timestamp_sec:.2f}s from Ollama.")
                            
                            # Attempt to parse JSON from the response content
                            try:
                                frame_analysis = json.loads(response_content)
                                qwen_analysis_results.append(frame_analysis)
                                self.log_info(f"Processed frame at {timestamp_sec:.2f}s with Ollama.")
                            except json.JSONDecodeError as json_e:
                                self.log_error(f"Error parsing JSON response for frame {frame_path}: {json_e}. Response: {response_content[:500]}...")
                                qwen_analysis_results.append({
                                    "frame_path": frame_path,
                                    "timestamp": timestamp_sec,
                                    "description": "Error during JSON parsing",
                                    "objects": [],
                                    "events": [],
                                    "raw_response": response_content
                                })
                        else:
                            raise ValueError(f"Unexpected Ollama response format: {response}")

                    except Exception as e:
                        self.log_error(f"Error processing frame {frame_path} with Ollama: {e}")
                        qwen_analysis_results.append({
                            "frame_path": frame_path,
                            "timestamp": timestamp_sec,
                            "description": "Error during analysis",
                            "objects": [],
                            "events": []
                        })

            context['qwen_vision_analysis_results'] = qwen_analysis_results
            self.log_info("Qwen-VL integration with Ollama complete.")
            set_stage_status('qwen_vision_analysis_complete', 'complete', {'num_frames_analyzed': len(qwen_analysis_results)})
            return context

        except Exception as e:
            self.log_error(f"Error during Qwen-VL integration with Ollama: {e}")
            set_stage_status('qwen_vision_analysis', 'failed', {'reason': str(e)})
            context['qwen_vision_analysis_status'] = 'failed'
            context['qwen_vision_analysis_error'] = str(e)
            return context