import os
from agents.base_agent import Agent
from core.state_manager import set_stage_status
from core.cache_manager import cache_manager # Import cache_manager

class FramePreprocessingAgent(Agent):
    def __init__(self, agent_config, state_manager):
        super().__init__("FramePreprocessingAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.qwen_vision_config = self.config.get('qwen_vision')

    def execute(self, context):
        stage_name = self.name
        print(f"\nExecuting stage: {stage_name}")

        # Reload context to ensure it has the latest video_info from VideoInputAgent
        context = self.state_manager._load_state_file()
        
        # DEBUG: Check video_info after reloading state
        print(f"DEBUG: FramePreprocessingAgent - video_info after reload: {context.get('video_info')}")
        print(f"DEBUG: FramePreprocessingAgent - metadata.video_info after reload: {context.get('metadata', {}).get('video_info')}")

        video_path = context.get('processed_video_path')
        if not video_path or not os.path.exists(video_path):
            self.log_error("Processed video path not found in context or does not exist.")
            set_stage_status('frame_preprocessing', 'failed', {'reason': 'Missing or invalid video path'})
            context['frame_preprocessing_status'] = 'failed'
            context['frame_preprocessing_error'] = 'Missing or invalid video path'
            return context

        # --- Pre-flight Check ---
        # If raw frames already exist in the context, skip this stage.
        if context.get('archived_data', {}).get('raw_frames') and context.get('pipeline_stages', {}).get(stage_name) == 'complete':
            print(f"✅ Skipping {stage_name}: Raw frames already processed.")
            return context

        print("📸 Starting frame preprocessing...")
        set_stage_status('frame_preprocessing', 'running')

        try:
            from video.frame_processor import FrameProcessor
            
            # Get target frame count from config or use a default
            target_frame_count = self.config.get('compression', {}).get('target_frame_count', 100)
            min_hash_diff = self.config.get('compression', {}).get('min_hash_diff', 10)

            cache_key = f"frames_{os.path.basename(video_path)}_{target_frame_count}_{min_hash_diff}"
            cached_frames = cache_manager.get(cache_key, level="disk")

            if cached_frames:
                print("⏩ Skipping frame preprocessing. Loaded from cache.")
                extracted_frames_info = cached_frames
            else:
                video_info = context.get('metadata', {}).get('video_info')
                if not video_info:
                    raise ValueError("Video info not found in context for FramePreprocessingAgent.")
                
                # Get target output resolution from processing settings
                processing_settings = context.get('metadata', {}).get('processing_settings', {})
                video_output_settings = processing_settings.get('video_output', {})
                
                # Default to source resolution if output settings are not fully defined
                target_width = video_output_settings.get('target_width', video_info['width'])
                target_height = video_output_settings.get('target_height', video_info['height'])

                # Initialize FrameProcessor with correct dimensions
                frame_processor = FrameProcessor(
                    original_w=video_info['width'],
                    original_h=video_info['height'],
                    output_w=target_width,
                    output_h=target_height
                )

                print(f"ℹ️ Selecting smart frames with target_count={target_frame_count}, min_hash_diff={min_hash_diff}...")
                extracted_frames_info = frame_processor.select_smart_frames(
                    video_path=video_path,
                    target_count=target_frame_count,
                    min_hash_diff=min_hash_diff
                )
                cache_manager.set(cache_key, extracted_frames_info, level="disk")
            
            print(f"✅ Selected {len(extracted_frames_info)} smart frames.")
            
            # Ensure 'archived_data' is a dictionary
            context.setdefault('archived_data', {})
            # Store raw frames info in archived_data and remove from current context
            context['archived_data']['raw_frames'] = extracted_frames_info
            if 'extracted_frames_info' in context:
                del context['extracted_frames_info']

            set_stage_status('frame_feature_extraction_complete', 'complete', {'num_frames': len(extracted_frames_info)})
            return context

        except Exception as e:
            self.log_error(f"Error during frame preprocessing: {e}")
            set_stage_status('frame_preprocessing', 'failed', {'reason': str(e)})
            context['frame_preprocessing_status'] = 'failed'
            context['frame_preprocessing_error'] = str(e)
            return context
