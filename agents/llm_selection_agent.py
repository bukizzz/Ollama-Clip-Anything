import os
from typing import List, Dict, Any

from agents.base_agent import Agent
from llm import llm_interaction
from llm.prompt_utils import build_adaptive_prompt
from core.resource_manager import resource_manager
from core.cache_manager import cache_manager
from core.state_manager import set_stage_status # Import set_stage_status directly

class LLMSelectionAgent(Agent):
    """Agent responsible for selecting engaging clips using an LLM."""

    def __init__(self, agent_config, state_manager):
        super().__init__("LLMSelectionAgent")
        self.config = agent_config
        self.state_manager = state_manager
        self.llm_selection_config = self.config.get('llm_selection', {})

    def _filter_by_engagement(self, transcription: List[Dict[str, Any]], engagement_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters transcription segments based on high engagement scores.
        """
        threshold = self.llm_selection_config.get('engagement_threshold', 0.6)
        filtered_segments = []
        for segment in transcription:
            segment_start = segment['start']
            segment_end = segment['end']
            
            # Find average engagement for this segment
            relevant_engagement = [
                e['score'] for e in engagement_results # Changed to 'score'
                if segment_start <= e['timestamp'] <= segment_end
            ]
            self.log_info(f"DEBUG: relevant_engagement for segment {segment_start}-{segment_end}: {relevant_engagement}")
            if relevant_engagement and sum(relevant_engagement) / len(relevant_engagement) >= threshold:
                filtered_segments.append(segment)
        return filtered_segments

    def _group_by_semantic_similarity(self, segments: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Groups transcription segments that are semantically similar or chronologically close.
        (Placeholder for actual semantic grouping, currently groups by chronological proximity)
        """
        grouped_segments = []
        if not segments:
            return []

        current_group = [segments[0]]
        for i in range(1, len(segments)):
            # Simple chronological grouping: if segments are close enough, group them
            if segments[i]['start'] - current_group[-1]['end'] < self.llm_selection_config.get('grouping_time_threshold', 5): # 5 seconds
                current_group.append(segments[i])
            else:
                grouped_segments.append(current_group)
                current_group = [segments[i]]
        grouped_segments.append(current_group) # Add the last group
        return grouped_segments

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        transcription = context.get("archived_data", {}).get("full_transcription") # Get raw transcription
        user_prompt = context.get("user_prompt")
        storyboarding_data = context.get('storyboarding_data')
        engagement_analysis_results = context.get('current_analysis', {}).get('multimodal_analysis_results', {}).get('engagement_metrics') # Get raw engagement metrics

        print("\n🧠 \u001b[94mSelecting coherent clips using LLM...\u001b[0m")
        set_stage_status('llm_selection', 'running') # Call the function directly

        processed_video_path = context.get("processed_video_path")
        if processed_video_path is None:
            raise RuntimeError("Processed video path is missing from context. Cannot create cache key.")

        cache_key = f"llm_selection_{os.path.basename(processed_video_path)}_{context.get('user_prompt', '')}"
        cached_clips = cache_manager.get(cache_key, level="disk")

        if cached_clips:
            print("⏩ Skipping LLM clip selection. Loaded from cache.")
            context['current_analysis']['clips'] = cached_clips # Update clips in current_analysis
            set_stage_status('llm_selection_complete', 'complete', {'loaded_from_cache': True}) # Call the function directly
            return context

        if transcription is None or engagement_analysis_results is None:
            self.log_error("Transcription or engagement data is missing from context. Cannot select clips.")
            set_stage_status('llm_selection', 'failed', {'reason': 'Missing transcription or engagement data'}) # Call the function directly
            return context
        
        try:
            # Stage 1: Quick filtering based on engagement scores
            filtered_transcription = self._filter_by_engagement(transcription, engagement_analysis_results)
            self.log_info(f"Filtered transcription segments by engagement: {len(filtered_transcription)} segments.")

            # Stage 2: Content-aware grouping (simple chronological grouping for now)
            grouped_segments = self._group_by_semantic_similarity(filtered_transcription)
            self.log_info(f"Grouped segments into {len(grouped_segments)} groups.")

            # Stage 3: LLM selection with compressed context
            # Prepare data for LLM, focusing on summarization and token limits
            llm_context_data = {
                "grouped_transcription": grouped_segments,
                "storyboarding_data": storyboarding_data,
                "user_instructions": user_prompt
            }

            base_llm_prompt = """
            Analyze the following grouped transcription segments and storyboarding data.
            Select the most engaging and viral-worthy clips, adhering to the specified duration.
            """

            formatted_llm_prompt = build_adaptive_prompt(
                base_prompt=base_llm_prompt,
                context_data=llm_context_data,
                max_tokens=self.config.get('resources.max_tokens_per_llm_call', 40000),
                model_name=self.config.get('llm.current_active_llm_model', 'gemma'),
                prioritize_keys=['grouped_transcription', 'storyboarding_data', 'user_instructions'],
                compress_strategies={
                    'grouped_transcription': {'strategy': 'first_n', 'n': 10},
                    'storyboarding_data': {'strategy': 'first_n', 'n': 5}
                }
            )

            clips = llm_interaction.get_clips_from_llm(
                user_prompt=formatted_llm_prompt, # Pass the adaptive prompt
                storyboarding_data=None # Storyboarding data is part of the adaptive prompt
            )

            context['current_analysis']['clips'] = clips # Update clips in current_analysis
            
            print(f"✅ Selected {len(clips)} clips:")
            
            set_stage_status('llm_selection_complete', 'complete', {'num_clips': len(clips)}) # Call the function directly
            llm_interaction.cleanup() # Clear VRAM after successful completion
            
            # Cache the results before returning
            cache_manager.set(cache_key, clips, level="disk")

            return context

        except Exception as e:
            self.log_error(f"Failed to select clips with LLM: {e}")
            self.log_error(f"DEBUG: Full exception in LLMSelectionAgent.execute: {type(e).__name__}: {e}")
            set_stage_status('llm_selection', 'failed', {'reason': str(e)}) # Call the function directly
            resource_manager.unload_all_models() # Unload models on error
            return context
