import os
import torch
import gc
import psutil
import logging
from typing import Dict, Any

from core.config import config
from core.gpu_manager import gpu_manager

logger = logging.getLogger(__name__)

class ResourceManager:
    def __init__(self):
        self.memory_limit_gb = config.get('resources.max_memory_gb', 16)
        self.vram_limit_gb = config.get('resources.max_vram_gb', 8)
        logger.info(f"Resource Manager initialized: RAM Limit={self.memory_limit_gb}GB, VRAM Limit={self.vram_limit_gb}GB")

    def get_current_memory_usage_gb(self) -> float:
        """Returns current RAM usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3) # Convert bytes to GB

    def get_current_vram_usage_gb(self) -> float:
        """Returns current VRAM usage in GB if CUDA is available."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0

    def should_unload_models(self) -> bool:
        """Checks if current memory or VRAM usage exceeds limits."""
        current_ram = self.get_current_memory_usage_gb()
        current_vram = self.get_current_vram_usage_gb()

        if current_ram > self.memory_limit_gb:
            logger.warning(f"RAM usage ({current_ram:.2f}GB) exceeds limit ({self.memory_limit_gb}GB).")
            return True
        if current_vram > self.vram_limit_gb:
            logger.warning(f"VRAM usage ({current_vram:.2f}GB) exceeds limit ({self.vram_limit_gb}GB).")
            return True
        return False

    def cleanup_inactive_data(self, context: Dict[str, Any]):
        """Removes inactive data from context to free up memory."""
        logger.info("Cleaning up inactive data from context...")
        # Example: Remove raw frames after they've been processed and summarized
        if 'archived_data' in context and 'raw_frames' in context['archived_data']:
            # Assuming raw_frames are file paths, they can be deleted from disk
            # For now, just remove the reference from context to free memory
            # Actual file deletion should be handled by temp_manager or specific agents
            del context['archived_data']['raw_frames']
            logger.debug("Removed 'raw_frames' from context.")
        
        # Example: Remove full transcription after summarization
        if 'archived_data' in context and 'full_transcription' in context['archived_data']:
            del context['archived_data']['full_transcription']
            logger.debug("Removed 'full_transcription' from context.")

        # Add more cleanup logic for other large, no-longer-needed data structures
        # For instance, if intermediate analysis results are no longer needed after summarization
        # and not for later stages, it could be cleared here.
        # This requires careful consideration of data dependencies between agents.

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Inactive data cleanup complete.")

    def unload_all_models(self):
        """Unloads all models managed by gpu_manager and clears GPU cache."""
        logger.info("Attempting to unload all models...")
        # Corrected: Call release_gpu_memory instead of non-existent unload_all_models
        gpu_manager.release_gpu_memory() 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded and GPU cache cleared.")

# Instantiate a global resource manager
resource_manager = ResourceManager()
