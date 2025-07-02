# gpu_manager.py
"""
Manages GPU resources, specifically for releasing memory.
"""

import torch
import gc

def release_gpu_memory():
    """
    Releases all unused cached memory from PyTorch.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print("✅ \033[92mGPU memory cache cleared.\033[0m")
        except Exception as e:
            print(f"⚠️ \033[93mCould not clear GPU memory cache: {e}\033[0m")
    
    # The garbage collector is also called to ensure that any Python objects
    # that are no longer in use are cleaned up, which can also help free up memory.
    gc.collect()
