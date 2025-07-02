# core/gpu_manager.py
"""
Manages GPU resources, specifically for releasing memory and handling multiple models.
"""

import torch
import gc
import threading
from collections import deque
import time

class GPUManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.loaded_models = {}  # Stores references to loaded models
        self.model_queue = deque()  # Queue for models to be loaded
        self.memory_threshold = 0.9  # Unload models if GPU memory usage exceeds this (e.g., 90%)
        self.model_priorities = {} # Model priorities for loading/unloading
        self._initialized = True

    def _get_gpu_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated + cached) / total
        return 0.0

    def release_gpu_memory(self):
        """
        Releases all unused cached memory from PyTorch.
        """
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                print("✅ \033[92mGPU memory cache cleared.\033[0m")
            except Exception as e:
                print(f"⚠️ \033[93mCould not clear GPU memory cache: {e}\033[0m")
        
        gc.collect()

    def load_model(self, model_name: str, model_instance, priority: int = 0):
        """
        Loads a model onto the GPU, managing memory and priority.
        """
        with self._lock:
            self.model_priorities[model_name] = priority
            self.model_queue.append((model_name, model_instance))
            self._process_model_queue()

    def unload_model(self, model_name: str):
        """
        Unloads a specific model from the GPU.
        """
        with self._lock:
            if model_name in self.loaded_models:
                model = self.loaded_models.pop(model_name)
                del model
                self.release_gpu_memory()
                print(f"✅ \033[92mModel '{model_name}' unloaded from GPU.\033[0m")
            if model_name in self.model_priorities:
                del self.model_priorities[model_name]

    def _process_model_queue(self):
        """
        Processes the model loading queue based on priority and memory availability.
        """
        while self.model_queue:
            # Sort queue by priority (higher priority first)
            self.model_queue = deque(sorted(self.model_queue, key=lambda x: self.model_priorities.get(x[0], 0), reverse=True))
            
            model_name, model_instance = self.model_queue[0]

            if self._get_gpu_memory_usage() < self.memory_threshold:
                # Attempt to load the model
                try:
                    if torch.cuda.is_available():
                        model_instance.to("cuda")
                    self.loaded_models[model_name] = model_instance
                    self.model_queue.popleft()
                    print(f"✅ \033[92mModel '{model_name}' loaded onto GPU.\033[0m")
                except Exception as e:
                    print(f"⚠️ \033[93mCould not load model '{model_name}' onto GPU: {e}\033[0m")
                    # If loading fails, move to end of queue or handle as error
                    self.model_queue.rotate(-1) # Move to end
                    time.sleep(1) # Prevent busy-waiting
            else:
                print(f"⚠️ \033[93mGPU memory usage high ({self._get_gpu_memory_usage():.2f}). Attempting to unload least priority model.\033[0m")
                self._unload_least_priority_model()
                if self._get_gpu_memory_usage() >= self.memory_threshold:
                    print("❌ \033[91mInsufficient GPU memory to load model even after unloading. Manual intervention may be required.\033[0m")
                    break # Cannot load, break the loop

    def _unload_least_priority_model(self):
        """
        Unloads the model with the lowest priority to free up GPU memory.
        """
        if not self.loaded_models:
            return

        least_priority_model_name = None
        least_priority = float('inf')

        for name, model in self.loaded_models.items():
            priority = self.model_priorities.get(name, 0)
            if priority < least_priority:
                least_priority = priority
                least_priority_model_name = name
        
        if least_priority_model_name:
            self.unload_model(least_priority_model_name)

# Global instance for easy access
gpu_manager = GPUManager()