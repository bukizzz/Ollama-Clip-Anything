# core/gpu_manager.py
"""
Manages GPU resources, specifically for releasing memory and handling multiple models.
"""

import torch
import gc
import threading
from collections import deque
import time
import logging
import subprocess
from core.config import config

logger = logging.getLogger(__name__)

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
        self.loaded_models = {}
        self.model_queue = deque()
        self.memory_threshold = 0.9
        self.model_priorities = {}
        self._initialized = True
        self.logger = logging.getLogger(__name__)

    def _get_gpu_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated + cached) / total
        return 0.0

    def release_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared.")
        gc.collect()

    def load_model(self, model_name: str, model_instance, priority: int = 0):
        with self._lock:
            self.model_priorities[model_name] = priority
            self.model_queue.append((model_name, model_instance))
            self._process_model_queue()

    def unload_model(self, model_name: str):
        with self._lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                self.release_gpu_memory()
                logger.info(f"Model '{model_name}' unloaded from GPU.")
            if model_name in self.model_priorities:
                del self.model_priorities[model_name]

    def _process_model_queue(self):
        while self.model_queue:
            self.model_queue = deque(sorted(self.model_queue, key=lambda x: self.model_priorities.get(x[0], 0), reverse=True))
            model_name, model_instance = self.model_queue[0]

            if self._get_gpu_memory_usage() < self.memory_threshold:
                try:
                    if torch.cuda.is_available() and isinstance(model_instance, torch.nn.Module):
                        model_instance.to("cuda")
                    self.loaded_models[model_name] = model_instance
                    self.model_queue.popleft()
                    logger.info(f"Model '{model_name}' loaded onto GPU.")
                except Exception as e:
                    logger.error(f"Could not load model '{model_name}' onto GPU: {e}")
                    self.model_queue.rotate(-1)
                    time.sleep(1)
            else:
                logger.warning(f"GPU memory usage high ({self._get_gpu_memory_usage():.2f}). Attempting to unload least priority model.")
                self._unload_least_priority_model()
                if self._get_gpu_memory_usage() >= self.memory_threshold:
                    logger.error("Insufficient GPU memory to load model.")
                    break

    def _unload_least_priority_model(self):
        if not self.loaded_models:
            return

        least_priority_model_name = min(self.loaded_models, key=lambda name: self.model_priorities.get(name, 0))
        if least_priority_model_name:
            # Check if it's an Ollama model and unload using bash command
            if least_priority_model_name in [config.get('llm_model'), config.get('llm.image_model')]:
                self.unload_ollama_model(least_priority_model_name)
            else:
                self.unload_model(least_priority_model_name)

    def unload_ollama_model(self, model_name: str):
        try:
            self.logger.info(f"Attempting to unload Ollama model: {model_name}")
            subprocess.run(["ollama", "stop", model_name], check=True, capture_output=True, text=True)
            self.logger.info(f"Successfully unloaded Ollama model: {model_name}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Could not unload Ollama model {model_name}: {e.stderr}")
        except FileNotFoundError:
            self.logger.error("'ollama' command not found. Please ensure Ollama is installed and in your PATH.")

# Global instance
gpu_manager = GPUManager()
