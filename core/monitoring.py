import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Monitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {
            "total_processing_time": 0.0,
            "token_usage": {"input": 0, "output": 0},
            "memory_usage_gb": {"peak_ram": 0.0, "peak_vram": 0.0},
            "error_counts": {},
            "success_rates": {},
            "cost_estimates": {},
            "stage_times": {}
        }
        self.current_stage_start_time = None

    def start_stage(self, stage_name: str):
        self.current_stage_start_time = time.time()
        logger.info(f"Monitoring: Starting stage '{stage_name}'...")

    def end_stage(self, stage_name: str):
        if self.current_stage_start_time:
            duration = time.time() - self.current_stage_start_time
            self.metrics['stage_times'][stage_name] = duration
            logger.info(f"Monitoring: Stage '{stage_name}' completed in {duration:.2f} seconds.")
            self.current_stage_start_time = None

    def record_token_usage(self, input_tokens: int, output_tokens: int, model_name: str):
        self.metrics['token_usage']['input'] += input_tokens
        self.metrics['token_usage']['output'] += output_tokens
        logger.debug(f"Monitoring: Recorded token usage for {model_name}: Input={input_tokens}, Output={output_tokens}")

    def record_memory_usage(self, current_ram_gb: float, current_vram_gb: float):
        self.metrics['memory_usage_gb']['peak_ram'] = max(self.metrics['memory_usage_gb']['peak_ram'], current_ram_gb)
        self.metrics['memory_usage_gb']['peak_vram'] = max(self.metrics['memory_usage_gb']['peak_vram'], current_vram_gb)
        logger.debug(f"Monitoring: Current RAM={current_ram_gb:.2f}GB, VRAM={current_vram_gb:.2f}GB")

    def record_error(self, error_type: str, stage: str):
        key = f"{stage}_{error_type}"
        self.metrics['error_counts'][key] = self.metrics['error_counts'].get(key, 0) + 1
        logger.error(f"Monitoring: Recorded error '{error_type}' at stage '{stage}'.")

    def record_success_rate(self, metric_name: str, success_count: int, total_count: int):
        if total_count > 0:
            self.metrics['success_rates'][metric_name] = (success_count / total_count) * 100
        else:
            self.metrics['success_rates'][metric_name] = 0.0
        logger.debug(f"Monitoring: Recorded success rate for '{metric_name}': {self.metrics['success_rates'][metric_name]:.2f}%")

    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int, provider: str):
        # This is a simplified example. Real cost estimation requires up-to-date pricing.
        cost_per_input_token = 0.0
        cost_per_output_token = 0.0

        if provider == "gemini":
            # Example pricing (hypothetical, replace with actual)
            cost_per_input_token = 0.000001 # $0.001 per 1k tokens
            cost_per_output_token = 0.000002 # $0.002 per 1k tokens
        elif provider == "ollama":
            cost_per_input_token = 0.0 # Typically free for local models
            cost_per_output_token = 0.0
        
        input_cost = input_tokens * cost_per_input_token
        output_cost = output_tokens * cost_per_output_token
        total_cost = input_cost + output_cost

        self.metrics['cost_estimates'][model_name] = self.metrics['cost_estimates'].get(model_name, 0.0) + total_cost
        logger.debug(f"Monitoring: Estimated cost for {model_name}: ${total_cost:.6f}")

    def finalize_metrics(self):
        self.metrics['total_processing_time'] = time.time() - self.start_time
        logger.info(f"Monitoring: Total processing time: {self.metrics['total_processing_time']:.2f} seconds.")

    def get_report(self) -> Dict[str, Any]:
        return self.metrics

# Instantiate a global monitor
monitor = Monitor()
