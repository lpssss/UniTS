import os
import psutil
import time
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import argparse

class LiteRTProfiler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.process = psutil.Process(os.getpid())
        self.results = {}

    def _get_mem(self):
        # Returns current memory in MB
        return self.process.memory_info().rss / (1024 * 1024)

    def run_profile(self):
        # 1. Baseline
        self.results['baseline'] = self._get_mem()

        # 2. Model Loading
        self.interpreter = Interpreter(model_path=self.model_path)
        self.results['after_load'] = self._get_mem()

        # 3. Tensor Allocation (The Arena)
        self.interpreter.allocate_tensors()
        self.results['after_alloc'] = self._get_mem()

        # 4. Inference Warmup
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        input_data = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
        
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        self.results['after_inference'] = self._get_mem()

    def print_summary(self):
        print("\n" + "="*40)
        print("     LITERT MEMORY CONSUMPTION REPORT")
        print("="*40)
        
        # Calculate Deltas
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        arena_size = self.results['after_alloc'] - self.results['after_load']
        total_overhead = self.results['after_inference'] - self.results['baseline']

        print(f"{'Metric':<25} | {'Value (MB)':<10}")
        print("-" * 40)
        print(f"{'Model File Size':<25} | {model_size_mb:>10.2f}")
        print(f"{'Baseline RAM':<25} | {self.results['baseline']:>10.2f}")
        print(f"{'RAM after Load':<25} | {self.results['after_load']:>10.2f}")
        print(f"{'RAM after Allocation':<25} | {self.results['after_alloc']:>10.2f}")
        print(f"{'RAM after Inference':<25} | {self.results['after_inference']:>10.2f}")
        print("-" * 40)
        print(f"{'ESTIMATED ARENA SIZE':<25} | {arena_size:>10.2f}")
        print(f"{'TOTAL RUNTIME IMPACT':<25} | {total_overhead:>10.2f}")
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile memory usage of a TFLite model on Raspberry Pi using LITERT.")
    parser.add_argument('--tflite_path', type=str, required=True, help='Path to the TFLite model file')
    args = parser.parse_args()
    # Ensure your model file exists here
    profiler = LiteRTProfiler(args.tflite_path)
    profiler.run_profile()
    profiler.print_summary()